import torch
import torch.nn.functional as F
from thop import profile
from tqdm import tqdm

from utils_sr.hparams import hparams
from .diffusion import GaussianDiffusion, noise_like, extract
from .module_util import default

import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.checkpoint import DetectionCheckpointer

from Mask2Former.mask2former import add_maskformer2_config
from Mask2Former.train_net import Trainer
import cv2
from torchvision.ops import nms
from PIL import Image
from torchvision import transforms
import numpy as np
from transformers import AutoProcessor, Blip2ForConditionalGeneration,AutoTokenizer,PretrainedConfig
from utils.seg_class import ADE20K_150_CATEGORIES
import torch.nn as nn

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")

def init_semantic_seg():
    cfg_seg = get_cfg()
    add_deeplab_config(cfg_seg)
    add_maskformer2_config(cfg_seg)
    cfg_seg.merge_from_file("./preset/models/mask2former/config/ade20k-maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml")
    cfg_seg.MODEL.WEIGHTS = "./preset/models/mask2former/model_final_6b4a3a.pkl"
    cfg_seg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    semantic_seg_model = Trainer.build_model(cfg_seg)
    DetectionCheckpointer(semantic_seg_model).load(cfg_seg.MODEL.WEIGHTS)
    semantic_seg_model.eval().to('cuda:0')
    return semantic_seg_model

def init_instance_seg():
    cfg_ins = get_cfg()
    cfg_ins.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg_ins.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 设置预测阈值
    cfg_ins.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
    instance_seg_model = DefaultPredictor(cfg_ins)
    return instance_seg_model

def init_caption_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b").to(device)
    return processor,model

def init_SD_model():
    pretrained_model_name_or_path = "preset/models/stable-diffusion-2-1-base"
    revision = None
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=revision,
        use_fast=False,
    )
    text_encoder_cls = import_model_class_from_model_name_or_path(pretrained_model_name_or_path, revision)
    text_encoder = text_encoder_cls.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder", revision=revision
    )
    text_encoder.requires_grad_(False)
    text_encoder.to("cuda:0")
    return tokenizer, text_encoder

def init_labels_embedding():
    ADE20k_NAMES = [k["name"] for k in ADE20K_150_CATEGORIES]
    for i, name in enumerate(ADE20k_NAMES):
        ADE20k_NAMES[i] = name.split(",")[0]
    labels_embedding_list = []
    for i, name in enumerate(ADE20k_NAMES):
        class_token = tokenizer(name, return_tensors="pt")
        class_token.input_ids = class_token.input_ids[0][1].unsqueeze(0) # only take the first token
        now_embedding = text_encoder(class_token.input_ids.to("cuda:0"))[0].squeeze(0).view(-1)
        labels_embedding_list.append(now_embedding)

    return labels_embedding_list

semantic_seg_model = init_semantic_seg()
instance_seg_model = init_instance_seg()
caption_processor, caption_model = init_caption_model()
tokenizer, text_encoder = init_SD_model()
labels_embedding_list = init_labels_embedding()

def get_flops(model, inputs):
    flops, params = profile(model, inputs=inputs)
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')

class GaussianDiffusion_sam(GaussianDiffusion):
    def __init__(self, denoise_fn, rrdb_net, timesteps=1000, loss_type='l1', sam_config=None):
        super().__init__(denoise_fn, rrdb_net, timesteps, loss_type)
        self.sam_config = sam_config
        
        self.num_query_token = 30
        labels_embedding_dim = 1024
        caption_embedding_dim = 30720
        hidden_dim = 100
        self.labels_mlp1 = nn.Linear(in_features=labels_embedding_dim ,out_features=hidden_dim)
        self.labels_ac1 = nn.ReLU()
        self.labels_mlp2 = nn.Linear(in_features=hidden_dim ,out_features=1)
        self.labels_ac2 = nn.ReLU()

        self.caption_mlp1 = nn.Linear(in_features=caption_embedding_dim, out_features=hidden_dim)
        self.caption_ac1 = nn.ReLU()
        self.caption_mlp2 = nn.Linear(in_features=hidden_dim, out_features=1)
        self.caption_ac2 = nn.ReLU()

    def p_losses(self, x_start, t, cond, img_lr_up, img_lr_up_255,noise=None, sam_mask=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        caption_mask = self.get_caption_mask(img_lr_up_255)
        _caption_mask = caption_mask.unsqueeze(1)

        if self.sam_config['p_losses_sam']:
            _sam_mask = F.interpolate(sam_mask, noise.shape[2:], mode='bilinear')
            if self.sam_config.get('mask_coefficient', False):
                _sam_mask *= extract(self.mask_coefficient.to(_sam_mask.device), t, x_start.shape)
                _caption_mask *= extract(self.mask_coefficient.to(_caption_mask.device), t, x_start.shape)
            # sam_mask在batch里面都处理好了，这里没有再处理了
            noise += _sam_mask  # _sam_mask是什么格式的?
            noise += _caption_mask
        
        x_tp1_gt = self.q_sample(x_start=x_start, t=t, noise=noise) # x_(t+1)
        x_t_gt = self.q_sample(x_start=x_start, t=t - 1, noise=noise)   # x_t

        # 这两行其实都没用上sam_mask
        noise_pred = self.denoise_fn(x_tp1_gt, t, cond, img_lr_up, sam_mask=sam_mask)
        x_t_pred, x0_pred = self.p_sample(x_tp1_gt, t, cond, img_lr_up, noise_pred=noise_pred, sam_mask=sam_mask)
        
        if self.loss_type == 'l1':
            loss = (noise - noise_pred).abs().mean()
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, noise_pred)
        elif self.loss_type == 'ssim':
            loss = (noise - noise_pred).abs().mean()
            loss = loss + (1 - self.ssim_loss(noise, noise_pred))
        else:
            raise NotImplementedError()
        return loss, x_tp1_gt, noise_pred, x_t_pred, x_t_gt, x0_pred
    
    @torch.no_grad()
    def p_sample(self, x, t, cond, img_lr_up, noise_pred=None, clip_denoised=True, repeat_noise=False, sam_mask=None):
        if noise_pred is None:
            noise_pred = self.denoise_fn(x, t, cond=cond, img_lr_up=img_lr_up, sam_mask=sam_mask)
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance, x0_pred = self.p_mean_variance(
                x=x, t=t, noise_pred=noise_pred, clip_denoised=clip_denoised)
        
        noise = noise_like(x.shape, device, repeat_noise)
        
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0_pred
    
    @torch.no_grad()
    def sample(self, img_lr, img_lr_up, shape, sam_mask=None, save_intermediate=False):
        device = self.betas.device
        b = shape[0]
        
        if not hparams['res']:
            t = torch.full((b,), self.num_timesteps - 1, device=device, dtype=torch.long)
            noise = None
            img = self.q_sample(img_lr_up, t, noise=noise)
        else:
            img = torch.randn(shape, device=device)
        
        if hparams['use_rrdb']:
            rrdb_out, cond = self.rrdb(img_lr, True)
        else:
            rrdb_out = img_lr_up
            cond = img_lr
        
        it = reversed(range(0, self.num_timesteps))
        
        if self.sample_tqdm:
            it = tqdm(it, desc='sampling loop time step', total=self.num_timesteps)
        
        images = []
        for i in it:
            img, x_recon = self.p_sample(
                    img, torch.full((b,), i, device=device, dtype=torch.long), cond, img_lr_up, sam_mask=sam_mask)
            if save_intermediate:
                img_ = self.res2img(img, img_lr_up)
                x_recon_ = self.res2img(x_recon, img_lr_up)
                images.append((img_.cpu(), x_recon_.cpu()))
        img = self.res2img(img, img_lr_up)
        
        if save_intermediate:
            return img, rrdb_out, images
        else:
            return img, rrdb_out

    def get_caption_mask(self,img_lr_up_255):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        N,H,W,C = img_lr_up_255.shape
        merge_mask = torch.zeros((N,H,W)).to("cuda:0")

        # 语义分割部分获得标签
        semantic_img = img_lr_up_255.float()
        semantic_img = semantic_img.permute(0,3,1,2)
        semantic_img = [{'image' : (img)} for img in semantic_img]
        labels = semantic_seg_model(semantic_img)
        labels = torch.cat([label['sem_seg'].argmax(dim=0).unsqueeze(0) for label in labels], dim=0)


        # 实例分割部分获得精确描述
        instance_img = img_lr_up_255.cpu().numpy()
        instance_img = instance_img.astype(np.uint8)
        for i,img in enumerate(instance_img):
            instance_output = instance_seg_model(img)
            instance_mask = instance_output["instances"].pred_masks.cpu()
            instance_box = instance_output["instances"].pred_boxes.tensor.cpu().numpy()
            caption_list = []
            instance_size=(160,160) #裁剪之后img_lr_up的大小为(160,160)
            # 如果有实例分割的结果
            if instance_mask.shape[0] != 0:
                for i, mask in enumerate(instance_mask):
                    segmented_img = np.zeros_like(img, dtype=np.uint8)
                    segmented_img[mask] = img[mask]

                    x1, y1, x2, y2 = instance_box[i].astype(int)
                    if x1 < x2 and y1 < y2 and (x2 - x1) > 0 and (y2 - y1) > 0:
                        cropped_img = segmented_img[y1:y2, x1:x2]
                        resized_img = cv2.resize(cropped_img, instance_size)
                        
                        inputs = caption_processor(resized_img, return_tensors="pt").to(device, torch.float16)
                        generated_ids = caption_model.generate(**inputs, max_new_tokens=20)
                        generated_text = caption_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                        generated_text = generated_text.lower().replace('.', ',').rstrip(',')
                        class_token = tokenizer(generated_text,return_tensors="pt")

                        _, tokens_len = class_token.input_ids.shape
                        if tokens_len >= self.num_query_token:
                            final_tokens = class_token.input_ids[:,(tokens_len-self.num_query_token):tokens_len]
                        else:
                            last_token = class_token.input_ids[:,-1].unsqueeze(1)
                            final_tokens = torch.cat([class_token.input_ids,last_token.expand(1,self.num_query_token-tokens_len)],dim=1)
                        caption_embedding = text_encoder(final_tokens.to("cuda:0"))
                        # 展成一个一维张量
                        caption_embedding = caption_embedding[0].squeeze(0).view(-1)    #获取到caption的clip embedding

                        caption_num = self.get_caption_num(caption_embedding)
                        caption_list.append(caption_num)
            
            # 把labels的编码的num写入mask
            for j in torch.unique(labels[i]):
                merge_mask[i][labels[i]==j] = self.get_labels_num(labels_embedding_list[j])

            for j in range(len(caption_list)):
                merge_mask[i][instance_mask[j]] = caption_list[j]
        
        return merge_mask

    def get_labels_num(self,labels_embedding):
        emb = self.labels_mlp1(labels_embedding)
        emb = self.labels_ac1(emb)
        emb = self.labels_mlp2(emb)
        emb = self.labels_ac2(emb)
        return emb
    
    def get_caption_num(self,caption_embedding):
        emb = self.caption_mlp1(caption_embedding)
        emb = self.caption_ac1(emb)
        emb = self.caption_mlp2(emb)
        emb = self.caption_ac2(emb)
        return emb