import os

import cv2
import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"

# from multiprocessing import Pool
import multiprocessing
from os import path as osp
from PIL import Image
from numpy import asarray
from tqdm import tqdm

from utils_sr.hparams import hparams
from utils_sr.indexed_datasets import IndexedDatasetBuilder
from utils_sr.matlab_resize import imresize

import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.checkpoint import DetectionCheckpointer

import torch
from transformers import AutoProcessor, Blip2ForConditionalGeneration,AutoTokenizer,PretrainedConfig
from Mask2Former.mask2former import add_maskformer2_config
from Mask2Former.train_net import Trainer
from utils11.seg_class import ADE20K_150_CATEGORIES

os.environ['TRANSFORMERS_CACHE'] = '/home/jianglei/work/SAM-DiffSR/cache'
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

def init_caption_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b").to(device)
    return processor,model

def init_instance_seg():
    cfg_ins = get_cfg()
    cfg_ins.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg_ins.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 设置预测阈值
    cfg_ins.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
    cfg_ins.MODEL.DEVICE = 'cuda'  # 使用 GPU
    instance_seg_model = DefaultPredictor(cfg_ins)
    return instance_seg_model

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
    text_encoder.cuda()
    return tokenizer, text_encoder

def init_semantic_seg():
    cfg_seg = get_cfg()
    add_deeplab_config(cfg_seg)
    add_maskformer2_config(cfg_seg)
    cfg_seg.merge_from_file("./preset/models/mask2former/config/ade20k-maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml")
    cfg_seg.MODEL.WEIGHTS = "./preset/models/mask2former/model_final_6b4a3a.pkl"
    cfg_seg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    semantic_seg_model = Trainer.build_model(cfg_seg)
    DetectionCheckpointer(semantic_seg_model).load(cfg_seg.MODEL.WEIGHTS)
    # semantic_seg_model.eval().to('cuda:0')
    
    semantic_seg_model.eval().cuda()
    return semantic_seg_model

def init_labels_embedding():
    ADE20k_NAMES = [k["name"] for k in ADE20K_150_CATEGORIES]
    for i, name in enumerate(ADE20k_NAMES):
        ADE20k_NAMES[i] = name.split(",")[0]
    labels_embedding_list = []
    for i, name in enumerate(ADE20k_NAMES):
        class_token = tokenizer(name, return_tensors="pt")
        class_token.input_ids = class_token.input_ids[0][1].unsqueeze(0) # only take the first token
        now_embedding = text_encoder(class_token.input_ids.cuda())[0].squeeze(0).view(-1)
        now_embedding.cuda().detach()
        labels_embedding_list.append(now_embedding)

    return labels_embedding_list

instance_seg_model = init_instance_seg()
caption_processor, caption_model = init_caption_model()
tokenizer, text_encoder = init_SD_model()

semantic_seg_model = init_semantic_seg()
labels_embedding_list = init_labels_embedding()

def worker(args):
    i, path, patch_size, crop_size, thresh_size, sr_scale = args
    img_name, extension = osp.splitext(osp.basename(path))
    img = Image.open(path).convert('RGB')
    img = asarray(img)
    h, w, c = img.shape
    h = h - h % sr_scale
    w = w - w % sr_scale
    img = img[:h, :w]
    h, w, c = img.shape
    img_lr = imresize(img, 1 / sr_scale)
    ret = []
    x = 0
    while x < h - thresh_size:
        y = 0
        while y < w - thresh_size:
            x_l_left = x // sr_scale
            x_l_right = (x + crop_size[0]) // sr_scale
            y_l_left = y // sr_scale
            y_l_right = (y + crop_size[1]) // sr_scale
            cropped_img = img[x:x + crop_size[0], y:y + crop_size[1], ...]
            cropped_img_lr = img_lr[x_l_left:x_l_right, y_l_left:y_l_right]
            ret.append({
                    'item_name': img_name,
                    'loc': [x // crop_size[0], y // crop_size[1]],
                    'loc_bdr': [(h + crop_size[0] - 1) // crop_size[0], (w + crop_size[1] - 1) // crop_size[1]],
                    'path': path, 'img': cropped_img,
                    'img_lr': cropped_img_lr,
            })
            y += crop_size[1]
        x += crop_size[0]
    
    return i, ret

def worker_sam(args):

    i, path, patch_size, crop_size, thresh_size, sr_scale, sam_dir = args
    img_name, extension = osp.splitext(osp.basename(path))
    sam_path = osp.join(sam_dir, f'{img_name}.npy')
    img = Image.open(path).convert('RGB')
    img = asarray(img)
    
    # 这里不是裁剪，只是修边，保证整除
    h, w, c = img.shape
    h = h - h % sr_scale
    w = w - w % sr_scale
    img = img[:h, :w]
    h, w, c = img.shape
    img_lr = imresize(img, 1 / sr_scale)
    
    # 这里就不应该try，要是sam_path错了也发现不了
    # try:
    #     sam_mask = np.load(sam_path)
    # except:
    #     sam_mask = np.zeros(img.shape[:2])

    sam_mask = np.load(sam_path)
    
    if sam_mask.shape != img.shape[:2]:
        sam_mask = cv2.resize(sam_mask, dsize=img.shape[:2][::-1])
    
    ret = []
    x = 0
    while x < h - thresh_size:
        y = 0
        while y < w - thresh_size:
            x_l_left = x // sr_scale
            x_l_right = (x + crop_size[0]) // sr_scale
            y_l_left = y // sr_scale
            y_l_right = (y + crop_size[1]) // sr_scale
            cropped_img = img[x:x + crop_size[0], y:y + crop_size[1], ...]
            cropped_img_lr = img_lr[x_l_left:x_l_right, y_l_left:y_l_right]
            cropped_sam_mask = sam_mask[x_l_left:x_l_right, y_l_left:y_l_right]

            h, w, c = cropped_img.shape
            h = h - h % (sr_scale * 2)
            w = w - w % (sr_scale * 2)
            h_l = h // sr_scale
            w_l = w // sr_scale
            cropped_img = cropped_img[:h, :w]
            cropped_sam_mask = cropped_sam_mask[:h, :w]
            cropped_img_lr = cropped_img_lr[:h_l, :w_l]

            patch_caption = []
            patch_instance_mask = []
            patch_label = []
            device = "cuda" if torch.cuda.is_available() else "cpu"
            num_query_token = 30
            # print(f'h:{h},w:{w}')
            for j in range(4):
                now_caption = []
                if j == 0 :
                    now_img = cropped_img[:patch_size , :patch_size]
                elif j == 1 :
                    now_img = cropped_img[:patch_size , w-patch_size:]
                elif j == 2 :
                    now_img = cropped_img[h-patch_size: , :patch_size]
                elif j == 3:
                    now_img = cropped_img[h-patch_size: , w-patch_size:]
                
                # print(f'now_img.shape:{now_img.shape},i:{i}')
                instance_output = instance_seg_model(now_img)
                instance_mask = instance_output["instances"].pred_masks.cpu()
                instance_box = instance_output["instances"].pred_boxes.tensor.cpu().numpy()
                instance_size = (patch_size,patch_size)

                if instance_mask.shape[0] != 0:
                    for m, mask in enumerate(instance_mask):
                        segmented_img = np.zeros_like(now_img, dtype=np.uint8)
                        segmented_img[mask] = now_img[mask]

                        x1, y1, x2, y2 = instance_box[m].astype(int)
                        cropped_img_ins = segmented_img[y1:y2, x1:x2]
                        resized_img = cv2.resize(cropped_img_ins, instance_size)

                        inputs = caption_processor(resized_img, return_tensors="pt").to(device, torch.float16)
                        generated_ids = caption_model.generate(**inputs, max_new_tokens=20)
                        generated_text = caption_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                        generated_text = generated_text.lower().replace('.', ',').rstrip(',')
                        print(generated_text)
                        class_token = tokenizer(generated_text,return_tensors="pt")

                        _, tokens_len = class_token.input_ids.shape
                        if tokens_len >= num_query_token:
                            final_tokens = class_token.input_ids[:,(tokens_len-num_query_token):tokens_len]
                        else:
                            last_token = class_token.input_ids[:,-1].unsqueeze(1)
                            final_tokens = torch.cat([class_token.input_ids,last_token.expand(1,num_query_token-tokens_len)],dim=1)
                        caption_embedding = text_encoder(final_tokens.cuda())
                        # 展成一个一维张量
                        caption_embedding = caption_embedding[0].squeeze(0).view(-1)    #获取到caption的clip embedding 
                        now_caption.append(caption_embedding.cpu())    
                
                patch_caption.append(now_caption)  
                patch_instance_mask.append(instance_mask.cpu()) 

                semantic_img = torch.from_numpy(now_img).permute(2,0,1).contiguous()
                semantic_img = [{'image' : (semantic_img)}]
                labels = semantic_seg_model(semantic_img)
                labels = torch.cat([label['sem_seg'].argmax(dim=0).unsqueeze(0) for label in labels], dim=0).squeeze(0)


            ret.append({  
                    'item_name': img_name,
                    'loc': [x // crop_size[0], y // crop_size[1]],
                    'loc_bdr': [(h + crop_size[0] - 1) // crop_size[0], (w + crop_size[1] - 1) // crop_size[1]],
                    'path': path, 'img': cropped_img,
                    'img_lr': cropped_img_lr,
                    'sam_mask': cropped_sam_mask,
                    'mask_path': sam_path,
                    'patch_caption' : patch_caption, 'patch_instance_mask' : patch_instance_mask
            })
            y += crop_size[1]
        x += crop_size[0]
    return i, ret

def build_bin_dataset(paths, binary_data_dir, prefix, patch_size, crop_size, thresh_size):
    if isinstance(crop_size, int):
        crop_size = [crop_size, crop_size]
    sr_scale = hparams['sr_scale']
    assert crop_size[0] % sr_scale == 0
    assert crop_size[1] % sr_scale == 0
    assert patch_size % sr_scale == 0
    assert thresh_size % sr_scale == 0
    
    builder = IndexedDatasetBuilder(f'{binary_data_dir}/{prefix}')
    
    def get_worker_args():
        for i, path in enumerate(paths):
            yield i, path, patch_size, crop_size, thresh_size, sr_scale
    
    with multiprocessing.Pool(processes=1) as pool:
        for ret in tqdm(pool.imap_unordered(worker, list(get_worker_args())), total=len(paths)):
            if 'test' in prefix:
                builder.add_item(ret[1][0], id=ret[0])
            else:
                for r in ret[1]:
                    builder.add_item(r)
    builder.finalize()

def build_bin_dataset_sam(paths, binary_data_dir, prefix, patch_size, crop_size, thresh_size, sam_dir):
    
    if isinstance(crop_size, int):
        crop_size = [crop_size, crop_size]
    sr_scale = hparams['sr_scale']
    assert crop_size[0] % sr_scale == 0
    assert crop_size[1] % sr_scale == 0
    assert patch_size % sr_scale == 0
    assert thresh_size % sr_scale == 0
    
    builder = IndexedDatasetBuilder(f'{binary_data_dir}/{prefix}')
    
    def get_worker_args():
        for i, path in enumerate(paths):
            yield i, path, patch_size, crop_size, thresh_size, sr_scale, sam_dir
    
    # multiprocessing.set_start_method('spawn', force=True)
    # with multiprocessing.Pool(processes=1) as pool:
    #     for ret in tqdm(pool.imap_unordered(worker_sam, list(get_worker_args())), total=len(paths)):
    #         if 'test' in prefix:
    #             builder.add_item(ret[1][0], id=ret[0])
    #         else:
    #             for r in ret[1]:
    #                 builder.add_item(r)
    # builder.finalize()
    # 不使用 multiprocessing 的版本
    for args in tqdm(get_worker_args(), total=len(paths)):
        ret = worker_sam(args)
        if 'test' in prefix:
            builder.add_item(ret[1][0], id=ret[0])
        else:
            for r in ret[1]:
                builder.add_item(r)

    builder.finalize()

