import os
import random

import cv2
import numpy as np
import torch
from PIL import Image
from rotary_embedding_torch import RotaryEmbedding
from torchvision import transforms

from models_sr.diffsr_modules import RRDBNet, Unet
from models_sr.diffusion_sam import GaussianDiffusion_sam
from tasks.srdiff import SRDiffTrainer
from utils_sr.dataset import SRDataSet
from utils_sr.hparams import hparams
from utils_sr.indexed_datasets import IndexedDataset
from utils_sr.matlab_resize import imresize
from utils_sr.utils import load_ckpt

from torch import nn


def normalize_01(data):
    mu = np.mean(data)
    sigma = np.std(data)
    
    if sigma == 0.:
        return data - mu
    else:
        return (data - mu) / sigma


def normalize_11(data):
    mu = np.mean(data)
    sigma = np.std(data)
    
    if sigma == 0.:
        return data - mu
    else:
        return (data - mu) / sigma - 1


class Df2kDataSet_sam(SRDataSet):
    def __init__(self, prefix='train'):
        
        if prefix == 'valid':
            _prefix = 'test'
        else:
            _prefix = prefix
        
        super().__init__(_prefix)
        
        self.patch_size = hparams['patch_size']
        self.patch_size_lr = hparams['patch_size'] // hparams['sr_scale']
        if prefix == 'valid':
            self.len = hparams['eval_batch_size'] * hparams['valid_steps']
        
        self.data_position_aug_transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20, interpolation=Image.BICUBIC),
        ])
        
        self.data_color_aug_transforms = transforms.Compose([
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        ])
        
        self.sam_config = hparams.get('sam_config', False)
        
        if self.sam_config.get('mask_RoPE', False):
            h, w = map(int, self.sam_config['mask_RoPE_shape'].split('-'))
            rotary_emb = RotaryEmbedding(dim=h)
            sam_mask = rotary_emb.rotate_queries_or_keys(torch.ones(1, 1, w, h))
            self.RoPE_mask = sam_mask.cpu().numpy()[0, 0, ...]
    
    def _get_item(self, index):
        if self.indexed_ds is None:
            self.indexed_ds = IndexedDataset(f'{self.data_dir}/{self.prefix}')
        return self.indexed_ds[index]
    
    def __getitem__(self, index):
        item = self._get_item(index)
        hparams = self.hparams
        sr_scale = hparams['sr_scale']
        
        img_hr = np.uint8(item['img'])
        img_lr = np.uint8(item['img_lr'])
        if 'patch_caption' in item:
            patch_caption = item['patch_caption']
        else:
            patch_caption = torch.zeros(30720)
            caption_num = int(0)
        if 'patch_instance_mask' in item:
            patch_instance_mask = item['patch_instance_mask']
        else:
            patch_instance_mask = torch.zeros(30, 160, 160)
            caption_num = int(0)

        # patch_caption = item['patch_caption']
        # patch_instance_mask = item['patch_instance_mask']
        
        if self.sam_config.get('mask_RoPE', False):
            sam_mask = self.RoPE_mask
        else:
            if 'sam_mask' in item:
                sam_mask = item['sam_mask']
                if sam_mask.shape != img_hr.shape[:2]:
                    sam_mask = cv2.resize(sam_mask, dsize=img_hr.shape[:2][::-1])
            else:
                sam_mask = np.zeros_like(img_lr)
        
        # TODO: clip for SRFlow
        h, w, c = img_hr.shape
        h = h - h % (sr_scale * 2)
        w = w - w % (sr_scale * 2)
        h_l = h // sr_scale
        w_l = w // sr_scale
        img_hr = img_hr[:h, :w]
        sam_mask = sam_mask[:h, :w]
        img_lr = img_lr[:h_l, :w_l]

        # random crop
        if self.prefix == 'train':
            if self.data_augmentation and random.random() < 0.5:
                img_hr, img_lr, sam_mask = self.data_augment(img_hr, img_lr, sam_mask)
            # i = random.randint(0, h - self.patch_size) // sr_scale * sr_scale
            # i_lr = i // sr_scale
            # j = random.randint(0, w - self.patch_size) // sr_scale * sr_scale
            # j_lr = j // sr_scale
            # img_hr = img_hr[i:i + self.patch_size, j:j + self.patch_size]
            # sam_mask = sam_mask[i:i + self.patch_size, j:j + self.patch_size]
            # img_lr = img_lr[i_lr:i_lr + self.patch_size_lr, j_lr:j_lr + self.patch_size_lr]

            random_number = random.choice([0, 1, 2, 3])
            patch_caption = patch_caption[random_number]
            patch_instance_mask = patch_instance_mask[random_number]
            caption_num = len(patch_caption)
            patch_caption, patch_instance_mask, caption_num = self.pad_caption_mask(patch_caption=patch_caption,
                                                                       patch_instance_mask=patch_instance_mask,
                                                                       caption_num=caption_num)
            
            patch_caption = torch.stack(patch_caption)
            # print(patch_caption.shape, patch_instance_mask.shape)

            if random_number == 0:
                img_hr = img_hr[:self.patch_size, :self.patch_size]
                sam_mask = sam_mask[:self.patch_size, :self.patch_size]
                img_lr = img_lr[:self.patch_size_lr, :self.patch_size_lr]
            elif random_number == 1:
                img_hr = img_hr[:self.patch_size, w-self.patch_size:]
                sam_mask = sam_mask[:self.patch_size, w-self.patch_size:]
                img_lr = img_lr[:self.patch_size_lr, w_l-self.patch_size_lr:]
            elif random_number == 2:
                img_hr = img_hr[h-self.patch_size: , :self.patch_size]
                sam_mask = sam_mask[h-self.patch_size: , :self.patch_size]
                img_lr = img_lr[h_l-self.patch_size_lr: , :self.patch_size_lr]
            elif random_number == 3:
                img_hr = img_hr[h-self.patch_size:, w-self.patch_size:]
                sam_mask = sam_mask[h-self.patch_size:, w-self.patch_size:]
                img_lr = img_lr[h_l-self.patch_size_lr:, w_l-self.patch_size_lr:]
        
        # print(np.array(img_hr).shape)
        # print(np.array(img_lr).shape)ls
        # input('debug')
        
        img_lr_up_255 = imresize(img_lr, hparams['sr_scale'])
        img_lr_up = imresize(img_lr / 256, hparams['sr_scale'])  # np.float [H, W, C]

        img_hr, img_lr, img_lr_up = [self.to_tensor_norm(x).float() for x in [img_hr, img_lr, img_lr_up]]

        
        if hparams['sam_data_config']['all_same_mask_to_zero']:
            if len(np.unique(sam_mask)) == 1:
                sam_mask = np.zeros_like(sam_mask)
        
        if hparams['sam_data_config']['normalize_01']:
            if len(np.unique(sam_mask)) != 1:
                sam_mask = normalize_01(sam_mask)
        
        if hparams['sam_data_config']['normalize_11']:
            if len(np.unique(sam_mask)) != 1:
                sam_mask = normalize_11(sam_mask)
        
        sam_mask = torch.FloatTensor(sam_mask).unsqueeze(dim=0)
        
        

        return {
                'img_hr': img_hr, 'img_lr': img_lr,
                'img_lr_up': img_lr_up, 'img_lr_up_255':img_lr_up_255,'item_name': item['item_name'],
                'loc': np.array(item['loc']), 'loc_bdr': np.array(item['loc_bdr']),
                'sam_mask': sam_mask, 'patch_caption': patch_caption,'patch_instance_mask': patch_instance_mask,
                'caption_num': caption_num
        }
        # return {
        #     'img_hr': img_hr, 'img_lr': img_lr,
        #     'img_lr_up': img_lr_up, 'img_lr_up_255':img_lr_up_255,'item_name': item['item_name'],
        #     'loc': np.array(item['loc']), 'loc_bdr': np.array(item['loc_bdr']),
        #     'sam_mask': sam_mask
        # }
    
    def __len__(self):
        return self.len
    
    def data_augment(self, img_hr, img_lr, sam_mask):
        sr_scale = self.hparams['sr_scale']
        img_hr = Image.fromarray(img_hr)
        img_hr, sam_mask = self.data_position_aug_transforms([img_hr, sam_mask])
        img_hr = self.data_color_aug_transforms(img_hr)
        img_hr = np.asarray(img_hr)  # np.uint8 [H, W, C]
        img_lr = imresize(img_hr, 1 / sr_scale)
        return img_hr, img_lr, sam_mask

    def pad_caption_mask(self,patch_caption, patch_instance_mask, caption_num):
        current_length, h ,w = patch_instance_mask.shape
        target_length = 30
        caption_num = caption_num

        if current_length == 0 :
            patch_caption = [torch.zeros(30720)] * target_length
            patch_instance_mask = torch.zeros((target_length,h,w),dtype=torch.bool)

        else:
            padding_length = target_length - current_length
            if padding_length > 0 :
                patch_caption.extend([patch_caption[0]]* padding_length)

                padding_tensor = torch.zeros((padding_length,h,w),dtype=torch.bool)
                patch_instance_mask = torch.cat([patch_instance_mask, padding_tensor],dim=0)
            else:
                patch_caption = patch_caption[:target_length]
                patch_instance_mask = patch_instance_mask[:target_length]
                caption_num = target_length
        
        return patch_caption, patch_instance_mask, caption_num

class SRDiffDf2k_sam(SRDiffTrainer):
    def __init__(self):
        super().__init__()
        self.dataset_cls = Df2kDataSet_sam
        self.sam_config = hparams['sam_config']
    
    def build_model(self):
        hidden_size = hparams['hidden_size']
        dim_mults = hparams['unet_dim_mults']
        dim_mults = [int(x) for x in dim_mults.split('|')]
        
        denoise_fn = Unet(
                hidden_size, out_dim=3, cond_dim=hparams['rrdb_num_feat'], dim_mults=dim_mults)
        if hparams['use_rrdb']:
            rrdb = RRDBNet(3, 3, hparams['rrdb_num_feat'], hparams['rrdb_num_block'],
                           hparams['rrdb_num_feat'] // 2)
            if hparams['rrdb_ckpt'] != '' and os.path.exists(hparams['rrdb_ckpt']):
                load_ckpt(rrdb, hparams['rrdb_ckpt'])
        else:
            rrdb = None
        self.model = GaussianDiffusion_sam(
                denoise_fn=denoise_fn,
                rrdb_net=rrdb,
                timesteps=hparams['timesteps'],
                loss_type=hparams['loss_type'],
                sam_config=hparams['sam_config']
        )
        # if torch.cuda.device_count() > 1:
        #     print("Using 2 GPUs for training")
        #     self.model = nn.DataParallel(self.model, device_ids=[0, 1])  # 指定使用 GPU 0 和 GPU 
        # self.model.cuda()
        self.global_step = 0
        return self.model
    
    # def sample_and_test(self, sample):
    #     ret = {k: 0 for k in self.metric_keys}
    #     ret['n_samples'] = 0
    #     img_hr = sample['img_hr']
    #     img_lr = sample['img_lr']
    #     img_lr_up = sample['img_lr_up']
    #     sam_mask = sample['sam_mask']
    #
    #     img_sr, rrdb_out = self.model.sample(img_lr, img_lr_up, img_hr.shape, sam_mask=sam_mask)
    #
    #     for b in range(img_sr.shape[0]):
    #         s = self.measure.measure(img_sr[b], img_hr[b], img_lr[b], hparams['sr_scale'])
    #         ret['psnr'] += s['psnr']
    #         ret['ssim'] += s['ssim']
    #         ret['lpips'] += s['lpips']
    #         ret['lr_psnr'] += s['lr_psnr']
    #         ret['n_samples'] += 1
    #     return img_sr, rrdb_out, ret
    
    def training_step(self, batch):
        img_hr = batch['img_hr']
        img_lr = batch['img_lr']
        img_lr_up = batch['img_lr_up']
        img_lr_up_255 = batch['img_lr_up_255']
        sam_mask = batch['sam_mask']
        caption_num = batch['caption_num']
        patch_caption = batch['patch_caption']
        patch_instance_mask = batch['patch_instance_mask']
        losses, _, _ = self.model(img_hr, img_lr, img_lr_up, img_lr_up_255, caption_num, patch_caption, patch_instance_mask,sam_mask=sam_mask)
        total_loss = sum(losses.values())
        return losses, total_loss
