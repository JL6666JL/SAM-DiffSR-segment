import os
import sys

if __name__ == '__main__':
    pythonpath = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    sys.path.insert(0,pythonpath)
    from data_gen.utils import build_bin_dataset_sam, build_bin_dataset
    from utils_sr.hparams import hparams, set_hparams
    set_hparams()
    
    train_img_dir = '/data1/jianglei/DF2K/DF2K/DF2K_train_HR'
    test_img_dir = '/data1/jianglei/DF2K/DIV2K/DIV2K_valid_HR'
    
    train_sam_embed_dir = '/data1/jianglei/DF2K/sam_embed/DF2K/DF2K_train_HR'
    
    binary_data_dir = hparams['binary_data_dir']
    input(binary_data_dir)
    os.makedirs(binary_data_dir, exist_ok=True)
    
    train_img_list = sorted(os.path.join(train_img_dir, filename) for filename in os.listdir(train_img_dir))
    test_img_list = sorted(os.path.join(test_img_dir, filename) for filename in os.listdir(test_img_dir))

    crop_size = hparams['crop_size']
    patch_size = hparams['patch_size']
    thresh_size = hparams['thresh_size']
    test_crop_size = hparams['test_crop_size']
    test_thresh_size = hparams['test_thresh_size']
    
    build_bin_dataset_sam(train_img_list, binary_data_dir, 'train', patch_size, crop_size, thresh_size,
                          train_sam_embed_dir)
    
    build_bin_dataset(test_img_list, binary_data_dir, 'test', patch_size, crop_size, thresh_size)
