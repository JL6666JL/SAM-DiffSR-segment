import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.checkpoint import DetectionCheckpointer
# ade20k_metadata = MetadataCatalog.get("ade20k_sem_seg_val")

from Mask2Former.mask2former import add_maskformer2_config
from Mask2Former.train_net import Trainer
import torch
import cv2
from torchvision.ops import nms
from utils.seg_class import ADE20K_150_CATEGORIES
import numpy as np

cfg_seg = get_cfg()
add_deeplab_config(cfg_seg)
add_maskformer2_config(cfg_seg)
cfg_seg.merge_from_file("./preset/models/mask2former/config/ade20k-maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml")
cfg_seg.MODEL.WEIGHTS = "./preset/models/mask2former/model_final_6b4a3a.pkl"
cfg_seg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True

seg_model = Trainer.build_model(cfg_seg)
DetectionCheckpointer(seg_model).load(cfg_seg.MODEL.WEIGHTS)
seg_model.eval().to('cuda:0')

cfg_ins = get_cfg()
cfg_ins.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg_ins.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 设置预测阈值
cfg_ins.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
ins_model = DefaultPredictor(cfg_ins)

# 1. 使用OpenCV加载图片
image = cv2.imread('/home/jianglei/work/SAM-DiffSR/ball_dog.png')  # 加载为 BGR 格式

# 2. 转换为 RGB 格式
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
outputs = ins_model(image)

# 3. 获取分割掩码、边界框和置信度
masks = outputs["instances"].pred_masks.cpu().numpy()
boxes = outputs["instances"].pred_boxes.tensor.cpu()
scores = outputs["instances"].scores.cpu()
classes = outputs["instances"].pred_classes.cpu().numpy()  # 类别索引

# 仅保留置信度大于 0.7 的检测结果
confidence_threshold = 0.7
high_conf_indices = scores > confidence_threshold
masks = masks[high_conf_indices.numpy()]
boxes = boxes[high_conf_indices]
scores = scores[high_conf_indices]
classes = classes[high_conf_indices.numpy()]

# 4. 应用非极大值抑制，设定重叠阈值
iou_threshold = 0.7  # IOU 阈值（重叠度）
keep = nms(boxes, scores, iou_threshold)  # 返回保留的框索引

# 5. 保留 NMS 后的分割掩码和边界框
selected_masks = masks[keep.numpy()]
selected_boxes = boxes[keep].numpy()
selected_classes = classes[keep.numpy()]
caption = 10000
captions = []

# 6. 提取每个分割区域并保存
output_size = (512, 512)  # 统一大小
for i, mask in enumerate(selected_masks):
# 创建一个黑色的图像（只包含 RGB 三通道）
    segmented_img = np.zeros_like(image, dtype=np.uint8)
    
    # 在分割区域内保留图像内容，其他区域为黑色
    segmented_img[mask] = image[mask]
    
    x1, y1, x2, y2 = selected_boxes[i].astype(int)
    if x1 < x2 and y1 < y2 and (x2 - x1) > 0 and (y2 - y1) > 0:
        cropped_img = segmented_img[y1:y2, x1:x2]
        resized_img = cv2.resize(cropped_img, output_size)
        # 下一步就是描述，得到描述的list
        captions.append((caption))
        caption = caption+100



# 3. 将图片转换为张量并添加批量维度
image_tensor = torch.tensor(image).float()  # 将图像转换为 float 张量
image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # 调整形状为 (N, C, H, W)
lr_up_seg = [{'image': (img)} for img in image_tensor]

# print(image_tensor.shape)  # 输出: torch.Size([1, 3, H, W])
labels = seg_model(lr_up_seg)


print(type(labels[0]['sem_seg']))
print(labels[0]['sem_seg'].shape)
arglabels = labels[0]['sem_seg'].argmax(dim=0).unsqueeze(0)
print(arglabels.shape)
print(torch.unique(arglabels))

classes = outputs["instances"].pred_classes.cpu().numpy()
masks = outputs["instances"].pred_masks.cpu().numpy()
print(classes)
print(masks.shape)
print(masks[0])


final_mask = arglabels[0]
for i,mask in  enumerate(selected_masks):
    final_mask[mask] = captions[i]

final_mask_numpy = final_mask.cpu().numpy()
with open('test.txt','w') as f:
    for row in final_mask_numpy:
        f.write(' '.join(map(str,row))+'\n')

# 为每个随机数分配一个随机颜色
random_colors = {}
unique_values = final_mask.unique().tolist()

# 生成随机颜色
for value in unique_values:
    random_colors[value] = np.random.randint(0, 256, 3)  # 生成 RGB 随机颜色

H,W = final_mask.shape

for i in range(H):
    for j in range(W):
        value = final_mask[i, j].item()  # 获取当前随机数
        image[i, j] = random_colors[value]  # 设置对应位置的颜色

cv2.imwrite('random_colored_image.png', image)