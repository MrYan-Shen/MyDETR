# modified from https://github.com/anhtuan85/Data-Augmentation-for-Object-Detection/blob/master/augmentation.ipynb

import PIL #version 1.2.0
from PIL import Image #version 6.1.0
import torch
import os
import torchvision.transforms.functional as F
import numpy as np
import random

from .random_crop import random_crop
from util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh

class AdjustContrast:
    """动态调整图像对比度，保持边界框不变

    设计特点：
    - 随机因子：在[0.5×factor, factor]范围内随机选择对比度调整强度
    - 无状态操作：不修改边界框，仅改变像素值
    - 保留目标：适用于增强低对比度图像中的目标可见性

    使用示例：
    transform = AdjustContrast(contrast_factor=1.5)
    img, target = transform(img, target)
    """
    def __init__(self, contrast_factor):
        self.contrast_factor = contrast_factor

    def __call__(self, img, target):
        """
        img (PIL Image or Tensor): Image to be adjusted.
        """
        _contrast_factor = ((random.random() + 1.0) / 2.0) * self.contrast_factor
        img = F.adjust_contrast(img, _contrast_factor)
        return img, target

class AdjustBrightness:
    """动态调整图像亮度，保持边界框不变

    设计理念：
    - 与AdjustContrast对称设计，保持API一致性
    - 随机亮度调整增强模型对光照变化的鲁棒性
    - 不影响边界框坐标，仅修改像素强度

    典型应用场景：
    - 模拟不同光照条件（黎明、黄昏、阴影）
    - 增强低光照图像中的目标可见性
    """
    def __init__(self, brightness_factor):
        self.brightness_factor = brightness_factor

    def __call__(self, img, target):
        """
        img (PIL Image or Tensor): Image to be adjusted.
        """
        _brightness_factor = ((random.random() + 1.0) / 2.0) * self.brightness_factor
        img = F.adjust_brightness(img, _brightness_factor)
        return img, target

def lighting_noise(image):
    '''
        color channel swap in image
        image: A PIL image

    颜色通道随机交换，模拟不同光照条件下的色彩变化
    参数：
    image: PIL图像对象

    返回：
    new_image: 颜色通道被随机重排的PIL图像

    算法原理：
    1. 定义所有可能的RGB通道排列组合（6种）
    2. 随机选择一种排列
    3. 将图像转换为张量
    4. 按照选定排列重排通道维度
    5. 转换回PIL图像

    通道排列示例：
    (0,1,2) -> RGB (不变)
    (0,2,1) -> RBG
    (1,0,2) -> GRB
    (1,2,0) -> GBR
    (2,0,1) -> BRG
    (2,1,0) -> BGR

    设计考虑：
    - 保持像素值范围不变，仅改变通道顺序
    - 适用于增强模型对色彩变化的鲁棒性
    - 不同于白平衡调整，这是一种更极端的增强
    '''
    new_image = image
    perms = ((0, 1, 2), (0, 2, 1), (1, 0, 2), 
             (1, 2, 0), (2, 0, 1), (2, 1, 0))
    swap = perms[random.randint(0, len(perms)- 1)]
    new_image = F.to_tensor(new_image)
    new_image = new_image[swap, :, :]
    new_image = F.to_pil_image(new_image)
    return new_image

class LightingNoise:
    """封装lighting_noise函数，提供标准接口

    设计特点：
    - 无参初始化，使用默认行为
    - 保持与PyTorch变换一致的接口
    - 不修改边界框，仅改变图像色彩分布

    使用场景：
    - 增强模型对不同光照条件的鲁棒性
    - 模拟相机白平衡误差
    - 防止模型过度依赖特定颜色特征
    """
    def __init__(self) -> None:
        pass

    def __call__(self, img, target):
        return lighting_noise(img), target


def rotate(image, boxes, angle):
    '''
        Rotate image and bounding box
        image: A Pil image (w, h)
        boxes: A tensors of dimensions (#objects, 4)
        
        Out: rotated image (w, h), rotated boxes

    旋转图像和对应的边界框，保持目标完整性
    参数：
    image: PIL图像 (宽w, 高h)
    boxes: 目标边界框张量，维度(#objects, 4)
           格式: [x_min, y_min, x_max, y_max] (左上右下)
    angle: 旋转角度（度），正值为逆时针旋转

    返回：
    new_image: 旋转后的PIL图像 (保持原始尺寸w×h)
    new_boxes: 旋转并调整后的边界框张量

    算法流程：
    1. 复制原始图像和边界框
    2. 计算旋转中心（图像中心）
    3. 应用旋转并扩展图像（expand=True）
    4. 计算旋转后边界框的新坐标
    5. 调整边界框为最小外接矩形
    6. 将图像缩放回原始尺寸
    7. 调整边界框坐标并裁剪到图像范围内

    关键数学：
    - 旋转矩阵: [[cosθ, -sinθ], [sinθ, cosθ]]
    - 仿射变换: [x', y'] = A·[x, y, 1]^T
    - 边界框调整: 旋转后取4个角点的最小/最大坐标

    注意事项：
    - 旋转可能导致部分目标移出图像
    - 边界框在旋转后不再是轴对齐，需重新计算
    - 图像内容在旋转后可能有黑边，需裁剪/填充
    '''
    new_image = image.copy()
    new_boxes = boxes.clone()

    # 获取图像尺寸和中心点
    #Rotate image, expand = True
    w = image.width
    h = image.height
    cx = w/2
    cy = h/2
    # 旋转图像，expand=True确保可以容纳所有内容
    new_image = new_image.rotate(angle, expand=True)
    angle = np.radians(angle)
    alpha = np.cos(angle)
    beta = np.sin(angle)
    # Get affine matrix 构建仿射变换矩阵[2 x 3]
    AffineMatrix = torch.tensor([[alpha, beta, (1-alpha)*cx - beta*cy],
                                 [-beta, alpha, beta*cx + (1-alpha)*cy]])
    
    #Rotation boxes
    box_width = (boxes[:,2] - boxes[:,0]).reshape(-1,1)
    box_height = (boxes[:,3] - boxes[:,1]).reshape(-1,1)
    
    # Get corners for boxes 获取每个边界框的4个角点坐标
    x1 = boxes[:,0].reshape(-1,1)   # 左上x
    y1 = boxes[:,1].reshape(-1,1)   # 左上y
    
    x2 = x1 + box_width   # 右上x
    y2 = y1    # 右上y
    
    x3 = x1   # 左下x
    y3 = y1 + box_height   # 左下y
    
    x4 = boxes[:,2].reshape(-1,1)   # 右下y
    y4 = boxes[:,3].reshape(-1,1)   # 右下y
    
    corners = torch.stack((x1,y1,x2,y2,x3,y3,x4,y4), dim= 1)
    # corners.reshape(-1, 8)    #Tensors of dimensions (#objects, 8)
    corners = corners.reshape(-1,2) #Tensors of dimension (4* #objects, 2)
    corners = torch.cat((corners, torch.ones(corners.shape[0], 1)), dim= 1) #(Tensors of dimension (4* #objects, 3))
    
    cos = np.abs(AffineMatrix[0, 0])
    sin = np.abs(AffineMatrix[0, 1])
    
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    AffineMatrix[0, 2] += (nW / 2) - cx
    AffineMatrix[1, 2] += (nH / 2) - cy
    

    #Apply affine transform
    rotate_corners = torch.mm(AffineMatrix, corners.t().to(torch.float64)).t()
    rotate_corners = rotate_corners.reshape(-1,8)
    
    x_corners = rotate_corners[:,[0,2,4,6]]
    y_corners = rotate_corners[:,[1,3,5,7]]
    
    #Get (x_min, y_min, x_max, y_max)
    x_min, _ = torch.min(x_corners, dim= 1)
    x_min = x_min.reshape(-1, 1)
    y_min, _ = torch.min(y_corners, dim= 1)
    y_min = y_min.reshape(-1, 1)
    x_max, _ = torch.max(x_corners, dim= 1)
    x_max = x_max.reshape(-1, 1)
    y_max, _ = torch.max(y_corners, dim= 1)
    y_max = y_max.reshape(-1, 1)
    
    new_boxes = torch.cat((x_min, y_min, x_max, y_max), dim= 1)
    
    scale_x = new_image.width / w
    scale_y = new_image.height / h
    
    #Resize new image to (w, h)

    new_image = new_image.resize((w, h))
    
    #Resize boxes
    new_boxes /= torch.Tensor([scale_x, scale_y, scale_x, scale_y])
    new_boxes[:, 0] = torch.clamp(new_boxes[:, 0], 0, w)
    new_boxes[:, 1] = torch.clamp(new_boxes[:, 1], 0, h)
    new_boxes[:, 2] = torch.clamp(new_boxes[:, 2], 0, w)
    new_boxes[:, 3] = torch.clamp(new_boxes[:, 3], 0, h)
    return new_image, new_boxes

# def convert_xywh_to_xyxy(boxes: torch.Tensor):
#     _boxes = boxes.clone()
#     box_xy = _boxes[:, :2]
#     box_wh = _boxes[:, 2:]
#     box_x1y1 = box_xy - box_wh/2 
#     box_x2y2 = box_xy + box_wh/2
#     box_xyxy = torch.cat((box_x1y1, box_x2y2), dim=-1)
#     return box_xyxy

class Rotate:
    def __init__(self, angle=10) -> None:
        self.angle = angle

    def __call__(self, img, target):
        w,h = img.size
        whwh = torch.Tensor([w, h, w, h])
        boxes_xyxy = box_cxcywh_to_xyxy(target['boxes']) * whwh
        img, boxes_new = rotate(img, boxes_xyxy, self.angle)
        target['boxes'] = box_xyxy_to_cxcywh(boxes_new).to(boxes_xyxy.dtype) / (whwh + 1e-3)
        return img, target


class RandomCrop:
    def __init__(self) -> None:
        pass

    def __call__(self, img, target):
        w,h = img.size
        try:
            boxes_xyxy = target['boxes']
            labels = target['labels']
            img, new_boxes, new_labels, _ = random_crop(img, boxes_xyxy, labels)
            target['boxes'] = new_boxes
            target['labels'] = new_labels
        except Exception as e:
            pass
        return img, target


class RandomCropDebug:
    def __init__(self) -> None:
        pass

    def __call__(self, img, target):
        boxes_xyxy = target['boxes'].clone()
        labels = target['labels'].clone()
        img, new_boxes, new_labels, _ = random_crop(img, boxes_xyxy, labels)
        target['boxes'] = new_boxes
        target['labels'] = new_labels


        return img, target
        
class RandomSelectMulti(object):
    """
    Randomly selects between transforms1 and transforms2,
    """
    def __init__(self, transformslist, p=-1):
        self.transformslist = transformslist
        self.p = p
        assert p == -1

    def __call__(self, img, target):
        if self.p == -1:
            return random.choice(self.transformslist)(img, target)


class Albumentations:
    def __init__(self):
        import albumentations as A
        self.transform = A.Compose([
            A.Blur(p=0.01),
            A.MedianBlur(p=0.01),
            A.ToGray(p=0.01),
            A.CLAHE(p=0.01),
            A.RandomBrightnessContrast(p=0.005),
            A.RandomGamma(p=0.005),
            A.ImageCompression(quality_lower=75, p=0.005)],
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    def __call__(self, img, target, p=1.0):
        """
        Input:
            target['boxes']: xyxy, unnormalized data.
        
        """
        boxes_raw = target['boxes']
        labels_raw = target['labels']
        img_np = np.array(img)
        if self.transform and random.random() < p:
            new_res = self.transform(image=img_np, bboxes=boxes_raw, class_labels=labels_raw)  # transformed
            boxes_new = torch.Tensor(new_res['bboxes']).to(boxes_raw.dtype).reshape_as(boxes_raw)
            img_np = new_res['image']
            labels_new = torch.Tensor(new_res['class_labels']).to(labels_raw.dtype)
        img_new = Image.fromarray(img_np)
        target['boxes'] = boxes_new
        target['labels'] = labels_new
        
        return img_new, target