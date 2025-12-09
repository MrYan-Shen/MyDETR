import PIL #version 1.2.0
import torch
import os
import torchvision.transforms.functional as F
import numpy as np
import random


def intersect(boxes1, boxes2):
    '''
        Find intersection of every box combination between two sets of box
        boxes1: bounding boxes 1, a tensor of dimensions (n1, 4)
        boxes2: bounding boxes 2, a tensor of dimensions (n2, 4)
        
        Out: Intersection each of boxes1 with respect to each of boxes2, 
             a tensor of dimensions (n1, n2)

        计算两组边界框之间所有可能组合的交集面积
        参数：
        boxes1: 第一组边界框，维度为(n1, 4)的张量
                格式: [x_min, y_min, x_max, y_max]
        boxes2: 第二组边界框，维度为(n2, 4)的张量
                格式: [x_min, y_min, x_max, y_max]

        返回：
        inter: 交集面积矩阵，维度为(n1, n2)
               inter[i, j]表示boxes1[i]与boxes2[j]的交集面积

        算法原理：
        1. 对于每对边界框，计算交集区域的右下角坐标：
           max_xy = min(boxes1右下角, boxes2右下角)
        2. 计算交集区域的左上角坐标：
           min_xy = max(boxes1左上角, boxes2左上角)
        3. 交集宽度 = max(0, max_xy.x - min_xy.x)
           交集高度 = max(0, max_xy.y - min_xy.y)
        4. 交集面积 = 宽度 × 高度

        广播机制：
        - boxes1[:, 2:].unsqueeze(1).expand(n1, n2, 2)：将boxes1的右下角扩展为(n1, n2, 2)
        - boxes2[:, 2:].unsqueeze(0).expand(n1, n2, 2)：将boxes2的右下角扩展为(n1, n2, 2)
        - 通过张量广播实现高效批量计算，避免Python循环
    '''
    n1 = boxes1.size(0)  # boxes1,2中的框数量
    n2 = boxes2.size(0)
    # 计算交集区域的右下角坐标：取两组框右下角的最小值
    max_xy =  torch.min(boxes1[:, 2:].unsqueeze(1).expand(n1, n2, 2),  # shape: (n1, 1, 2) -> (n1, n2, 2)
                        boxes2[:, 2:].unsqueeze(0).expand(n1, n2, 2))  # shape: (1, n2, 2) -> (n1, n2, 2)
    # 计算交集区域的左上角坐标：取两组框左上角的最小值
    min_xy = torch.max(boxes1[:, :2].unsqueeze(1).expand(n1, n2, 2),
                       boxes2[:, :2].unsqueeze(0).expand(n1, n2, 2))
    # 计算交集宽度和高度，确保非负
    inter = torch.clamp(max_xy - min_xy , min=0)  # (n1, n2, 2)
    # 交集面积 = 宽度 × 高度
    return inter[:, :, 0] * inter[:, :, 1]  # (n1, n2)


def find_IoU(boxes1, boxes2):
    '''
        Find IoU between every boxes set of boxes 
        boxes1: a tensor of dimensions (n1, 4) (left, top, right , bottom)
        boxes2: a tensor of dimensions (n2, 4)
        
        Out: IoU each of boxes1 with respect to each of boxes2, a tensor of 
             dimensions (n1, n2)
        
        Formula: 
        (box1 ∩ box2) / (box1 u box2) = (box1 ∩ box2) / (area(box1) + area(box2) - (box1 ∩ box2 ))

        算法步骤：
        1. 计算交集面积：使用intersect函数
        2. 计算boxes1中每个框的面积
        3. 计算boxes2中每个框的面积
        4. 通过广播机制扩展面积张量以匹配交集矩阵维度
        5. 计算并集面积 = area1 + area2 - intersection
        6. 计算IoU = intersection / union
    '''
    inter = intersect(boxes1, boxes2)
    area_boxes1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area_boxes2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    area_boxes1 = area_boxes1.unsqueeze(1).expand_as(inter) #(n1, n2)
    area_boxes2 = area_boxes2.unsqueeze(0).expand_as(inter)  #(n1, n2)
    union = (area_boxes1 + area_boxes2 - inter)
    return inter / union


def random_crop(image, boxes, labels, difficulties=None):
    '''
        image: A PIL image
        boxes: Bounding boxes, a tensor of dimensions (#objects, 4)
        labels: labels of object, a tensor of dimensions (#objects)
        difficulties: difficulties of detect object, a tensor of dimensions (#objects)
        
        Out: cropped image , new boxes, new labels, new difficulties

    算法策略：
    1. 随机选择裁剪模式（严格程度）：
       - 0.1, 0.3, 0.5, 0.9: 要求裁剪区域与至少一个边界框的IoU大于此阈值
       - None: 不裁剪，直接返回原始图像（保留原始数据分布）

    2. 尝试最多50次找到有效裁剪：
       a. 生成随机裁剪尺寸（原始尺寸的30%-100%）
       b. 确保宽高比在0.5-2.0之间
       c. 随机选择裁剪区域左上角位置
       d. 计算裁剪区域与所有边界框的IoU
       e. 检查是否有边界框满足IoU阈值要求
       f. 检查是否有边界框的中心点落在裁剪区域内

    3. 调整保留边界框的坐标：
       - 将绝对坐标转换为相对于裁剪区域的坐标
       - 裁剪边界框使其完全位于裁剪区域内

    4. 仅保留中心点在裁剪区域内的边界框及其相关数据
    '''
    if type(image) == PIL.Image.Image:
        image = F.to_tensor(image)
    original_h = image.size(1)
    original_w = image.size(2)

    # 尝试找到有效裁剪，最多50次
    while True:
        # 随机选择裁剪模式（IoU阈值），包括不裁剪选项
        mode = random.choice([0.1, 0.3, 0.5, 0.9, None])

        #若不裁剪，直接返回原始图像
        if mode is None:
            return F.to_pil_image(image), boxes, labels, difficulties

        # 初始化变量，以防重试
        new_image = image
        new_boxes = boxes
        new_difficulties = difficulties
        new_labels = labels
        for _ in range(50):
            # Crop dimensions: [0.3, 1] of original dimensions
            new_h = random.uniform(0.3*original_h, original_h)
            new_w = random.uniform(0.3*original_w, original_w)
            
            # Aspect ratio constraint b/t .5 & 2
            if new_h/new_w < 0.5 or new_h/new_w > 2:
                continue

            # 3. 随机选择裁剪区域的左上角坐标
            #Crop coordinate
            left = random.uniform(0, original_w - new_w)
            right = left + new_w
            top = random.uniform(0, original_h - new_h)
            bottom = top + new_h
            # 4. 创建裁剪区域的边界框 [left, top, right, bottom]
            crop = torch.FloatTensor([int(left), int(top), int(right), int(bottom)])

            # 5. 计算裁剪区域与所有边界框的IoU
            # crop.unsqueeze(0) 将裁剪框扩展为(1, 4)以匹配find_IoU输入格式
            # Calculate IoU  between the crop and the bounding boxes
            overlap = find_IoU(crop.unsqueeze(0), boxes) #(1, #objects)
            overlap = overlap.squeeze(0)

            # 6. 边界情况检查：无边界框
            # If not a single bounding box has a IoU of greater than the minimum, try again
            if overlap.shape[0] == 0:
                continue
            # 7. 计算所有边界框的中心点
            if overlap.max().item() < mode:
                continue
            
            #Crop
            new_image = image[:, int(top):int(bottom), int(left):int(right)] #(3, new_h, new_w)
            
            #Center of bounding boxes
            center_bb = (boxes[:, :2] + boxes[:, 2:])/2.0
            
            #Find bounding box has been had center in crop
            center_in_crop = (center_bb[:, 0] >left) * (center_bb[:, 0] < right
                             ) *(center_bb[:, 1] > top) * (center_bb[:, 1] < bottom)    #( #objects)

            # 至少确保有一个边界框的中心在裁剪区域内
            if not center_in_crop.any():
                continue
            
            #take matching bounding box
            new_boxes = boxes[center_in_crop, :]
            
            #take matching labels
            new_labels = labels[center_in_crop]
            
            #take matching difficulities
            if difficulties is not None:
                new_difficulties = difficulties[center_in_crop]
            else:
                new_difficulties = None
            
            #Use the box left and top corner or the crop's
            new_boxes[:, :2] = torch.max(new_boxes[:, :2], crop[:2])
            
            #adjust to crop
            new_boxes[:, :2] -= crop[:2]
            
            new_boxes[:, 2:] = torch.min(new_boxes[:, 2:],crop[2:])
            
            #adjust to crop
            new_boxes[:, 2:] -= crop[:2]
            
            return F.to_pil_image(new_image), new_boxes, new_labels, new_difficulties