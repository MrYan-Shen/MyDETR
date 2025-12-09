# # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# import json
# from pathlib import Path
#
# import numpy as np
# import torch
# from PIL import Image
#
# from panopticapi.utils import rgb2id
# from util.box_ops import masks_to_boxes
#
# from .coco import make_coco_transforms
#
#
# class CocoPanoptic:
#     def __init__(self, img_folder, ann_folder, ann_file, transforms=None, return_masks=True):
#         with open(ann_file, 'r') as f:
#             self.coco = json.load(f)
#
#         # sort 'images' field so that they are aligned with 'annotations'
#         # i.e., in alphabetical order
#         self.coco['images'] = sorted(self.coco['images'], key=lambda x: x['id'])
#         # sanity check
#         if "annotations" in self.coco:
#             for img, ann in zip(self.coco['images'], self.coco['annotations']):
#                 assert img['file_name'][:-4] == ann['file_name'][:-4]
#
#         self.img_folder = img_folder
#         self.ann_folder = ann_folder
#         self.ann_file = ann_file
#         self.transforms = transforms
#         self.return_masks = return_masks
#
#     def __getitem__(self, idx):
#         ann_info = self.coco['annotations'][idx] if "annotations" in self.coco else self.coco['images'][idx]
#         img_path = Path(self.img_folder) / ann_info['file_name'].replace('.png', '.jpg')
#         ann_path = Path(self.ann_folder) / ann_info['file_name']
#
#         img = Image.open(img_path).convert('RGB')
#         w, h = img.size
#         if "segments_info" in ann_info:
#             masks = np.asarray(Image.open(ann_path), dtype=np.uint32)
#             masks = rgb2id(masks)
#
#             ids = np.array([ann['id'] for ann in ann_info['segments_info']])
#             masks = masks == ids[:, None, None]
#
#             masks = torch.as_tensor(masks, dtype=torch.uint8)
#             labels = torch.tensor([ann['category_id'] for ann in ann_info['segments_info']], dtype=torch.int64)
#
#         target = {}
#         target['image_id'] = torch.tensor([ann_info['image_id'] if "image_id" in ann_info else ann_info["id"]])
#         if self.return_masks:
#             target['masks'] = masks
#         target['labels'] = labels
#
#         target["boxes"] = masks_to_boxes(masks)
#
#         target['size'] = torch.as_tensor([int(h), int(w)])
#         target['orig_size'] = torch.as_tensor([int(h), int(w)])
#         if "segments_info" in ann_info:
#             for name in ['iscrowd', 'area']:
#                 target[name] = torch.tensor([ann[name] for ann in ann_info['segments_info']])
#
#         if self.transforms is not None:
#             img, target = self.transforms(img, target)
#
#         return img, target
#
#     def __len__(self):
#         return len(self.coco['images'])
#
#     def get_height_and_width(self, idx):
#         img_info = self.coco['images'][idx]
#         height = img_info['height']
#         width = img_info['width']
#         return height, width
#
#
# def build(image_set, args):
#     img_folder_root = Path(args.coco_path)
#     ann_folder_root = Path(args.coco_panoptic_path)
#     assert img_folder_root.exists(), f'provided COCO path {img_folder_root} does not exist'
#     assert ann_folder_root.exists(), f'provided COCO path {ann_folder_root} does not exist'
#     mode = 'panoptic'
#     PATHS = {
#         "train": ("train2017", Path("annotations") / f'{mode}_train2017.json'),
#         "val": ("val2017", Path("annotations") / f'{mode}_val2017.json'),
#     }
#
#     img_folder, ann_file = PATHS[image_set]
#     img_folder_path = img_folder_root / img_folder
#     ann_folder = ann_folder_root / f'{mode}_{img_folder}'
#     ann_file = ann_folder_root / ann_file
#
#     dataset = CocoPanoptic(img_folder_path, ann_folder, ann_file,
#                            transforms=make_coco_transforms(image_set), return_masks=args.masks)
#
#     return dataset


import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from panopticapi.utils import rgb2id
# 移除这行：from util.box_ops import masks_to_boxes

from .coco import make_coco_transforms


def masks_to_boxes(masks):  # COCO全景分割
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    # 此代码使用了panopticapi库的rgb2id函数，这提示我们全景分割标注是以RGB格式存储的，每个唯一的RGB值对应一个唯一的实例ID。
    if masks.numel() == 0:  # if masks.numel() == 0处理空掩码的边界情况，确保在无检测结果时返回形状正确的空张量
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float, device=masks.device)
    x = torch.arange(0, w, dtype=torch.float, device=masks.device)
    y, x = torch.meshgrid(y, x, indexing='ij')

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]
    # 使用masked_fill(~(masks.bool()), 1e8)：将非掩码区域填充为极大值，确保min()操作只考虑掩码区域，这种技巧在处理不规则形状的掩码时特别有效

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    # 返回xyxy格式（左上角x,y，右下角x,y），与COCO标准的xywh（左上角x,y，宽高）不同，这是为了与模型训练保持一致
    return torch.stack([x_min, y_min, x_max, y_max], 1)


class CocoPanoptic:
    def __init__(self, img_folder, ann_folder, ann_file, transforms=None, return_masks=True):
        with open(ann_file, 'r') as f:
            self.coco = json.load(f)

        # sort 'images' field so that they are aligned with 'annotations'
        # i.e., in alphabetical order
        # 按ID对图像进行排序，确保图像与标注顺序一致，在原始COCO数据集中，图像和标注可能不是按相同顺序排列的
        self.coco['images'] = sorted(self.coco['images'], key=lambda x: x['id'])
        # sanity check：部分验证图像文件名与标注文件名匹配
        if "annotations" in self.coco:
            for img, ann in zip(self.coco['images'], self.coco['annotations']):
                assert img['file_name'][:-4] == ann['file_name'][:-4]  # 使用[:-4]去除文件扩展名（如.jpg或.png），比较基本文件名

        self.img_folder = img_folder
        self.ann_folder = ann_folder
        self.ann_file = ann_file
        self.transforms = transforms
        self.return_masks = return_masks

    def __getitem__(self, idx):
        ann_info = self.coco['annotations'][idx] if "annotations" in self.coco else self.coco['images'][idx]
        img_path = Path(self.img_folder) / ann_info['file_name'].replace('.png', '.jpg')
        ann_path = Path(self.ann_folder) / ann_info['file_name']

        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        if "segments_info" in ann_info:
            masks = np.asarray(Image.open(ann_path), dtype=np.uint32)
            masks = rgb2id(masks)  # rgb2id函数将RGB编码的分割图转换为整数ID

            ids = np.array([ann['id'] for ann in ann_info['segments_info']])
            masks = masks == ids[:, None, None]

            masks = torch.as_tensor(masks, dtype=torch.uint8)  # 将NumPy数组转换为PyTorch张量
            labels = torch.tensor([ann['category_id'] for ann in ann_info['segments_info']], dtype=torch.int64)

        target = {}
        target['image_id'] = torch.tensor([ann_info['image_id'] if "image_id" in ann_info else ann_info["id"]])
        if self.return_masks:
            target['masks'] = masks
        target['labels'] = labels

        target["boxes"] = masks_to_boxes(masks)  # 使用内部实现的函数

        target['size'] = torch.as_tensor([int(h), int(w)])
        target['orig_size'] = torch.as_tensor([int(h), int(w)])
        if "segments_info" in ann_info:
            for name in ['iscrowd', 'area']:
                target[name] = torch.tensor([ann[name] for ann in ann_info['segments_info']])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.coco['images'])

    def get_height_and_width(self, idx):
        img_info = self.coco['images'][idx]
        height = img_info['height']
        width = img_info['width']
        return height, width


def build(image_set, args):  # 数据集构建函数
    img_folder_root = Path(args.coco_path)
    ann_folder_root = Path(args.coco_panoptic_path)
    assert img_folder_root.exists(), f'provided COCO path {img_folder_root} does not exist'
    assert ann_folder_root.exists(), f'provided COCO path {ann_folder_root} does not exist'
    mode = 'panoptic'
    PATHS = {
        "train": ("train2017", Path("annotations") / f'{mode}_train2017.json'),
        "val": ("val2017", Path("annotations") / f'{mode}_val2017.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    img_folder_path = img_folder_root / img_folder
    ann_folder = ann_folder_root / f'{mode}_{img_folder}'
    ann_file = ann_folder_root / ann_file

    dataset = CocoPanoptic(img_folder_path, ann_folder, ann_file,
                           transforms=make_coco_transforms(image_set), return_masks=args.masks)

    return dataset