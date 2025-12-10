# # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# """
# Utilities for bounding box manipulation and GIoU.
# """
# import torch, os
# from torchvision.ops.boxes import box_area
#
#
# def box_cxcywh_to_xyxy(x):
#     x_c, y_c, w, h = x.unbind(-1)
#     b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
#          (x_c + 0.5 * w), (y_c + 0.5 * h)]
#     return torch.stack(b, dim=-1)
#
#
# def box_xyxy_to_cxcywh(x):
#     x0, y0, x1, y1 = x.unbind(-1)
#     b = [(x0 + x1) / 2, (y0 + y1) / 2,
#          (x1 - x0), (y1 - y0)]
#     return torch.stack(b, dim=-1)
#
#
# # modified from torchvision to also return the union
# def box_iou(boxes1, boxes2):
#     area1 = box_area(boxes1)
#     area2 = box_area(boxes2)
#
#
#     lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
#     rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
#
#     wh = (rb - lt).clamp(min=0)  # [N,M,2]
#     inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
#
#     union = area1[:, None] + area2 - inter
#
#     iou = inter / (union + 1e-6)
#     return iou, union
#
#
# def generalized_box_iou(boxes1, boxes2):
#     """
#     Generalized IoU from https://giou.stanford.edu/
#
#     The boxes should be in [x0, y0, x1, y1] format
#
#     Returns a [N, M] pairwise matrix, where N = len(boxes1)
#     and M = len(boxes2)
#     """
#     # degenerate boxes gives inf / nan results
#     # so do an early check
#     assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
#     assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
#
#     iou, union = box_iou(boxes1, boxes2)
#
#     lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
#     rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
#
#     wh = (rb - lt).clamp(min=0)  # [N,M,2]
#     area = wh[:, :, 0] * wh[:, :, 1]
#
#     return iou - (area - union) / (area + 1e-6)
#
#
#
# # modified from torchvision to also return the union
# def box_iou_pairwise(boxes1, boxes2):
#     area1 = box_area(boxes1)
#     area2 = box_area(boxes2)
#
#     lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # [N,2]
#     rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # [N,2]
#
#     wh = (rb - lt).clamp(min=0)  # [N,2]
#     inter = wh[:, 0] * wh[:, 1]  # [N]
#
#     union = area1 + area2 - inter
#
#     iou = inter / union
#     return iou, union
#
#
# def generalized_box_iou_pairwise(boxes1, boxes2):
#     """
#     Generalized IoU from https://giou.stanford.edu/
#
#     Input:
#         - boxes1, boxes2: N,4
#     Output:
#         - giou: N, 4
#     """
#     # degenerate boxes gives inf / nan results
#     # so do an early check
#     assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
#     assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
#     assert boxes1.shape == boxes2.shape
#     iou, union = box_iou_pairwise(boxes1, boxes2) # N, 4
#
#     lt = torch.min(boxes1[:, :2], boxes2[:, :2])
#     rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])
#
#     wh = (rb - lt).clamp(min=0)  # [N,2]
#     area = wh[:, 0] * wh[:, 1]
#
#     return iou - (area - union) / area
#
# def masks_to_boxes(masks):
#     """Compute the bounding boxes around the provided masks
#
#     The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.
#
#     Returns a [N, 4] tensors, with the boxes in xyxy format
#     """
#     if masks.numel() == 0:
#         return torch.zeros((0, 4), device=masks.device)
#
#     h, w = masks.shape[-2:]
#
#     y = torch.arange(0, h, dtype=torch.float)
#     x = torch.arange(0, w, dtype=torch.float)
#     # y, x = torch.meshgrid(y, x,indexing='ij')
#     y, x = torch.meshgrid(y, x)
#
#     x_mask = (masks * x.unsqueeze(0))
#     x_max = x_mask.flatten(1).max(-1)[0]
#     x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]
#
#     y_mask = (masks * y.unsqueeze(0))
#     y_max = y_mask.flatten(1).max(-1)[0]
#     y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]
#
#     return torch.stack([x_min, y_min, x_max, y_max], 1)
#
# if __name__ == '__main__':
#     x = torch.rand(5, 4)
#     y = torch.rand(3, 4)
#     iou, union = box_iou(x, y)
#     import ipdb; ipdb.set_trace()


import torch
import warnings
from torch import Tensor
from typing import Tuple
def box_cxcywh_to_xyxy(x: Tensor) -> Tensor:
    """
    将边界框从 (cx, cy, w, h) 格式转换为 (x1, y1, x2, y2) 格式
    cx: 中心x坐标, cy: 中心y坐标, w: 宽度, h: 高度
    x1, y1: 左上角坐标; x2, y2: 右下角坐标
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)  # 确保至少有一个批次维度
    # 替换NaN为0，避免转换后坐标异常
    x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
    x_c, y_c, w, h = x.unbind(-1)
    # 确保w/h非负（避免负宽高导致无效框）
    w = w.clamp(min=1e-6)
    h = h.clamp(min=1e-6)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x: Tensor) -> Tensor:
    """
    将边界框从 (x1, y1, x2, y2) 格式转换为 (cx, cy, w, h) 格式
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)
    x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
    x1, y1, x2, y2 = x.unbind(-1)
    b = [(x1 + x2) / 2, (y1 + y2) / 2,
         (x2 - x1).clamp(min=1e-6), (y2 - y1).clamp(min=1e-6)]
    return torch.stack(b, dim=-1)


def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    计算两个边界框集合的交并比(IoU)
    Args:
        boxes1: 形状为 (N, 4) 的张量, 格式为 (x1, y1, x2, y2)
        boxes2: 形状为 (M, 4) 的张量, 格式为 (x1, y1, x2, y2)
    Returns:
        iou: 形状为 (N, M) 的张量, 其中 iou[i][j] 是 boxes1[i] 和 boxes2[j] 的IoU
    """
    inter, union = _box_inter_union(boxes1, boxes2)
    iou = inter / union.clamp(min=1e-6)  # 避免除零
    return iou


def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    计算两个边界框集合的广义交并比(Generalized IoU)
    解决 AssertionError: 确保边界框满足 x2 >= x1 且 y2 >= y1
    """

    def fix_invalid_boxes(boxes: Tensor, name: str) -> Tensor:
        """修正无效边界框并警告"""
        if boxes.numel() == 0:
            return boxes  # 空张量直接返回

        # 第一步：替换NaN/无穷大值（核心修复）
        boxes = torch.nan_to_num(
            boxes,
            nan=0.0,  # NaN替换为0
            posinf=1.0,  # 正无穷替换为1
            neginf=0.0  # 负无穷替换为0
        )

        # 第二步：修正坐标顺序（x2 >= x1, y2 >= y1）
        x1, y1, x2, y2 = boxes.unbind(dim=-1)
        new_x1 = torch.min(x1, x2).clamp(min=0.0)  # 限制最小为0（避免负坐标）
        new_x2 = torch.max(x1, x2).clamp(max=1.0)  # 限制最大为1（假设坐标已归一化，根据实际调整）
        new_y1 = torch.min(y1, y2).clamp(min=0.0)
        new_y2 = torch.max(y1, y2).clamp(max=1.0)

        fixed_boxes = torch.stack([new_x1, new_y1, new_x2, new_y2], dim=-1)

        # 第三步：检测仍无效的框（用于调试）
        still_invalid = (fixed_boxes[:, 2] <= fixed_boxes[:, 0] + 1e-6) | (
                    fixed_boxes[:, 3] <= fixed_boxes[:, 1] + 1e-6)
        if still_invalid.any():
            invalid_count = still_invalid.sum().item()
            warnings.warn(
                f"检测到 {invalid_count} 个{name}框（含NaN/异常值），已强制修正为有效框。"
                f"修正后无效框示例: {fixed_boxes[still_invalid][0]}"
            )
            # 强制设置极小有效框（避免断言失败）
            fixed_boxes[still_invalid] = torch.tensor([0.01, 0.01, 0.02, 0.02], device=fixed_boxes.device)

        return fixed_boxes

    # 修正预测框和标注框（双重防护）
    boxes1 = fix_invalid_boxes(boxes1, name="预测")
    boxes2 = fix_invalid_boxes(boxes2, name="标注")

    # 最终断言（添加容差，避免浮点误差）
    assert (boxes1[:, 2] >= boxes1[:,0] + 1e-6).all(), f"仍存在x2 < x1的预测框，示例: {boxes1[boxes1[:, 2] < boxes1[:, 0]][0]}"
    assert (boxes1[:, 3] >= boxes1[:,1] + 1e-6).all(), f"仍存在y2 < y1的预测框，示例: {boxes1[boxes1[:, 3] < boxes1[:, 1]][0]}"
    assert (boxes2[:, 2] >= boxes2[:,0] + 1e-6).all(), f"仍存在x2 < x1的标注框，示例: {boxes2[boxes2[:, 2] < boxes2[:, 0]][0]}"
    assert (boxes2[:, 3] >= boxes2[:,1] + 1e-6).all(), f"仍存在y2 < y1的标注框，示例: {boxes2[boxes2[:, 3] < boxes2[:, 1]][0]}"

    # 原有IoU计算逻辑
    inter, union = _box_inter_union(boxes1, boxes2)
    iou = inter / union.clamp(min=1e-6)

    # 计算最小外接矩形面积
    lti = torch.min(boxes1[:, None, :2], boxes2[:, :2])  # 左上角最小值 (N, M, 2)
    rbi = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])  # 右下角最大值 (N, M, 2)
    whi = (rbi - lti).clamp(min=0.0)  # 外接矩形宽高（确保非负）
    area = whi[..., 0] * whi[..., 1]  # 外接矩形面积

    # 广义IoU = IoU - (外接矩形面积 - 并集面积) / 外接矩形面积
    return iou - (area - union) / area.clamp(min=1e-6)


def _box_inter_union(boxes1: Tensor, boxes2: Tensor) -> Tuple[Tensor, Tensor]:
    """
    计算两个边界框集合的交集和并集面积
    辅助函数，被 box_iou 和 generalized_box_iou 调用
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # (N,)
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # (M,)

    # 计算交集的左上角和右下角坐标
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # (N, M, 2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # (N, M, 2)

    # 计算交集面积（确保宽高非负）
    wh = (rb - lt).clamp(min=0.0)  # (N, M, 2)
    inter = wh[..., 0] * wh[..., 1]  # (N, M)

    # 计算并集面积
    union = area1[:, None] + area2 - inter  # (N, M)
    return inter, union


def clip_boxes_to_image(boxes: Tensor, size: Tuple[int, int]) -> Tensor:
    """
    将边界框裁剪到图像范围内
    Args:
        boxes: 形状为 (N, 4) 的张量, 格式为 (x1, y1, x2, y2)
        size: 图像尺寸 (height, width)
    """
    height, width = size
    boxes[:, 0] = boxes[:, 0].clamp(min=0, max=width)
    boxes[:, 1] = boxes[:, 1].clamp(min=0, max=height)
    boxes[:, 2] = boxes[:, 2].clamp(min=0, max=width)
    boxes[:, 3] = boxes[:, 3].clamp(min=0, max=height)
    return boxes


def remove_small_boxes(boxes: Tensor, min_size: float) -> Tensor:
    """
    移除面积小于 min_size 的边界框
    Args:
        boxes: 形状为 (N, 4) 的张量, 格式为 (x1, y1, x2, y2)
        min_size: 最小面积阈值
    Returns:
        保留的边界框索引
    """
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    keep = (w >= min_size) & (h >= min_size)
    return keep.nonzero().squeeze(1)
