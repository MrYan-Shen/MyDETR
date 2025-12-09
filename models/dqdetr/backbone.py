# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Backbone modules.
"""
from collections import OrderedDict
import os

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List


from util.misc import NestedTensor, clean_state_dict, is_main_process
from .position_encoding import build_position_encoding




class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        # 注册参数和统计量，这些在 forward 中会被使用，但不会被优化器更新
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        # 核心：执行 BatchNorm 计算
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        # 计算尺度因子
        scale = w * (rv + eps).rsqrt()
        # 计算偏置项
        bias = b - rm * scale
        # 最终输出
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_indices: list):
        super().__init__()
        # 1. 参数冻结逻辑
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                # 如果 train_backbone 为 False
                # 或者参数不属于 ResNet 的 layer2, layer3, layer4 (即 layer1 和 conv1) 则将这些参数设置为 requires_grad=False (冻结)
                parameter.requires_grad_(False)

        # 2. 设置 IntermediateLayerGetter
        # ResNet 的层通常是 layer1, layer2, layer3, layer4
        # return_interm_indices: 例如 [1, 2, 3] 代表返回 layer2, layer3, layer4 的输出
        return_layers = {}
        for idx, layer_index in enumerate(return_interm_indices):
            return_layers.update({"layer{}".format(5 - len(return_interm_indices) + idx): "{}".format(layer_index)})

        # if len:
        #     if use_stage1_feature:
        #         return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        #     else:
        #         return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
        # else:
        #     return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels  # 返回特征图的通道数列表

    def forward(self, tensor_list: NestedTensor):
        # 1. 骨干网络前向传播
        xs = self.body(tensor_list.tensors)
        # 2. 处理掩码（Mask）
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            # 将原始输入掩码 m (H, W) 插值缩放到当前特征图 x 的尺寸 (H', W')
            # 原始掩码 M 的大小与输入图像相同，特征图 X 的大小通常是 M 的 1/32, 1/16 等。
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            # 结果打包成 NestedTensor，包含特征图和对应的掩码
            out[name] = NestedTensor(x, mask)

        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 dilation: bool,
                 return_interm_indices:list,
                 batch_norm=FrozenBatchNorm2d,
                 ):
        # 1. 加载 ResNet 模型
        if name in ['resnet18', 'resnet34', 'resnet50', 'resnet101']:
            backbone = getattr(torchvision.models, name)(
                replace_stride_with_dilation=[False, False, dilation],
                pretrained=is_main_process(), norm_layer=batch_norm)
        else:
            raise NotImplementedError("Why you can get here with name {}".format(name))
        
        #print("------------------------------------------------")
        #print(is_main_process())
        # num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        assert name not in ('resnet18', 'resnet34'), "Only resnet50 and resnet101 are available."
        assert return_interm_indices in [[0,1,2,3], [1,2,3], [3]]
        # 2. 确定输出通道数
        num_channels_all = [256, 512, 1024, 2048]
        # 3. 根据返回的中间层数量，确定最终输出的通道数列表
        num_channels = num_channels_all[4-len(return_interm_indices):]
        super().__init__(backbone, train_backbone, num_channels, return_interm_indices)


class Joiner(nn.Sequential):  # `Joiner` 负责将骨干网络（`Backbone`）和位置编码（`PositionEmbedding`）结合起来。
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        # self[0] 代表 backbone
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        # self[1] 代表 position_embedding
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        # 返回特征图列表和对应的位置编码列表
        return out, pos


def build_backbone(args):  # 这是一个工厂函数，根据输入参数创建并配置完整的 Joiner 模型
    """
    Useful args:
        - backbone: backbone name
        - lr_backbone: 
        - dilation
        - return_interm_indices: available: [0,1,2,3], [1,2,3], [3]
        - backbone_freeze_keywords: 
        - use_checkpoint: for swin only for now

    """
    # 1. 构建位置编码模块
    position_embedding = build_position_encoding(args)
    # 2. 检查骨干网络训练配置
    train_backbone = args.lr_backbone > 0
    if not train_backbone:
        raise ValueError("Please set lr_backbone > 0")
    return_interm_indices = args.return_interm_indices
    assert return_interm_indices in [[0,1,2,3], [1,2,3], [3]]
    backbone_freeze_keywords = args.backbone_freeze_keywords
    use_checkpoint = getattr(args, 'use_checkpoint', False)

    # 构建 Backbone 模块
    if args.backbone in ['resnet50', 'resnet101']:
        backbone = Backbone(args.backbone, train_backbone, args.dilation,   
                                return_interm_indices,   
                                batch_norm=FrozenBatchNorm2d)
        bb_num_channels = backbone.num_channels
    else:
        raise NotImplementedError("Unknown backbone {}".format(args.backbone))
    
    # 最终检查
    assert len(bb_num_channels) == len(return_interm_indices), f"len(bb_num_channels) {len(bb_num_channels)} != len(return_interm_indices) {len(return_interm_indices)}"

    # 组合 Backbone 和 position_embedding
    model = Joiner(backbone, position_embedding)
    model.num_channels = bb_num_channels 
    assert isinstance(bb_num_channels, List), "bb_num_channels is expected to be a List but {}".format(type(bb_num_channels))
    return model
