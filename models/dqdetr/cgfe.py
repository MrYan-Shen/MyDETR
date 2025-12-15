# Modified from https://github.com/Jongchan/attention-module
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class Conv_GN(nn.Module):
    """
        包含卷积、Group Normalization (GN) 和 ReLU 的模块。
    """
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, gn=True, bias=False):
        super(Conv_GN, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        # Group Normalization: 使用 32 组
        self.gn = nn.GroupNorm(32, out_channel)
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.gn is not None:
            x = self.gn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Conv_BN(nn.Module):
    """
        包含卷积、Batch Normalization (BN) 和 ReLU 的模块。
    """
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(Conv_BN, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channel, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    """将输入张量展平 (保留 Batch 维度)。"""
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    """
        这个模块实现了通道注意力机制。它通过在空间维度上进行池化，然后通过一个 MLP (多层感知机) 来预测每个通道的重要性。
    """
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        # 1. MLP 结构: (C -> C/R -> C)，用于学习通道间的依赖关系
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            # 四种不同类型的池化
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            # 将所有池化类型的结果相加
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        # 2. Sigmoid 激活生成尺度因子
        # scale: (B, C) -> (B, C, 1, 1) -> (B, C, H, W) [广播]
        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return scale  # 返回注意力尺度因子

def logsumexp_2d(tensor):
    """
        在空间维度 (H*W) 上计算 LogSumExp。用于代替最大池化或平均池化。这是一种软池化形式
    """
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    """
        在通道维度上进行最大池化和平均池化，将 (B, C, H, W)
        压缩为 (B, 2, H, W) 作为空间注意力模块的输入。
    """
    def forward(self, x):
        # torch.max(x, 1)[0].unsqueeze(1) 是通道最大池化 (B, 1, H, W)
        # torch.mean(x, 1).unsqueeze(1) 是通道平均池化 (B, 1, H, W)
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    """
        这个模块实现了空间注意力机制。
        它首先对通道信息进行聚合（使用 ChannelPool），然后通过一个卷积层来预测每个空间位置的重要性。
    """
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        # 卷积层: 2 通道 -> 1 通道
        # padding=(kernel_size-1)//2 确保空间尺寸不变
        self.spatial = Conv_BN(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)  # (B, C, H, W) -> (B, 2, H, W)
        x_out = self.spatial(x_compress)  # (B, 2, H, W) -> (B, 1, H, W)

        # Sigmoid 激活生成尺度因子
        scale = F.sigmoid(x_out)  # broadcasting
        return scale  # 返回注意力尺度因子


class CGFE(nn.Module):
    """。
        用于对多尺度特征进行通道和空间注意力增强。
    """
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False, num_feature_levels=4):
        super(CGFE, self).__init__()
        self.num_feat = num_feature_levels  # 特征级别数量
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()  # 空间注意力模块

    def forward(self, x, memory, spatial_shapes):
        """
            Args:
                x (List[Tensor]): 额外的空间特征列表 (如多尺度特征图)。
                memory (Tensor): 编码器输出的展平特征 (BS, SeqLen, C)。
                spatial_shapes (List[Tuple[H, W]]): 每个特征图的原始空间形状。
        """
        feats = []
        idx = 0
        # 1. 准备编码器特征 (memory)
        # (BS, SeqLen, C) -> (BS, C, SeqLen)
        encoder_feat = memory.transpose(1, 2)
        bs, c, hw = encoder_feat.shape

        # 2. 逐级别处理特征
        for i in range(self.num_feat):
            h, w = spatial_shapes[i]
            # 2.1 提取并恢复特征图形状
            # 从展平的 memory 中提取当前级别的特征: (BS, 256, H*W) -> (BS, 256, H, W)
            feat = encoder_feat[:,:,idx:idx+h*w].view(bs, 256, h, w)
            # 2.2 空间注意力增强
            # x[i] 是额外的空间特征输入，用于计算空间注意力 c2
            c2 = self.SpatialGate(x[i])
            feat = feat * c2
            # 2.3 通道注意力增强
            c1 = self.ChannelGate(feat)  # c1: (B, 256, 1, 1)
            feat = feat * c1       # 应用通道注意力
            # 2.4 恢复序列形状并收集
            # (BS, 256, H, W) -> (BS, 256, H*W) -> (BS, H*W, 256)
            feat = feat.flatten(2).transpose(1, 2)
            feats.append(feat)
            idx += h*w  # 更新索引，指向下一个特征级别的起始位置

        # 3. 拼接所有级别的特征
        x_out = torch.cat(feats, 1)  # (BS, sum(H*W), 256)
        return x_out


class MultiScaleFeature(nn.Module):
    """
        通过步长为 2 的卷积层生成多尺度特征金字塔。
    """
    def __init__(self, channels=256, is_5_scale=False):
        super(MultiScaleFeature, self).__init__()
        # 步长为 2 的卷积层，将 H/W 减半，但保持通道数 C 不变
        self.conv1 = Conv_GN(channels, channels, kernel_size=3, stride=2, padding=1)
        self.conv2 = Conv_GN(channels, channels, kernel_size=3, stride=2, padding=1)
        self.conv3 = Conv_GN(channels, channels, kernel_size=3, stride=2, padding=1)
        # 可选的第 5 个尺度
        if is_5_scale:
            self.conv4 = Conv_GN(channels, channels, kernel_size=3, stride=2, padding=1)
        self.is_5_scale = is_5_scale

    def forward(self, x):
        x_out = []
        x_out.append(x)  # 原始尺度 (Scale 1)
        x = self.conv1(x)
        x_out.append(x)  # Scale 2 (1/2 size)
        x = self.conv2(x)
        x_out.append(x)  # Scale 3 (1/4 size)
        x = self.conv3(x)
        x_out.append(x)  # Scale 4 (1/8 size)

        if self.is_5_scale:
           x = self.conv4(x)
           x_out.append(x)   # Scale 5 (1/16 size)
        return x_out
