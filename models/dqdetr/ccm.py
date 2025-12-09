import torch.nn as nn
import torch
from torchvision import models
import torch.nn.functional as F

class CategoricalCounting(nn.Module):
    """
        它接收图像特征，通过一系列卷积层和自适应平均池化，
        最终输出一个表示各类别计数的向量。
    """
    def __init__(self, cls_num=4):  # cls_num是需要计数的类别数量，默认为4
        super(CategoricalCounting, self).__init__()        
        # 1. CCM卷积层的配置
        # 定义CCM中使用的卷积层的输出通道数
        self.ccm_cfg = [512, 512, 512, 256, 256, 256]
        self.in_channels = 512  # CCM 模块第一个卷积层的输入通道数（在 conv1 之后）
        # 2. 特征通道调整层（CONV1）
        # 假设输入的特征（v_feat）通道数为 256，将其提升到 512
        self.conv1 = nn.Conv2d(256, self.in_channels, kernel_size=1)
        # 3. CCM 核心卷积块
        # 使用 make_layers 函数构建一系列带空洞卷积 (d_rate=2) 的卷积层
        self.ccm = make_layers(self.ccm_cfg, in_channels=self.in_channels, d_rate=2)
        # 4. 空间信息压缩层
        # 将空间维度 (H, W) 压缩为 (1, 1)，用于将特征图转换为向量
        self.output = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # 5. 最终输出层
        # CCM 最后一层的输出通道数为 256（来自 ccm_cfg 的最后一个元素）。
        # 线性层将 256 维的特征向量映射到 cls_num 维的计数向量。
        self.linear = nn.Linear(256, cls_num)
        
    def forward(self, features, spatial_shapes=None):
        """
            前向传播函数。
            Args:
                features (Tensor): 输入特征。通常来自 Transformer 解码器的输出，
                                   形状为 (BS, SequenceLength, 256)。
                spatial_shapes (list of tuple): 描述特征图的原始空间形状，
                                                例如 [(H, W), ...]。
            Returns:
                out (Tensor): 预测的类别计数向量，形状为 (BS, cls_num)。
                x (Tensor): CCM 模块输出的特征图（在 AdaptiveAvgPool2d 之前）。
        """
        # 输入检查
        if torch.isnan(features).any() or torch.isinf(features).any():
            print("  Warning: CCM input features have NaN/Inf")
            features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)
            features = features.clamp(min=-10.0, max=10.0)
        # 1.维度调整
        # 输入特征形状: (BS, SeqLen, C) -> (BS, C, SeqLen)
        features = features.transpose(1, 2)
        bs, c, hw = features.shape
        # 2. 提取空间特征的高和宽，并恢复形状
        h, w = spatial_shapes[0][0], spatial_shapes[0][1]

        # 假设 SeqLen 的前 H*W 个元素是空间特征
        # (BS, C, H*W) -> (BS, 256, H, W)
        v_feat = features[:,:,0:h*w].view(bs, 256, h, w)
        # 3. CCM 核心计算
        x = self.conv1(v_feat)  # 通道数提升: (BS, 256, H, W) -> (BS, 512, H, W)
        x = self.ccm(x)       # 通过 CCM 卷积块 (通道数最终变为 256)
        # 4. 空间池化
        out = self.output(x)  # AdaptiveAvgPool2d: (BS, 256, H, W) -> (BS, 256, 1, 1)
        # 5. 展平
        out = out.squeeze(3)  # (BS, 256, 1)
        out = out.squeeze(2)  # (BS, 256)
        # 6. 线性计数预测
        out = self.linear(out)  # 线性层: (BS, 256) -> (BS, cls_num)

        return out, x          
                
def make_layers(cfg, in_channels=3, batch_norm=False, d_rate=1):
    """
        根据配置列表 cfg 创建卷积层序列。

        Args:
            cfg (list): 输出通道数的列表。
            in_channels (int): 第一个卷积层的输入通道数。
            batch_norm (bool): 是否在每个卷积层后添加 BatchNorm。
            d_rate (int): 空洞率 (dilation rate)。用于创建空洞卷积。

        Returns:
            nn.Sequential: 包含卷积、(BatchNorm)、ReLU 层的序列模块。
    """
    layers = []
    for v in cfg:
        # 创建 Conv2d 层
        # padding=d_rate 确保输入和输出特征图的空间尺寸保持不变 (对于 kernel_size=3)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                # 若包含 BatchNorm
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)  
