import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DynamicQueryModule(nn.Module):
    def __init__(self,
                 feature_dim=256,
                 num_boundaries=3,
                 max_objects=1500,
                 query_levels=[300, 500, 900, 1500],
                 initial_smoothness=1.0):
        super().__init__()
        self.num_boundaries = num_boundaries
        self.max_objects = max_objects
        self.query_levels = query_levels

        # 1. 边界预测模块 - 保持不变
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.input_norm = nn.LayerNorm(feature_dim * 2)

        self.fc_boundary = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, num_boundaries)
        )

        self.register_buffer('smoothness', torch.tensor(initial_smoothness))

        # 2. 数量回归模块 - 关键修复点
        self.count_regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),  # 添加正则化
            nn.Linear(feature_dim, 1)
            # 移除 Softplus,改用后处理
        )

        # 3. 质量感知初始化 - 保持不变
        self.ca_fc1 = nn.Conv2d(feature_dim, feature_dim // 16, 1)
        self.ca_relu = nn.ReLU(inplace=True)
        self.ca_fc2 = nn.Conv2d(feature_dim // 16, feature_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.sa_conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.spatial_weight_conv = nn.Conv2d(feature_dim, 1, 1)

        self._init_weights()

    def _init_weights(self):
        # [修复1] 边界初始化 - 让初始边界更合理分布
        nn.init.constant_(self.fc_boundary[-1].weight, 0.0)
        # 使用对数空间均匀分布初始化
        init_boundaries = torch.log(torch.tensor([300.0, 700.0, 1200.0]))
        nn.init.constant_(self.fc_boundary[-1].bias[0], init_boundaries[0].item() - 6.0)
        nn.init.constant_(self.fc_boundary[-1].bias[1], init_boundaries[1].item() - 6.0)
        nn.init.constant_(self.fc_boundary[-1].bias[2], init_boundaries[2].item() - 6.0)

        # [修复2] 数量回归初始化 - 预测log scale
        # 期望初始输出 log(50) ≈ 3.9
        nn.init.xavier_uniform_(self.count_regressor[-1].weight, gain=0.01)
        nn.init.constant_(self.count_regressor[-1].bias, 3.9)

    def forward(self, density_feature, encoder_features, real_counts=None):
        bs = density_feature.shape[0]
        device = density_feature.device

        # 输入保护
        density_feature = torch.clamp(density_feature, min=-5.0, max=5.0)
        encoder_features = torch.clamp(encoder_features, min=-5.0, max=5.0)

        # === A. 边界预测 ===
        feat_avg = self.global_avg_pool(density_feature).flatten(1)
        feat_max = self.global_max_pool(density_feature).flatten(1)
        global_feat = torch.cat([feat_avg, feat_max], dim=1)
        global_feat = self.input_norm(global_feat)

        raw_boundaries = self.fc_boundary(global_feat)  # [BS, 3]

        # [修复3] 改进边界生成 - 使用exp而非softplus,更大的动态范围
        boundaries = []
        for i in range(self.num_boundaries):
            # 边界 = exp(raw) * 缩放因子 + 前一个边界 + 最小间隔
            val = torch.exp(raw_boundaries[:, i]) * 100.0  # 扩大缩放因子
            if i == 0:
                boundaries.append(val + 50.0)  # b1 最小50
            else:
                boundaries.append(boundaries[-1] + val + 50.0)  # 保证递增且间隔至少50

        boundaries = torch.stack(boundaries, dim=1)  # [BS, 3]
        boundaries = boundaries.clamp(max=self.max_objects)  # 限制最大值

        outputs = {
            'pred_boundaries': boundaries,
            'raw_boundaries': raw_boundaries
        }

        # === B. 数量预测 ===
        raw_count = self.count_regressor(density_feature).squeeze(1)  # [BS]
        # [修复4] 使用exp转换,并加入合理的范围限制
        pred_count = torch.exp(raw_count).clamp(min=1.0, max=self.max_objects)

        outputs['predicted_count'] = pred_count
        outputs['raw_count'] = raw_count  # 保存用于L2正则

        # === C. 概率分布计算 (仅训练时) ===
        if real_counts is not None:
            N_real = real_counts.unsqueeze(1).float()
            b1, b2, b3 = boundaries[:, 0:1], boundaries[:, 1:2], boundaries[:, 2:3]
            tau = self.smoothness

            s1 = torch.sigmoid((b1 - N_real) / tau)
            s2 = torch.sigmoid((b2 - N_real) / tau)
            s3 = torch.sigmoid((b3 - N_real) / tau)

            probs = torch.cat([s1, s2 - s1, s3 - s2, 1.0 - s3], dim=1).clamp(min=1e-6)
            outputs['interval_probs'] = probs

        # === D. 确定查询数量 ===
        # [修复5] 训练时使用真实数量(确保recall),推理时使用预测数量
        if self.training and real_counts is not None:
            # 训练策略: 使用GT,但稍微放宽(1.2倍)确保足够的queries
            N_eval = real_counts.float() * 1.2
        else:
            # 推理策略: 使用预测值,并稍微放宽(1.3倍)
            N_eval = pred_count * 1.3

        # 根据N_eval确定区间
        level_indices = torch.zeros(bs, dtype=torch.long, device=device)
        level_indices += (N_eval > boundaries[:, 0]).long()
        level_indices += (N_eval > boundaries[:, 1]).long()
        level_indices += (N_eval > boundaries[:, 2]).long()

        query_levels_tensor = torch.tensor(self.query_levels, device=device)
        num_queries = query_levels_tensor[level_indices]

        outputs['num_queries'] = num_queries

        # === E. 质量感知位置初始化 ===
        x = encoder_features

        # Channel Attention
        ca = self.global_avg_pool(x)
        ca = self.ca_fc1(ca)
        ca = self.ca_relu(ca)
        ca = self.ca_fc2(ca)
        ca = self.sigmoid(ca)
        x = x * ca

        # Spatial Attention
        sa_avg = torch.mean(x, dim=1, keepdim=True)
        sa_max, _ = torch.max(x, dim=1, keepdim=True)
        sa = torch.cat([sa_avg, sa_max], dim=1)
        sa = self.sa_conv(sa)
        sa = self.sigmoid(sa)
        x = x * sa

        # 生成权重图
        weight_map = self.spatial_weight_conv(x).sigmoid()
        outputs['spatial_weight_map'] = weight_map.flatten(2)

        # 生成参考点
        max_k = max(self.query_levels)
        H, W = weight_map.shape[2], weight_map.shape[3]
        weight_flat = weight_map.flatten(2).squeeze(1)

        actual_k = min(H * W, max_k)
        _, topk_ind = torch.topk(weight_flat, actual_k, dim=1)

        topk_y = (topk_ind // W).float() + 0.5
        topk_x = (topk_ind % W).float() + 0.5

        topk_y = topk_y / H
        topk_x = topk_x / W

        ref_points = torch.stack([topk_x, topk_y], dim=-1)
        ref_points = torch.cat([ref_points, torch.ones_like(ref_points) * 0.05], dim=-1)

        if actual_k < max_k:
            pad_len = max_k - actual_k
            padding = torch.zeros(bs, pad_len, 4, device=device)
            ref_points = torch.cat([ref_points, padding], dim=1)

        outputs['reference_points'] = ref_points

        return outputs

    def update_smoothness(self, epoch, total_epochs):
        """动态调整平滑系数"""
        new_tau = 1.0 - 0.9 * (epoch / total_epochs)
        self.smoothness.fill_(max(new_tau, 0.1))