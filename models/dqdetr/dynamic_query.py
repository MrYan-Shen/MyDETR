# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math
#
#
# class DynamicQueryModule(nn.Module):
#     def __init__(self,
#                  feature_dim=256,
#                  num_boundaries=3,
#                  max_objects=1500,
#                  query_levels=[300, 500, 900, 1500],
#                  initial_smoothness=1.0):
#         super().__init__()
#         self.num_boundaries = num_boundaries
#         self.max_objects = max_objects
#         self.query_levels = query_levels
#
#         # 1. 边界预测模块
#         self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.global_max_pool = nn.AdaptiveMaxPool2d(1)
#
#         # 输入归一化，防止特征值过大
#         self.input_norm = nn.LayerNorm(feature_dim * 2)
#
#         self.fc_boundary = nn.Sequential(
#             nn.Linear(feature_dim * 2, feature_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(feature_dim, num_boundaries)
#         )
#
#         self.register_buffer('smoothness', torch.tensor(initial_smoothness))
#
#         # 2. 数量回归模块 (新增：用于预测 Npred)
#         self.count_regressor = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Flatten(),
#             nn.LayerNorm(feature_dim),  # [新增] 归一化
#             nn.Linear(feature_dim, feature_dim // 2),
#             nn.ReLU(inplace=True),
#             nn.Linear(feature_dim // 2, 1),
#             nn.Softplus()  # 确保数量 > 0
#         )
#
#         # 3. 质量感知的查询位置初始化
#         # 通道注意力
#         self.ca_fc1 = nn.Conv2d(feature_dim, feature_dim // 16, 1)
#         self.ca_relu = nn.ReLU(inplace=True)
#         self.ca_fc2 = nn.Conv2d(feature_dim // 16, feature_dim, 1)
#         self.sigmoid = nn.Sigmoid()
#
#         # 空间注意力
#         self.sa_conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
#
#         # 空间权重图生成
#         self.spatial_weight_conv = nn.Conv2d(feature_dim, 1, 1)
#
#         # 初始化权重，防止初始边界爆炸
#         self._init_weights()
#
#     def _init_weights(self):
#         # 将边界预测的最后一层初始化为极小值
#         nn.init.constant_(self.fc_boundary[-1].weight, 0.0)
#         nn.init.constant_(self.fc_boundary[-1].bias, 0.1)  # 初始输出接近 0.1
#
#         # [修改] 数量回归初始化：提高初始偏置，防止初始值过低
#         nn.Softplus(5.0)  # ≈ 5.0，给一个适中的初始值
#         # 数量回归初始化
#         nn.init.constant_(self.count_regressor[-2].weight, 0.0)
#         nn.init.constant_(self.count_regressor[-2].bias, 1.0)  # 初始数量约等于 Softplus(1.0)
#
#     def forward(self, density_feature, encoder_features, real_counts=None):
#         """
#         Args:
#             density_feature: [BS, C, H, W]
#             encoder_features: [BS, C, H, W]
#             real_counts: [BS] (Training only)
#         """
#         bs = density_feature.shape[0]
#         device = density_feature.device
#
#         # 截断输入特征，防止异常值
#         density_feature = torch.clamp(density_feature, min=-10.0, max=10.0)
#         encoder_features = torch.clamp(encoder_features, min=-10.0, max=10.0)
#
#         # --- A. 边界预测 ---
#         feat_avg = self.global_avg_pool(density_feature).flatten(1)
#         feat_max = self.global_max_pool(density_feature).flatten(1)
#         global_feat = torch.cat([feat_avg, feat_max], dim=1)
#
#         # 应用归一化
#         global_feat = self.input_norm(global_feat)
#
#         # 预测原始边界值
#         raw_boundaries = self.fc_boundary(global_feat)
#
#         # 累加 Softplus 变换，确保有序性 b1 < b2 < b3
#         boundaries = []
#         cum_value = 0
#         for i in range(self.num_boundaries):
#             val = F.softplus(raw_boundaries[:, i]) * (self.max_objects / self.num_boundaries)
#             cum_value = cum_value + val + 10.0  # 保证最小间隔 10
#             boundaries.append(cum_value)
#         boundaries = torch.stack(boundaries, dim=1)  # [BS, 3]
#
#         outputs = {
#             'pred_boundaries': boundaries,
#             'raw_boundaries': raw_boundaries
#         }
#
#         # --- B. 计算概率分布与确定查询数量 ---
#
#         # 1. 预测当前图像的目标数量
#         pred_count = self.count_regressor(density_feature).squeeze(1)  # [BS]
#         outputs['predicted_count'] = pred_count  # [新增] 记录预测值供调试
#
#         # 2. 如果是训练阶段且有真实数量，计算 interval_probs 用于 Loss
#         if real_counts is not None:
#             N_real = real_counts.unsqueeze(1).float()
#             b1, b2, b3 = boundaries[:, 0:1], boundaries[:, 1:2], boundaries[:, 2:3]
#             tau = self.smoothness
#
#             # 使用 Sigmoid 计算软区间概率
#             s1 = torch.sigmoid((b1 - N_real) / tau)
#             s2 = torch.sigmoid((b2 - N_real) / tau)
#             s3 = torch.sigmoid((b3 - N_real) / tau)
#
#             p1 = s1
#             p2 = s2 - s1
#             p3 = s3 - s2
#             p4 = 1.0 - s3
#
#             probs = torch.cat([p1, p2, p3, p4], dim=1).clamp(min=1e-6)
#             outputs['interval_probs'] = probs
#
#         # 3. 确定查询数量 (KeyError 修复点)
#         # 策略：推理时使用预测数量；训练时可以使用真实数量来指导(Elastic Training)，或者也使用预测数量
#         # 这里使用预测数量 pred_count 与 预测边界 boundaries 进行比较
#         # 训练时使用 real_counts (如果有) 来决定查询数量，确保 recall
#         # 推理时使用 pred_count
#         if self.training and real_counts is not None:
#             # 训练策略：取 real 和 pred 的最大值，或者直接用 real
#             # 这里为了稳定性，建议直接用 real_counts 所在的区间，
#             # 这样 SetCriterion 的分类 Loss 会教导 count_regressor 去拟合这个区间
#             N_eval = real_counts
#         else:
#             N_eval = pred_count
#
#         # 向量化判断落在哪个区间: 0, 1, 2, 3
#         level_indices = torch.zeros(bs, dtype=torch.long, device=device)
#         level_indices += (N_eval > boundaries[:, 0]).long()
#         level_indices += (N_eval > boundaries[:, 1]).long()
#         level_indices += (N_eval > boundaries[:, 2]).long()
#
#         query_levels_tensor = torch.tensor(self.query_levels, device=device)
#         num_queries = query_levels_tensor[level_indices]
#
#         outputs['num_queries'] = num_queries  # 修复 KeyError
#
#         # --- C. 质量感知位置初始化 (Top-K) ---
#
#         # 双重注意力处理
#         x = encoder_features
#         # Channel Attention
#         ca = self.global_avg_pool(x)
#         ca = self.ca_fc1(ca)
#         ca = self.ca_relu(ca)
#         ca = self.ca_fc2(ca)
#         ca = self.sigmoid(ca)
#         x = x * ca
#
#         # Spatial Attention
#         sa_avg = torch.mean(x, dim=1, keepdim=True)
#         sa_max, _ = torch.max(x, dim=1, keepdim=True)
#         sa = torch.cat([sa_avg, sa_max], dim=1)
#         sa = self.sa_conv(sa)
#         sa = self.sigmoid(sa)
#         x = x * sa
#
#         # 生成权重图
#         weight_map = self.spatial_weight_conv(x).sigmoid()  # [BS, 1, H, W]
#         outputs['spatial_weight_map'] = weight_map.flatten(2)
#
#         # 生成 Reference Points (归一化坐标 cx, cy)
#         # 我们总是生成 max(query_levels) 个点，然后在 Transformer 中根据 num_queries 截断
#         max_k = max(self.query_levels)
#         H, W = weight_map.shape[2], weight_map.shape[3]
#         weight_flat = weight_map.flatten(2).squeeze(1)  # [BS, HW]
#
#         # 防止 HW < max_k 的情况
#         actual_k = min(H * W, max_k)
#
#         # 选取权重最高的 Top-K 索引
#         _, topk_ind = torch.topk(weight_flat, actual_k, dim=1)  # [BS, actual_k]
#
#         # 索引转坐标
#         topk_y = (topk_ind // W).float() + 0.5
#         topk_x = (topk_ind % W).float() + 0.5
#
#         # 归一化 [0, 1]
#         topk_y = topk_y / H
#         topk_x = topk_x / W
#
#         # 堆叠为 [BS, actual_k, 2] -> (cx, cy)
#         ref_points = torch.stack([topk_x, topk_y], dim=-1)
#
#         # 添加初始宽高 (例如 0.05) -> [BS, actual_k, 4]
#         ref_points = torch.cat([ref_points, torch.ones_like(ref_points) * 0.05], dim=-1)
#
#         # 如果 HW 不够，进行 Padding
#         if actual_k < max_k:
#             pad_len = max_k - actual_k
#             padding = torch.zeros(bs, pad_len, 4, device=device)
#             ref_points = torch.cat([ref_points, padding], dim=1)
#
#         outputs['reference_points'] = ref_points
#
#         return outputs
#
#     def update_smoothness(self, epoch, total_epochs):
#         new_tau = 1.0 - 0.9 * (epoch / total_epochs)
#         self.smoothness.fill_(max(new_tau, 0.1))

# models/dqdetr/dynamic_query.py

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

        # 1. 边界预测模块
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)

        self.input_norm = nn.LayerNorm(feature_dim * 2)

        self.fc_boundary = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, num_boundaries)
        )

        self.register_buffer('smoothness', torch.tensor(initial_smoothness))

        # 2. 数量回归模块
        # 加强该模块的表达能力，防止梯度消失
        self.count_regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),  # 增加宽度
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, 1),  # 这是我们要初始化的层，索引为 -2
            nn.Softplus()  # 这是最后一层，索引为 -1，无参数
        )

        # 3. 质量感知的查询位置初始化
        self.ca_fc1 = nn.Conv2d(feature_dim, feature_dim // 16, 1)
        self.ca_relu = nn.ReLU(inplace=True)
        self.ca_fc2 = nn.Conv2d(feature_dim // 16, feature_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.sa_conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.spatial_weight_conv = nn.Conv2d(feature_dim, 1, 1)

        self._init_weights()

    def _init_weights(self):
        # [核心修复] 降低初始边界值
        # 使用负偏置，使 Softplus 输出接近 0
        # Softplus(-3.0) ≈ 0.05.
        # 缩放因子约为 250 (1500/3 * 0.5)
        # 0.05 * 250 = 12.5. 加上 offset 10 -> 初始边界 b1 ≈ 22.5
        # 这样 GT=58 时，就会跨过 b1，选择更高的查询等级
        nn.init.constant_(self.fc_boundary[-1].weight, 0.0)
        nn.init.constant_(self.fc_boundary[-1].bias, -3.0)

        # [核心修复] 提高初始预测数量
        # 初始化为预测约 50 个目标，而不是 1-2 个
        # Softplus(4.0) ≈ 4.0. 也就是 Log(Exp(4)+1)

        # !!! 修正点：使用 [-2] 来访问 Linear 层，而不是 [-1] (Softplus)
        nn.init.constant_(self.count_regressor[-2].weight, 0.0)
        nn.init.constant_(self.count_regressor[-2].bias, 4.0)

    def forward(self, density_feature, encoder_features, real_counts=None):
        bs = density_feature.shape[0]
        device = density_feature.device

        # [保护] 防止梯度爆炸/消失
        density_feature = torch.clamp(density_feature, min=-5.0, max=5.0)
        encoder_features = torch.clamp(encoder_features, min=-5.0, max=5.0)

        # --- A. 边界预测 ---
        feat_avg = self.global_avg_pool(density_feature).flatten(1)
        feat_max = self.global_max_pool(density_feature).flatten(1)
        global_feat = torch.cat([feat_avg, feat_max], dim=1)
        global_feat = self.input_norm(global_feat)

        raw_boundaries = self.fc_boundary(global_feat)

        boundaries = []
        cum_value = 0
        # 减小缩放因子，让边界更紧凑
        scale_factor = (self.max_objects / self.num_boundaries) * 0.3

        for i in range(self.num_boundaries):
            val = F.softplus(raw_boundaries[:, i]) * scale_factor
            # 减小固定间隔，允许边界从很小的值开始
            cum_value = cum_value + val + 5.0
            boundaries.append(cum_value)
        boundaries = torch.stack(boundaries, dim=1)  # [BS, 3]

        outputs = {
            'pred_boundaries': boundaries,
            'raw_boundaries': raw_boundaries
        }

        # --- B. 计算概率分布与确定查询数量 ---
        pred_count = self.count_regressor(density_feature).squeeze(1)
        outputs['predicted_count'] = pred_count

        if real_counts is not None:
            N_real = real_counts.unsqueeze(1).float()
            b1, b2, b3 = boundaries[:, 0:1], boundaries[:, 1:2], boundaries[:, 2:3]
            tau = self.smoothness

            s1 = torch.sigmoid((b1 - N_real) / tau)
            s2 = torch.sigmoid((b2 - N_real) / tau)
            s3 = torch.sigmoid((b3 - N_real) / tau)

            probs = torch.cat([s1, s2 - s1, s3 - s2, 1.0 - s3], dim=1).clamp(min=1e-6)
            outputs['interval_probs'] = probs

        # [关键策略] 混合决策逻辑
        if self.training and real_counts is not None:
            # 训练时：为了让模型见过更多的 Queries，我们人为“拔高”需求
            # 策略：取 real 和 pred 的最大值，并额外加上 10% 的余量
            # 这样能迫使模型经常选择 Level 1 (500) 或 Level 2 (900)
            N_eval = torch.max(real_counts, pred_count) * 1.1
        else:
            N_eval = pred_count

        # 确定区间
        level_indices = torch.zeros(bs, dtype=torch.long, device=device)
        level_indices += (N_eval > boundaries[:, 0]).long()
        level_indices += (N_eval > boundaries[:, 1]).long()
        level_indices += (N_eval > boundaries[:, 2]).long()

        query_levels_tensor = torch.tensor(self.query_levels, device=device)
        num_queries = query_levels_tensor[level_indices]

        outputs['num_queries'] = num_queries

        # --- C. 质量感知位置初始化 ---
        x = encoder_features
        ca = self.global_avg_pool(x)
        ca = self.ca_fc1(ca)
        ca = self.ca_relu(ca)
        ca = self.ca_fc2(ca)
        ca = self.sigmoid(ca)
        x = x * ca

        sa_avg = torch.mean(x, dim=1, keepdim=True)
        sa_max, _ = torch.max(x, dim=1, keepdim=True)
        sa = torch.cat([sa_avg, sa_max], dim=1)
        sa = self.sa_conv(sa)
        sa = self.sigmoid(sa)
        x = x * sa

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
        new_tau = 1.0 - 0.9 * (epoch / total_epochs)
        self.smoothness.fill_(max(new_tau, 0.1))

