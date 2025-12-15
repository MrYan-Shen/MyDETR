# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # import math
# #
# #
# # class DynamicQueryModule(nn.Module):
# #     def __init__(self,
# #                  feature_dim=256,
# #                  num_boundaries=3,
# #                  max_objects=1500,
# #                  query_levels=[300, 500, 900, 1500],
# #                  initial_smoothness=1.0):
# #         super().__init__()
# #         self.num_boundaries = num_boundaries
# #         self.max_objects = max_objects
# #         self.query_levels = query_levels
# #
# #         # 1. 边界预测模块 - 保持不变
# #         self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
# #         self.global_max_pool = nn.AdaptiveMaxPool2d(1)
# #         self.input_norm = nn.LayerNorm(feature_dim * 2)
# #
# #         self.fc_boundary = nn.Sequential(
# #             nn.Linear(feature_dim * 2, feature_dim),
# #             nn.ReLU(inplace=True),
# #             nn.Linear(feature_dim, num_boundaries)
# #         )
# #
# #         self.register_buffer('smoothness', torch.tensor(initial_smoothness))
# #
# #         # 2. 数量回归模块 - 关键修复点
# #         self.count_regressor = nn.Sequential(
# #             nn.AdaptiveAvgPool2d(1),
# #             nn.Flatten(),
# #             nn.LayerNorm(feature_dim),
# #             nn.Linear(feature_dim, feature_dim),
# #             nn.ReLU(inplace=True),
# #             nn.Dropout(0.1),  # 添加正则化
# #             nn.Linear(feature_dim, 1)
# #             # 移除 Softplus,改用后处理
# #         )
# #
# #         # 3. 质量感知初始化 - 保持不变
# #         self.ca_fc1 = nn.Conv2d(feature_dim, feature_dim // 16, 1)
# #         self.ca_relu = nn.ReLU(inplace=True)
# #         self.ca_fc2 = nn.Conv2d(feature_dim // 16, feature_dim, 1)
# #         self.sigmoid = nn.Sigmoid()
# #         self.sa_conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
# #         self.spatial_weight_conv = nn.Conv2d(feature_dim, 1, 1)
# #
# #         self._init_weights()
# #
# #     def _init_weights(self):
# #         # [修复1] 边界初始化 - 让初始边界更合理分布
# #         nn.init.constant_(self.fc_boundary[-1].weight, 0.0)
# #         # 使用对数空间均匀分布初始化
# #         init_boundaries = torch.log(torch.tensor([300.0, 700.0, 1200.0]))
# #         nn.init.constant_(self.fc_boundary[-1].bias[0], init_boundaries[0].item() - 6.0)
# #         nn.init.constant_(self.fc_boundary[-1].bias[1], init_boundaries[1].item() - 6.0)
# #         nn.init.constant_(self.fc_boundary[-1].bias[2], init_boundaries[2].item() - 6.0)
# #
# #         # 数量回归初始化 - 预测log scale
# #         # 期望初始输出 log(50) ≈ 3.9
# #         nn.init.xavier_uniform_(self.count_regressor[-1].weight, gain=0.01)
# #         nn.init.constant_(self.count_regressor[-1].bias, 3.9)
# #
# #         # [Fix] Initialize spatial weight map to low probability (0.01)
# #         # This prevents random high confidence peaks in early training
# #         nn.init.constant_(self.spatial_weight_conv.bias, -4.59)
# #         nn.init.xavier_uniform_(self.spatial_weight_conv.weight, gain=0.01)
# #
# #     def forward(self, density_feature, encoder_features, real_counts=None):
# #         bs = density_feature.shape[0]
# #         device = density_feature.device
# #
# #         # 输入保护
# #         density_feature = torch.clamp(density_feature, min=-5.0, max=5.0)
# #         encoder_features = torch.clamp(encoder_features, min=-5.0, max=5.0)
# #
# #         # === A. 边界预测 ===
# #         feat_avg = self.global_avg_pool(density_feature).flatten(1)
# #         feat_max = self.global_max_pool(density_feature).flatten(1)
# #         global_feat = torch.cat([feat_avg, feat_max], dim=1)
# #         global_feat = self.input_norm(global_feat)
# #
# #         raw_boundaries = self.fc_boundary(global_feat)  # [BS, 3]
# #
# #         # [修复3] 改进边界生成 - 使用exp而非softplus,更大的动态范围
# #         boundaries = []
# #         for i in range(self.num_boundaries):
# #             # 边界 = exp(raw) * 缩放因子 + 前一个边界 + 最小间隔
# #             val = torch.exp(raw_boundaries[:, i]) * 100.0  # 扩大缩放因子
# #             if i == 0:
# #                 boundaries.append(val + 50.0)  # b1 最小50
# #             else:
# #                 boundaries.append(boundaries[-1] + val + 50.0)  # 保证递增且间隔至少50
# #
# #         boundaries = torch.stack(boundaries, dim=1)  # [BS, 3]
# #         boundaries = boundaries.clamp(max=self.max_objects)  # 限制最大值
# #
# #         outputs = {
# #             'pred_boundaries': boundaries,
# #             'raw_boundaries': raw_boundaries
# #         }
# #
# #         # === B. 数量预测 ===
# #         raw_count = self.count_regressor(density_feature).squeeze(1)  # [BS]
# #         # [修复4] 使用exp转换,并加入合理的范围限制
# #         pred_count = torch.exp(raw_count).clamp(min=1.0, max=self.max_objects)
# #
# #         outputs['predicted_count'] = pred_count
# #         outputs['raw_count'] = raw_count  # 保存用于L2正则
# #
# #         # === C. 概率分布计算 (仅训练时) ===
# #         if real_counts is not None:
# #             N_real = real_counts.unsqueeze(1).float()
# #             b1, b2, b3 = boundaries[:, 0:1], boundaries[:, 1:2], boundaries[:, 2:3]
# #             tau = self.smoothness
# #
# #             s1 = torch.sigmoid((b1 - N_real) / tau)
# #             s2 = torch.sigmoid((b2 - N_real) / tau)
# #             s3 = torch.sigmoid((b3 - N_real) / tau)
# #
# #             probs = torch.cat([s1, s2 - s1, s3 - s2, 1.0 - s3], dim=1).clamp(min=1e-6)
# #             outputs['interval_probs'] = probs
# #
# #         # === D. 确定查询数量 ===
# #         # [修复5] 训练时使用真实数量(确保recall),推理时使用预测数量
# #         if self.training and real_counts is not None:
# #
# #             N_eval = real_counts.float()
# #         else:
# #
# #             N_eval = pred_count
# #
# #         # 根据N_eval确定区间
# #         level_indices = torch.zeros(bs, dtype=torch.long, device=device)
# #         level_indices += (N_eval > boundaries[:, 0]).long()
# #         level_indices += (N_eval > boundaries[:, 1]).long()
# #         level_indices += (N_eval > boundaries[:, 2]).long()
# #
# #         query_levels_tensor = torch.tensor(self.query_levels, device=device)
# #         num_queries = query_levels_tensor[level_indices]
# #
# #         outputs['num_queries'] = num_queries
# #
# #         # === E. 质量感知位置初始化 ===
# #         x = encoder_features
# #
# #         # Channel Attention
# #         ca = self.global_avg_pool(x)
# #         ca = self.ca_fc1(ca)
# #         ca = self.ca_relu(ca)
# #         ca = self.ca_fc2(ca)
# #         ca = self.sigmoid(ca)
# #         x = x * ca
# #
# #         # Spatial Attention
# #         sa_avg = torch.mean(x, dim=1, keepdim=True)
# #         sa_max, _ = torch.max(x, dim=1, keepdim=True)
# #         sa = torch.cat([sa_avg, sa_max], dim=1)
# #         sa = self.sa_conv(sa)
# #         sa = self.sigmoid(sa)
# #         x = x * sa
# #
# #         # 生成权重图
# #         weight_map = self.spatial_weight_conv(x).sigmoid()
# #         outputs['spatial_weight_map'] = weight_map.flatten(2)
# #
# #         # 生成参考点
# #         max_k = max(self.query_levels)
# #         H, W = weight_map.shape[2], weight_map.shape[3]
# #         weight_flat = weight_map.flatten(2).squeeze(1)
# #
# #         actual_k = min(H * W, max_k)
# #         _, topk_ind = torch.topk(weight_flat, actual_k, dim=1)
# #
# #         topk_y = (topk_ind // W).float() + 0.5
# #         topk_x = (topk_ind % W).float() + 0.5
# #
# #         topk_y = topk_y / H
# #         topk_x = topk_x / W
# #
# #         ref_points = torch.stack([topk_x, topk_y], dim=-1)
# #         ref_points = torch.cat([ref_points, torch.ones_like(ref_points) * 0.05], dim=-1)
# #
# #         if actual_k < max_k:
# #             pad_len = max_k - actual_k
# #             padding = torch.zeros(bs, pad_len, 4, device=device)
# #             ref_points = torch.cat([ref_points, padding], dim=1)
# #
# #         outputs['reference_points'] = ref_points
# #
# #         return outputs
# #
# #     def update_smoothness(self, epoch, total_epochs):
# #         """动态调整平滑系数"""
# #         new_tau = 2.0 - 0.9 * (epoch / total_epochs)
# #         self.smoothness.fill_(max(new_tau, 0.1))
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math
#
#
# # --- 来自原 ccm.py 的辅助类 ---
# class Conv_GN(nn.Module):
#     def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
#                  gn=True, bias=False):
#         super(Conv_GN, self).__init__()
#         self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding,
#                               dilation=dilation, groups=groups, bias=bias)
#         self.gn = nn.GroupNorm(32, out_channel) if gn else None
#         self.relu = nn.ReLU(inplace=True) if relu else None
#
#     def forward(self, x):
#         x = self.conv(x)
#         if self.gn is not None:
#             x = self.gn(x)
#         if self.relu is not None:
#             x = self.relu(x)
#         return x
#
#
# def make_ccm_layers(cfg, in_channels=256, d_rate=2):
#     layers = []
#     for v in cfg:
#         # 使用空洞卷积扩大感受野，利于密度估计
#         conv2d = Conv_GN(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
#         layers.append(conv2d)
#         in_channels = v
#     return nn.Sequential(*layers)
#
#
# # --- 融合后的统一动态查询模块 ---
# class DynamicQueryModule(nn.Module):
#     def __init__(self,
#                  feature_dim=256,
#                  num_boundaries=3,
#                  max_objects=1500,
#                  query_levels=[100, 300, 500, 900],  # 针对 AI-TOD 优化
#                  initial_smoothness=1.0,
#                  ccm_cls_num=4):  # CCM 分类数
#         super().__init__()
#         self.num_boundaries = num_boundaries
#         self.max_objects = max_objects
#         self.query_levels = query_levels
#
#         # === 1. 共享密度提取器 (Shared Density Extractor, 原 CCM 主干) ===
#         # 输入维度 256 -> 提升到 512 -> 最终降回 256
#         self.density_conv1 = nn.Conv2d(feature_dim, 512, kernel_size=1)
#         # CCM 配置: [512, 512, 512, 256, 256, 256]
#         self.density_backbone = make_ccm_layers([512, 512, 512, 256, 256, 256], in_channels=512, d_rate=2)
#
#         # === 2. 分支 A: CCM 类别计数头 (用于 ccm_loss) ===
#         self.ccm_pool = nn.AdaptiveAvgPool2d(1)
#         self.ccm_head = nn.Linear(256, ccm_cls_num)
#
#         # === 3. 分支 B: 动态边界与总数回归 (DQ Logic) ===
#         # 共享特征 -> 边界预测
#         self.boundary_pool = nn.AdaptiveMaxPool2d(1)  # 使用 MaxPool 捕捉最密集区域特征
#         self.boundary_norm = nn.LayerNorm(256)
#         self.fc_boundary = nn.Sequential(
#             nn.Linear(256, 128),
#             nn.ReLU(inplace=True),
#             nn.Linear(128, num_boundaries)
#         )
#
#         # 共享特征 -> 总数回归
#         self.count_regressor = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Flatten(),
#             nn.LayerNorm(256),
#             nn.Linear(256, 128),
#             nn.ReLU(inplace=True),
#             nn.Linear(128, 1)  # 输出 log(count)
#         )
#
#         # === 4. 分支 C: 参考点生成 (基于密度图) ===
#         # 将 256 通道的密度特征投影为 1 通道的 Heatmap
#         self.ref_point_conv = nn.Conv2d(256, 1, kernel_size=1)
#
#         self.register_buffer('smoothness', torch.tensor(initial_smoothness))
#         self._init_weights()
#
#     def _init_weights(self):
#         # 1. 初始化 CCM 部分 (He init)
#         for m in self.density_backbone.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#
#         # 2. 初始化边界 (Softplus 偏置初始化，对应 100, 300, 500)
#         nn.init.constant_(self.fc_boundary[-1].weight, 0.0)
#         nn.init.constant_(self.fc_boundary[-1].bias[0], 2.0)
#         nn.init.constant_(self.fc_boundary[-1].bias[1], 4.0)
#         nn.init.constant_(self.fc_boundary[-1].bias[2], 4.0)
#
#         # 3. 初始化数量回归 (log(50) ≈ 3.9)
#         nn.init.normal_(self.count_regressor[-1].weight, std=0.01)
#         nn.init.constant_(self.count_regressor[-1].bias, 3.9)
#
#         # 4. 参考点生成 (Bias设低，让初始 Heatmap 比较平滑但偏低)
#         nn.init.normal_(self.ref_point_conv.weight, std=0.01)
#         nn.init.constant_(self.ref_point_conv.bias, -2.0)
#
#     def forward(self, feature_map, real_counts=None):
#         """
#         Args:
#             feature_map: (BS, C, H, W) 来自 Encoder 的原始特征
#             real_counts: (BS) 真实目标数量 (仅训练)
#         """
#         bs, c, h, w = feature_map.shape
#         device = feature_map.device
#
#         # --- 1. 提取共享密度特征 ---
#         x = self.density_conv1(feature_map)
#         density_feat = self.density_backbone(x)  # (BS, 256, H, W)
#
#         # --- 2. 分支 A: CCM 类别预测 (辅助 Loss) ---
#         ccm_global = self.ccm_pool(density_feat).flatten(1)
#         pred_bbox_number = self.ccm_head(ccm_global)  # (BS, 4) -> 对应原来的 counting_output
#
#         # --- 3. 分支 B: 动态查询逻辑 ---
#         # 3.1 边界预测
#         bd_feat = self.boundary_pool(density_feat).flatten(1)
#         bd_feat = self.boundary_norm(bd_feat)
#         raw_boundaries = self.fc_boundary(bd_feat)
#
#         # 生成物理边界 (Softplus + 累加)
#         boundaries = []
#         scale_factor = 50.0
#         for i in range(self.num_boundaries):
#             val = F.softplus(raw_boundaries[:, i]) * scale_factor
#             if i == 0:
#                 boundaries.append(val + 10.0)
#             else:
#                 boundaries.append(boundaries[-1] + val + 10.0)
#         boundaries = torch.stack(boundaries, dim=1).clamp(max=self.max_objects)
#
#         # 3.2 总数回归
#         raw_count = self.count_regressor(density_feat).squeeze(1)
#         pred_count = torch.exp(raw_count).clamp(min=1.0, max=self.max_objects)
#
#         # 3.3 计算区间概率 (仅训练用)
#         interval_probs = None
#         if real_counts is not None:
#             N_real = real_counts.unsqueeze(1).float()
#             tau = self.smoothness
#             s1 = torch.sigmoid((boundaries[:, 0:1] - N_real) / tau)
#             s2 = torch.sigmoid((boundaries[:, 1:2] - N_real) / tau)
#             s3 = torch.sigmoid((boundaries[:, 2:3] - N_real) / tau)
#             interval_probs = torch.cat([s1, s2 - s1, s3 - s2, 1.0 - s3], dim=1).clamp(min=1e-6)
#
#         # 3.4 决定 Query 数量
#         if self.training and real_counts is not None:
#             N_eval = real_counts.float() * 1.1 + 10  # 训练时基于 GT 稍微放宽
#         else:
#             N_eval = pred_count  # 推理时基于预测
#
#         level_indices = torch.zeros(bs, dtype=torch.long, device=device)
#         for i in range(self.num_boundaries):
#             level_indices += (N_eval > boundaries[:, i]).long()
#
#         query_levels_tensor = torch.tensor(self.query_levels, device=device)
#         num_queries = query_levels_tensor[level_indices]  # (BS,)
#
#         # --- 4. 分支 C: 基于密度的参考点生成 ---
#         # 这一步是降低 FP/FN 的关键：我们直接用密度特征生成 Heatmap
#         heatmap = self.ref_point_conv(density_feat).sigmoid()  # (BS, 1, H, W)
#
#         # 选取 Top-K 参考点
#         max_k = max(self.query_levels)
#         heatmap_flat = heatmap.flatten(2).squeeze(1)  # (BS, H*W)
#
#         # 选取前 max_k 个点
#         actual_k = min(h * w, max_k)
#         _, topk_ind = torch.topk(heatmap_flat, actual_k, dim=1)
#
#         topk_y = (topk_ind // w).float() + 0.5
#         topk_x = (topk_ind % w).float() + 0.5
#
#         # 归一化
#         ref_points = torch.stack([topk_x / w, topk_y / h], dim=-1)
#
#         # 初始宽高：对于小物体，设小一点 (0.02)
#         initial_wh = torch.ones_like(ref_points) * 0.02
#         ref_points = torch.cat([ref_points, initial_wh], dim=-1)  # (BS, actual_k, 4)
#
#         # Padding (如果有必要)
#         if actual_k < max_k:
#             pad = torch.zeros(bs, max_k - actual_k, 4, device=device)
#             ref_points = torch.cat([ref_points, pad], dim=1)
#
#         # 组装输出
#         outputs = {
#             'pred_boundaries': boundaries,
#             'raw_boundaries': raw_boundaries,
#             'predicted_count': pred_count,
#             'interval_probs': interval_probs,
#             'num_queries': num_queries,
#             'reference_points': ref_points,
#             'pred_bbox_number': pred_bbox_number,  # CCM 的输出
#             'density_map': heatmap,  # 用于可视化调试
#             'density_feature': density_feat
#         }
#
#         return outputs
#
#     def update_smoothness(self, epoch, total_epochs):
#         new_tau = 1.0 - 0.9 * (epoch / total_epochs)
#         self.smoothness.fill_(max(new_tau, 0.1))

import torch
import torch.nn as nn
import torch.nn.functional as F


# --- 辅助模块 (保持不变) ---
class Conv_GN(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 gn=True, bias=False):
        super(Conv_GN, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.gn = nn.GroupNorm(32, out_channel) if gn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.gn is not None:
            x = self.gn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


def make_ccm_layers(cfg, in_channels=256, d_rate=2):
    layers = []
    for v in cfg:
        conv2d = Conv_GN(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
        layers.append(conv2d)
        in_channels = v
    return nn.Sequential(*layers)


class DynamicQueryModule(nn.Module):
    def __init__(self,
                 feature_dim=256,
                 num_boundaries=3,  # 3个边界 -> 4个区间
                 max_objects=1500,
                 query_levels=[300, 500, 900, 1500],  # 对应4个区间的查询数量
                 initial_smoothness=1.0,  # 不再使用，保留参数位防止报错
                 ccm_cls_num=4):  # 你的CCM分类数
        super().__init__()
        self.max_objects = max_objects
        self.query_levels = query_levels
        self.num_intervals = len(query_levels)  # 4

        # === 1. 共享密度提取器 (Backbone) ===
        # 输入: (BS, 256, H, W)
        self.density_conv1 = nn.Conv2d(feature_dim, 512, kernel_size=1)
        # CCM 结构: 扩大感受野，提取密度特征
        self.density_backbone = make_ccm_layers([512, 512, 512, 256, 256, 256], in_channels=512, d_rate=2)

        # === 2. 核心预测头 ===
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # A. 计数回归头 (预测具体的 N_pred)
        self.count_regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)  # 输出 log(count)
        )

        # B. 边界偏置预测头 (可选，或者我们直接使用固定锚点 + 计数来决定)
        # 这里我们采用“区间分类”思路：直接预测输入图像属于哪个密度等级
        # 这比预测边界值更容易训练
        self.interval_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.num_intervals)  # 输出 4 个 logits
        )

        # C. CCM 辅助头 (保持兼容性)
        self.ccm_head = nn.Linear(256, ccm_cls_num)

        # D. 参考点生成头 (Heatmap)
        self.ref_point_conv = nn.Conv2d(256, 1, kernel_size=1)

        # === 3. 定义固定的锚点边界 (经验值) ===
        # 这些值用来在训练时生成 Classification Target
        # 根据你的描述: 10, 100, 500
        # 区间0: [0, 10]     -> Query 100
        # 区间1: [10, 100]   -> Query 300
        # 区间2: [100, 500]  -> Query 500
        # 区间3: [500, inf]  -> Query 900
        self.register_buffer('anchor_boundaries', torch.tensor([10.0, 100.0, 500.0]))

        self._init_weights()

    def _init_weights(self):
        # 初始化 Backbone
        for m in self.density_backbone.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        # 初始化计数回归 (log(50) ≈ 3.9)
        nn.init.normal_(self.count_regressor[-1].weight, std=0.01)
        nn.init.constant_(self.count_regressor[-1].bias, 3.9)

        # 初始化参考点 (Heatmap 偏置设低)
        nn.init.normal_(self.ref_point_conv.weight, std=0.01)
        nn.init.constant_(self.ref_point_conv.bias, -2.19)  # sigmoid(-2.19) ≈ 0.1

    def forward(self, feature_map, real_counts=None):
        """
        feature_map: (BS, 256, H, W)
        real_counts: (BS) GT数量
        """
        bs, c, h, w = feature_map.shape
        device = feature_map.device

        # 1. 提取密度特征
        x = self.density_conv1(feature_map)
        density_feat = self.density_backbone(x)  # (BS, 256, H, W)

        # 全局特征用于回归和分类
        global_feat = self.global_pool(density_feat)  # (BS, 256, 1, 1)

        # 2. 预测计数 (N_pred)
        raw_count = self.count_regressor(global_feat)
        pred_count = torch.exp(raw_count).view(bs)  # (BS)

        # 3. 预测区间概率 (Logits)
        interval_logits = self.interval_classifier(global_feat)  # (BS, 4)
        interval_probs = F.softmax(interval_logits, dim=1)

        # 4. 决策逻辑 (决定 Query 数量)
        query_levels_tensor = torch.tensor(self.query_levels, device=device)

        if self.training and real_counts is not None:
            # === 训练阶段 ===
            # 直接使用 GT 数量来决定 Label，用于计算 interval_loss
            # 但为了增加鲁棒性，我们可以混合使用 (可选)
            # 这里我们主要为了生成 interval_probs 用于 loss 计算

            # 生成区间 Target (用于 CrossEntropy Loss)
            target_labels = torch.zeros(bs, dtype=torch.long, device=device)
            # 0: < 10
            # 1: 10 <= x < 100
            # 2: 100 <= x < 500
            # 3: >= 500
            b = self.anchor_boundaries
            target_labels[(real_counts >= b[0]) & (real_counts < b[1])] = 1
            target_labels[(real_counts >= b[1]) & (real_counts < b[2])] = 2
            target_labels[real_counts >= b[2]] = 3

            # 训练时也可以动态选择 Query，这里为了稳定，我们可以使用 pred_count
            # 或者稍微放宽的逻辑
            # 使用 N_pred 来决定 level，这样可以测试 N_pred 的准确性
            # 或者使用 GT (Standard DN-DETR 做法)

            # 策略：训练时使用 max(pred, gt) 保证 recall
            N_eval = torch.max(pred_count.detach(), real_counts)

        else:
            # === 推理阶段 ===
            # 完全依赖 N_pred
            N_eval = pred_count
            target_labels = None  # 推理时没有 Target

        # 根据 N_eval 映射到 Level
        b = self.anchor_boundaries
        level_indices = torch.zeros(bs, dtype=torch.long, device=device)
        level_indices[(N_eval >= b[0]) & (N_eval < b[1])] = 1
        level_indices[(N_eval >= b[1]) & (N_eval < b[2])] = 2
        level_indices[N_eval >= b[2]] = 3

        num_queries = query_levels_tensor[level_indices]

        # 5. 生成参考点 (Heatmap Based)
        heatmap = self.ref_point_conv(density_feat).sigmoid()

        # 动态 TopK
        max_k = max(self.query_levels)
        heatmap_flat = heatmap.flatten(2).squeeze(1)
        actual_k = min(h * w, max_k)
        _, topk_ind = torch.topk(heatmap_flat, actual_k, dim=1)

        topk_y = (topk_ind // w).float() + 0.5
        topk_x = (topk_ind % w).float() + 0.5
        ref_points = torch.stack([topk_x / w, topk_y / h], dim=-1)

        # 初始宽高 (小目标优化)
        initial_wh = torch.ones_like(ref_points) * 0.02
        ref_points = torch.cat([ref_points, initial_wh], dim=-1)

        if actual_k < max_k:
            pad = torch.zeros(bs, max_k - actual_k, 4, device=device)
            ref_points = torch.cat([ref_points, pad], dim=1)

        # 6. CCM 辅助输出
        pred_bbox_number = self.ccm_head(global_feat.flatten(1))

        # 组装输出
        outputs = {
            'predicted_count': pred_count,  # 用于 Regression Loss
            'interval_logits': interval_logits,  # 用于 Classification Loss
            'target_labels': target_labels,  # GT Labels (仅训练)
            'num_queries': num_queries,
            'reference_points': ref_points,
            'pred_bbox_number': pred_bbox_number,
            'density_map': heatmap,
            'density_feature': density_feat,
            # 用于日志打印
            'debug_level_indices': level_indices
        }

        return outputs

    def update_smoothness(self, epoch, total_epochs):
        pass  # 不再需要