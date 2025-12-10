"""
Dynamic Query Mechanism for Adaptive Object Detection
动态查询机制：基于可学习边界和软区间分配
完全重写版本，确保数值稳定性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os

# 添加路径以导入安全工具
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



def safe_prob_normalize(probs, eps=1e-6):
    """安全的概率归一化"""
    probs = torch.clamp(probs, min=0.0)  # 确保非负
    denom = probs.sum(dim=1, keepdim=True)
    denom = torch.clamp(denom, min=eps)  # 防止除零
    return probs / denom


class LearnableBoundaryPredictor(nn.Module):
    """
    可学习边界预测器 - 数值稳定版本
    """

    def __init__(self, feature_dim=256, num_boundaries=3, max_objects=1500,
                 initial_smoothness=1.0):
        super().__init__()
        self.num_boundaries = num_boundaries
        self.max_objects = max_objects

        # 全局特征提取
        self.global_pool_avg = nn.AdaptiveAvgPool2d(1)
        self.global_pool_max = nn.AdaptiveMaxPool2d(1)

        # 边界预测网络 - 使用更保守的初始化
        self.boundary_predictor = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),  # 使用 LayerNorm 提高稳定性
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.LayerNorm(feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 2, num_boundaries)
        )

        # 平滑系数（固定或可学习）
        self.register_buffer('smoothness', torch.tensor(initial_smoothness))

        # 🔥 保守的初始化
        self._init_weights()

    def _init_weights(self):
        """保守的权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用更小的初始化范围
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        # 最后一层特殊初始化，使初始输出接近 0
        nn.init.constant_(self.boundary_predictor[-1].weight, 0.01)
        nn.init.constant_(self.boundary_predictor[-1].bias, 0.0)

    def forward(self, density_feature):
        """
        前向传播 - 完全重写，确保数值稳定
        """
        # 🔥 Step 1: 输入清理
        density_feature = sanitize_tensor(
            density_feature,
            name="density_feature",
            default_value=0.0,
            min_val=-5.0,
            max_val=5.0
        )

        # 🔥 Step 2: 全局特征提取（带保护）
        try:
            feat_avg = self.global_pool_avg(density_feature).flatten(1)
            feat_max = self.global_pool_max(density_feature).flatten(1)
        except Exception as e:
            print(f"❌ Pooling error: {e}")
            BS, C = density_feature.shape[0], density_feature.shape[1]
            feat_avg = torch.zeros(BS, C, device=density_feature.device)
            feat_max = torch.zeros(BS, C, device=density_feature.device)

        # 清理特征
        feat_avg = sanitize_tensor(feat_avg, "feat_avg", min_val=-5.0, max_val=5.0)
        feat_max = sanitize_tensor(feat_max, "feat_max", min_val=-5.0, max_val=5.0)

        global_feat = torch.cat([feat_avg, feat_max], dim=1)

        # 🔥 Step 3: 预测原始边界（带限制）
        raw_boundaries = self.boundary_predictor(global_feat)
        raw_boundaries = torch.clamp(raw_boundaries, min=-3.0, max=3.0)

        # 🔥 Step 4: 使用更稳定的单调递增策略
        # 策略：b1, b2, b3 直接预测为递增序列
        # 使用 softmax 来自动保证权重为正且和为1
        weights = F.softmax(raw_boundaries, dim=1)  # (BS, 3)

        # 边界等于累积权重 * max_objects
        boundaries = torch.cumsum(weights, dim=1) * self.max_objects * 0.9

        # 最终保护
        boundaries = torch.clamp(boundaries, min=10.0, max=self.max_objects * 0.95)

        # 确保严格递增（数值修正）
        for i in range(1, boundaries.shape[1]):
            boundaries[:, i] = torch.max(
                boundaries[:, i],
                boundaries[:, i-1] + 5.0  # 最小间隔
            )

        return boundaries, raw_boundaries

    def compute_interval_probabilities(self, boundaries, real_count):
        """
        计算软区间概率 - 使用更稳定的公式
        """
        b1, b2, b3 = boundaries[:, 0], boundaries[:, 1], boundaries[:, 2]
        N = real_count.float().unsqueeze(1)  # (BS, 1)
        r = torch.clamp(self.smoothness, min=0.5, max=3.0)

        # 🔥 使用指数衰减的软指示函数，比 sigmoid 更稳定
        def soft_indicator(x, center, width):
            """
            软指示函数：在 center 附近为 1，远离时衰减
            使用高斯型函数
            """
            dist = (x - center) / (width + 1e-6)
            return torch.exp(-0.5 * dist ** 2)

        # 计算四个区间的中心
        c1 = b1 / 2
        c2 = (b1 + b2) / 2
        c3 = (b2 + b3) / 2
        c4 = (b3 + self.max_objects) / 2

        # 计算每个区间的宽度
        w1 = b1 / 2 + r
        w2 = (b2 - b1) / 2 + r
        w3 = (b3 - b2) / 2 + r
        w4 = (self.max_objects - b3) / 2 + r

        # 计算概率
        p1 = soft_indicator(N, c1.unsqueeze(1), w1.unsqueeze(1))
        p2 = soft_indicator(N, c2.unsqueeze(1), w2.unsqueeze(1))
        p3 = soft_indicator(N, c3.unsqueeze(1), w3.unsqueeze(1))
        p4 = soft_indicator(N, c4.unsqueeze(1), w4.unsqueeze(1))

        probs = torch.cat([p1, p2, p3, p4], dim=1)
        probs = safe_prob_normalize(probs)

        return probs

    def get_query_number(self, boundaries, predicted_count, query_levels):
        """推理阶段：确定查询数量"""
        BS = boundaries.shape[0]
        device = boundaries.device

        b1, b2, b3 = boundaries[:, 0], boundaries[:, 1], boundaries[:, 2]
        N = predicted_count.float()

        num_queries = torch.zeros(BS, dtype=torch.long, device=device)

        mask1 = N <= b1
        mask2 = (N > b1) & (N <= b2)
        mask3 = (N > b2) & (N <= b3)
        mask4 = N > b3

        num_queries[mask1] = query_levels[0]
        num_queries[mask2] = query_levels[1]
        num_queries[mask3] = query_levels[2]
        num_queries[mask4] = query_levels[3]

        return num_queries

    def update_smoothness(self, epoch, total_epochs, min_smoothness=0.5):
        """动态调整平滑系数"""
        if epoch == 0:
            return

        current = self.smoothness.item()
        target = min_smoothness
        decay_rate = (current - target) / total_epochs
        new_value = max(current - decay_rate, target)

        self.smoothness.fill_(new_value)


class QualityAwareQueryInitializer(nn.Module):
    """
    质量感知的查询初始化 - 数值稳定版本
    """

    def __init__(self, feature_dim=256, num_heads=8, max_queries=1500):
        super().__init__()
        self.feature_dim = feature_dim
        self.max_queries = max_queries

        # 🔥 简化网络结构，提高稳定性
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feature_dim, feature_dim // 16, 1),
            nn.LayerNorm([feature_dim // 16, 1, 1]),  # 添加归一化
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim // 16, feature_dim, 1),
            nn.Sigmoid()
        )

        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

        # 质量预测
        self.quality_predictor = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // 2, 3, padding=1),
            nn.GroupNorm(32, feature_dim // 2),  # 使用 GroupNorm
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim // 2, 1, 1),
            nn.Sigmoid()
        )

        # 坐标回归
        self.coord_regressor = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // 2, 3, padding=1),
            nn.GroupNorm(32, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim // 2, 4, 1),
            nn.Sigmoid()
        )

        # 保守初始化
        self._init_weights()

    def _init_weights(self):
        """保守初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, encoder_features, num_queries):
        """
        前向传播 - 完全重写
        """
        BS, C, H, W = encoder_features.shape
        device = encoder_features.device

        # 🔥 Step 1: 输入清理
        encoder_features = sanitize_tensor(
            encoder_features,
            name="encoder_features",
            min_val=-5.0,
            max_val=5.0
        )

        # 处理标量输入
        if isinstance(num_queries, int):
            num_queries = torch.full((BS,), num_queries, dtype=torch.long, device=device)

        # 🔥 Step 2: 处理空特征图
        if H == 0 or W == 0 or H * W < 10:
            print(f"⚠️ Warning: Invalid feature map size (H={H}, W={W})")
            max_K = num_queries.max().item()
            # 返回中心位置的默认框
            reference_points = torch.zeros(BS, max_K, 4, device=device)
            reference_points[..., :2] = 0.5  # 中心位置
            reference_points[..., 2:] = 0.1  # 小尺寸
            quality_scores = torch.ones(BS, max_K, device=device) * 0.5
            return reference_points, quality_scores

        # 🔥 Step 3: 注意力机制（带异常处理）
        try:
            # 通道注意力
            channel_attn = self.channel_attention(encoder_features)
            channel_attn = torch.clamp(channel_attn, min=0.0, max=1.0)
            feat_ca = encoder_features * channel_attn

            # 空间注意力
            feat_max = torch.max(feat_ca, dim=1, keepdim=True)[0]
            feat_avg = torch.mean(feat_ca, dim=1, keepdim=True)
            spatial_input = torch.cat([feat_max, feat_avg], dim=1)
            spatial_attn = self.spatial_attention(spatial_input)
            spatial_attn = torch.clamp(spatial_attn, min=0.0, max=1.0)
            feat_refined = feat_ca * spatial_attn

        except Exception as e:
            print(f"❌ Attention error: {e}")
            feat_refined = encoder_features

        # 清理特征
        feat_refined = sanitize_tensor(feat_refined, "feat_refined")

        # 🔥 Step 4: 质量预测
        quality_map = self.quality_predictor(feat_refined).squeeze(1)
        quality_map = torch.clamp(quality_map, min=0.0, max=1.0)

        # 🔥 Step 5: 坐标回归（关键：严格限制范围）
        coords_map = self.coord_regressor(feat_refined).permute(0, 2, 3, 1)
        # 限制在 [0.1, 0.9] 范围，避免边界值
        coords_map = torch.clamp(coords_map, min=0.1, max=0.9)

        # 🔥 Step 6: Top-K 选择（安全版本）
        max_K = num_queries.max().item()
        quality_flat = quality_map.flatten(1)  # (BS, H*W)

        actual_k = min(max_K, quality_flat.shape[1])
        if actual_k == 0:
            reference_points = torch.zeros(BS, max_K, 4, device=device)
            reference_points[..., :2] = 0.5
            reference_points[..., 2:] = 0.1
            quality_scores = torch.ones(BS, max_K, device=device) * 0.5
            return reference_points, quality_scores

        # Top-K
        topk_values, topk_indices = torch.topk(quality_flat, actual_k, dim=1)

        # 转换索引
        topk_y = topk_indices // W
        topk_x = topk_indices % W

        # 🔥 Step 7: 收集坐标（带边界检查）
        reference_points = torch.zeros(BS, max_K, 4, device=device)
        reference_points[..., :2] = 0.5  # 默认中心
        reference_points[..., 2:] = 0.1  # 默认小尺寸

        quality_scores = torch.zeros(BS, max_K, device=device)

        for b in range(BS):
            K = min(num_queries[b].item(), actual_k)
            if K > 0:
                try:
                    y_indices = torch.clamp(topk_y[b, :K], 0, H-1)
                    x_indices = torch.clamp(topk_x[b, :K], 0, W-1)
                    coords = coords_map[b, y_indices, x_indices, :]
                    reference_points[b, :K] = coords
                    quality_scores[b, :K] = topk_values[b, :K]
                except Exception as e:
                    print(f"❌ Coord gathering error for batch {b}: {e}")

        # 最终清理
        reference_points = torch.clamp(reference_points, min=0.05, max=0.95)
        quality_scores = torch.clamp(quality_scores, min=0.0, max=1.0)

        return reference_points, quality_scores


class DynamicQueryModule(nn.Module):
    """
    动态查询总模块 - 数值稳定版本
    """

    def __init__(self,
                 feature_dim=256,
                 num_boundaries=3,
                 max_objects=1500,
                 query_levels=None,
                 initial_smoothness=1.0):
        super().__init__()

        if query_levels is None:
            query_levels = [300, 500, 900, 1500]
        self.query_levels = query_levels

        self.boundary_predictor = LearnableBoundaryPredictor(
            feature_dim=feature_dim,
            num_boundaries=num_boundaries,
            max_objects=max_objects,
            initial_smoothness=initial_smoothness
        )

        self.query_initializer = QualityAwareQueryInitializer(
            feature_dim=feature_dim,
            max_queries=max(query_levels)
        )

    def forward(self, density_feature, encoder_feature, real_counts=None, training=True):
        """
        前向传播 - 带完整错误处理
        """
        BS = density_feature.shape[0]
        device = density_feature.device

        # 🔥 预测边界（带异常捕获）
        try:
            boundaries, raw_boundaries = self.boundary_predictor(density_feature)
        except Exception as e:
            print(f"❌ Boundary prediction error: {e}")
            # 使用默认边界
            boundaries = torch.tensor(
                [[300, 600, 1000]] * BS,
                dtype=torch.float32,
                device=device
            )
            raw_boundaries = torch.zeros(BS, 3, device=device)

        outputs = {
            'boundaries': boundaries,
            'raw_boundaries': raw_boundaries,
        }

        # 🔥 确定查询数量
        if training and real_counts is not None:
            try:
                interval_probs = self.boundary_predictor.compute_interval_probabilities(
                    boundaries, real_counts
                )
                outputs['interval_probs'] = interval_probs
                num_queries = self._get_expected_query_number(interval_probs)
            except Exception as e:
                print(f"❌ Interval probability error: {e}")
                # 使用默认查询数量
                num_queries = torch.full((BS,), self.query_levels[1], dtype=torch.long, device=device)
                outputs['interval_probs'] = None
        else:
            predicted_count = self._estimate_object_count(density_feature)
            num_queries = self.boundary_predictor.get_query_number(
                boundaries, predicted_count, self.query_levels
            )
            outputs['predicted_count'] = predicted_count

        outputs['num_queries'] = num_queries

        # 🔥 初始化查询（带异常捕获）
        try:
            reference_points, quality_scores = self.query_initializer(
                encoder_feature, num_queries
            )
            outputs['reference_points'] = reference_points
            outputs['quality_scores'] = quality_scores
        except Exception as e:
            print(f"❌ Query initialization error: {e}")
            import traceback
            traceback.print_exc()
            # 使用安全的默认值
            max_K = max(self.query_levels)
            reference_points = torch.zeros(BS, max_K, 4, device=device)
            reference_points[..., :2] = 0.5
            reference_points[..., 2:] = 0.1
            quality_scores = torch.ones(BS, max_K, device=device) * 0.5
            outputs['reference_points'] = reference_points
            outputs['quality_scores'] = quality_scores

        return outputs

    def _get_expected_query_number(self, interval_probs):
        """根据概率计算期望查询数量"""
        BS = interval_probs.shape[0]
        device = interval_probs.device

        query_levels_tensor = torch.tensor(
            self.query_levels, dtype=torch.float32, device=device
        )

        expected_queries = (interval_probs * query_levels_tensor).sum(dim=1)

        num_queries = torch.zeros(BS, dtype=torch.long, device=device)
        for b in range(BS):
            diffs = torch.abs(query_levels_tensor - expected_queries[b])
            num_queries[b] = self.query_levels[torch.argmin(diffs)]

        return num_queries

    def _estimate_object_count(self, density_feature):
        """估计目标数量"""
        # 简化版本：使用特征图的平均激活
        count = torch.mean(torch.abs(density_feature), dim=(1, 2, 3)) * 100
        return torch.clamp(count, min=1.0, max=self.query_levels[-1]).long()

    def update_smoothness(self, epoch, total_epochs):
        """更新平滑系数"""
        self.boundary_predictor.update_smoothness(epoch, total_epochs)


def build_dynamic_query_module(args):
    """工厂函数"""
    return DynamicQueryModule(
        feature_dim=args.hidden_dim,
        num_boundaries=getattr(args, 'num_boundaries', 3),
        max_objects=getattr(args, 'max_objects', 1500),
        query_levels=getattr(args, 'dynamic_query_levels', [300, 500, 900, 1500]),
        initial_smoothness=getattr(args, 'initial_smoothness', 1.0)
    )