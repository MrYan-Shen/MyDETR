"""
Dynamic Query Mechanism for Adaptive Object Detection
动态查询机制：基于可学习边界和软区间分配
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def safe_prob_normalize(probs):
    """安全概率归一化"""
    denom = probs.sum(dim=1, keepdim=True)
    denom = denom.clamp(min=1e-6) # 防止除以0
    return probs / denom
class LearnableBoundaryPredictor(nn.Module):
    """
    可学习边界预测器
    功能：根据密度特征预测边界[b1, b2, b3]，将目标数量分为4个区间
    """

    def __init__(self, feature_dim=256, num_boundaries=3, max_objects=1500,
                 initial_smoothness=1.0):
        """
        参数:
            feature_dim: 输入特征维度
            num_boundaries: 边界数量（默认3个，划分4个区间）
            max_objects: 最大目标数量
            initial_smoothness: 初始平滑系数r
        """
        super().__init__()
        self.num_boundaries = num_boundaries
        self.max_objects = max_objects

        # 全局特征提取
        self.global_pool_avg = nn.AdaptiveAvgPool2d(1)
        self.global_pool_max = nn.AdaptiveMaxPool2d(1)

        # 边界预测网络：融合avg和max pooling特征
        self.boundary_predictor = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 2, num_boundaries)
        )

        # 平滑系数r（可学习）
        self.register_buffer('smoothness', torch.tensor(initial_smoothness))

        # 初始化边界预测器，使输出接近[0.25, 0.5, 0.75] * max_objects
        nn.init.constant_(self.boundary_predictor[-1].bias, 0.0)
        with torch.no_grad():
            # 让初始边界大致落在均匀分布
            self.boundary_predictor[-1].weight.data *= 0.01

    def forward(self, density_feature):
        """
        前向传播
        输入:
            density_feature: (BS, C, H, W) - CCM输出的密度特征
        输出:
            boundaries: (BS, num_boundaries) - 学习到的边界值 [b1, b2, b3]
            raw_boundaries: (BS, num_boundaries) - 原始边界值 [t1, t2, t3]
        """
        # 🔥 输入检查
        if torch.isnan(density_feature).any() or torch.isinf(density_feature).any():
            # print("  Warning: density_feature has NaN/Inf, applying fix") # 可选：注释掉避免刷屏
            density_feature = torch.nan_to_num(density_feature, nan=0.0, posinf=1.0, neginf=0.0)
            density_feature = density_feature.clamp(min=-10.0, max=10.0)

        # 1. 全局特征提取
        feat_avg = self.global_pool_avg(density_feature).flatten(1)  # (BS, C)
        feat_max = self.global_pool_max(density_feature).flatten(1)  # (BS, C)

        # 🔥 特征裁剪
        feat_avg = feat_avg.clamp(min=-10.0, max=10.0)
        feat_max = feat_max.clamp(min=-10.0, max=10.0)

        global_feat = torch.cat([feat_avg, feat_max], dim=1)  # (BS, 2C)

        # 2. 预测原始边界
        raw_boundaries = self.boundary_predictor(global_feat)  # (BS, 3)
        raw_boundaries = raw_boundaries.clamp(min=-10.0, max=10.0)

        # 3. 累加Softplus
        softplus_values = F.softplus(raw_boundaries)
        softplus_values = softplus_values.clamp(min=1e-4, max=self.max_objects / 4)
        boundaries = torch.cumsum(softplus_values, dim=1)

        # 4. 归一化到 [0, max_objects] 范围
        boundaries = boundaries.clamp(min=1.0, max=self.max_objects * 0.9)

        # ==================== 🔥 修复开始 ====================
        # 修复说明：消除 boundaries[:, i] = ... 的原位修改
        # 改为使用列表收集每一列，最后 stack

        boundaries_list = []
        # 第一个边界 b1 直接取值
        boundaries_list.append(boundaries[:, 0])

        min_gap = 10.0
        for i in range(1, boundaries.shape[1]):
            # 获取上一个已处理的边界（来自列表，而不是原张量）
            prev_b = boundaries_list[-1]
            # 获取当前预测的原始边界
            curr_b = boundaries[:, i]

            # 计算新的当前边界：max(当前值, 上一个值 + 间隔)
            # 这会创建一个新的张量 new_b，而不是修改原张量
            new_b = torch.max(curr_b, prev_b + min_gap)
            boundaries_list.append(new_b)

        # 重新堆叠回 (BS, 3)
        boundaries = torch.stack(boundaries_list, dim=1)
        # ==================== 🔥 修复结束 ====================

        return boundaries, raw_boundaries

    def compute_interval_probabilities(self, boundaries, real_count):
        """
        计算目标数量real_count属于各区间的概率分布（软区间分配）
        输入:
            boundaries: (BS, 3) - 边界值 [b1, b2, b3]
            real_count: (BS,) - 真实目标数量
        输出:
            probs: (BS, 4) - 四个区间的概率分布
        """

        b1, b2, b3 = boundaries[:, 0], boundaries[:, 1], boundaries[:, 2]
        N = real_count.float().unsqueeze(1)
        r = self.smoothness.clamp(min=0.1, max=5.0)  # 限制范围

        # 使用 tanh 替代 sigmoid，数值更稳定
        def soft_interval(x, lower, upper, r):
            """软区间指示函数"""
            left = torch.tanh((x - lower) / r)
            right = torch.tanh((upper - x) / r)
            return ((left + 1) * (right + 1) / 4).clamp(0, 1)

        # 计算四个区间的概率
        p1 = soft_interval(N, 0, b1, r)
        p2 = soft_interval(N, b1, b2, r)
        p3 = soft_interval(N, b2, b3, r)
        p4 = soft_interval(N, b3, self.max_objects, r)

        probs = torch.cat([p1, p2, p3, p4], dim=1)
        probs = safe_prob_normalize(probs)

        return probs

    def get_query_number(self, boundaries, predicted_count, query_levels):
        """
        推理阶段：根据预测的目标数量和边界，确定查询数量
        输入:
            boundaries: (BS, 3) - 边界值
            predicted_count: (BS,) - 预测的目标数量
            query_levels: list[int] - 四个查询等级，如[500, 1000, 1500, 2000]
        输出:
            num_queries: (BS,) - 每个样本的查询数量
        """
        BS = boundaries.shape[0]
        device = boundaries.device

        b1, b2, b3 = boundaries[:, 0], boundaries[:, 1], boundaries[:, 2]
        N = predicted_count.float()

        # 根据N落在哪个区间，选择对应的查询数量
        num_queries = torch.zeros(BS, dtype=torch.long, device=device)

        # 区间1: N <= b1
        mask1 = N <= b1
        num_queries[mask1] = query_levels[0]

        # 区间2: b1 < N <= b2
        mask2 = (N > b1) & (N <= b2)
        num_queries[mask2] = query_levels[1]

        # 区间3: b2 < N <= b3
        mask3 = (N > b2) & (N <= b3)
        num_queries[mask3] = query_levels[2]

        # 区间4: N > b3
        mask4 = N > b3
        num_queries[mask4] = query_levels[3]

        return num_queries

    def update_smoothness(self, epoch, total_epochs, min_smoothness=0.1):
        """
        动态调整平滑系数r：从initial_smoothness逐渐衰减到min_smoothness
        """
        old_val = self.smoothness.item()
        decay_rate = (old_val - min_smoothness) / total_epochs
        new_smoothness = max(old_val - decay_rate, min_smoothness)

        # 🔥 限制单次变化幅度
        max_change = 0.5
        if abs(new_smoothness - old_val) > max_change:
            new_smoothness = old_val + max_change * (1 if new_smoothness > old_val else -1)

        self.smoothness.fill_(new_smoothness)


class QualityAwareQueryInitializer(nn.Module):
    """
    质量感知的查询位置初始化
    功能：通过双重注意力机制（通道+空间）预测高质量的查询初始位置
    """

    def __init__(self, feature_dim=256, num_heads=8, max_queries=1500):
        super().__init__()
        self.feature_dim = feature_dim
        self.max_queries = max_queries

        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feature_dim, feature_dim // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim // 16, feature_dim, 1),
            nn.Sigmoid()
        )

        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

        # 位置质量预测（用于筛选Top-K位置）
        self.quality_predictor = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, 1, 1),
            nn.Sigmoid()
        )

        # 坐标回归头（预测4维坐标 cx, cy, w, h）
        self.coord_regressor = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, 4, 1),
            nn.Sigmoid()  # 归一化到[0,1]
        )

    def forward(self, encoder_features, num_queries):
        """
        前向传播
        输入:
            encoder_features: (BS, C, H, W) - encoder输出的特征图
            num_queries: (BS,) 或 int - 每个样本需要的查询数量
        输出:
            reference_points: (BS, num_queries, 4) - 初始参考点 (cx, cy, w, h)
            quality_scores: (BS, num_queries) - 质量分数
        """
        BS, C, H, W = encoder_features.shape
        device = encoder_features.device
        #  输入检查
        if torch.isnan(encoder_features).any() or torch.isinf(encoder_features).any():
            encoder_features = torch.nan_to_num(encoder_features, nan=0.0).clamp(-10.0, 10.0)

        # 如果num_queries是标量，转为tensor
        if isinstance(num_queries, int):
            num_queries = torch.full((BS,), num_queries, dtype=torch.long, device=device)

        # 处理空特征图
        if H == 0 or W == 0:
            print(f"  Warning: Empty feature map (H={H}, W={W}), using random initialization")
            max_K = num_queries.max().item()
            reference_points = torch.rand(BS, max_K, 4, device=device) * 0.5 + 0.25
            quality_scores = torch.ones(BS, max_K, device=device) * 0.5
            return reference_points, quality_scores

        # 1. 通道注意力
        channel_attn = self.channel_attention(encoder_features)
        feat_ca = encoder_features * channel_attn

        # 2. 空间注意力
        feat_max = torch.max(feat_ca, dim=1, keepdim=True)[0]
        feat_avg = torch.mean(feat_ca, dim=1, keepdim=True)
        spatial_input = torch.cat([feat_max, feat_avg], dim=1)
        spatial_attn = self.spatial_attention(spatial_input)
        feat_refined = feat_ca * spatial_attn

        # 3. 质量预测
        quality_map = self.quality_predictor(feat_refined).squeeze(1)  # (BS, H, W)
        # 防止质量图异常
        quality_map = torch.nan_to_num(quality_map, nan=0.5, posinf=1.0, neginf=0.0)
        quality_map = quality_map.clamp(min=0.0, max=1.0)

        # 4. 坐标回归
        coords_map = self.coord_regressor(feat_refined).permute(0, 2, 3, 1)
        coords_map = torch.nan_to_num(coords_map, nan=0.5)
        # 修改：限制坐标范围，留出eps余量，防止inverse_sigmoid爆炸
        coords_map = coords_map.clamp(min=0.05, max=0.95)

        # 5. Top-K选择（为每个样本选择不同数量的查询）
        max_K = num_queries.max().item()
        quality_flat = quality_map.flatten(1)  # (BS, H*W)

        # 选择Top-K位置
        # topk_values, topk_indices = torch.topk(quality_flat, max_K, dim=1)  # (BS, max_K)
        # 核心修改：确保 k 不超过张量实际大小
        actual_k = min(max_K, quality_flat.shape[1])
        if actual_k == 0:
            # 处理特征图 H*W=0 的极端情况，返回空张量
            topk_values = torch.empty((BS, max_K), dtype=torch.float32, device=device)
            topk_indices = torch.empty((BS, max_K), dtype=torch.long, device=device)
        else:
            # 选择Top-K位置
            topk_values_actual, topk_indices_actual = torch.topk(quality_flat, actual_k, dim=1)

            # 如果实际选择的 < max_K，用0填充至 max_K
            topk_values = torch.zeros(BS, max_K, device=device)
            topk_indices = torch.zeros(BS, max_K, dtype=torch.long, device=device)
            topk_values[:, :actual_k] = topk_values_actual
            topk_indices[:, :actual_k] = topk_indices_actual

        # 6. 提取对应位置的坐标
        # 将1D索引转换为2D坐标
        topk_y = topk_indices // W
        topk_x = topk_indices % W

        # 收集坐标
        reference_points_list = []
        quality_scores_list = []

        for b in range(BS):
            K = num_queries[b].item()
            # 确保 K 不超过 actual_k，防止越界
            K_safe = min(K, actual_k)

            if K_safe > 0:
                coords_selected = coords_map[b, topk_y[b, :K_safe], topk_x[b, :K_safe], :]
                quality_selected = topk_values[b, :K_safe]
            else:
                coords_selected = torch.empty((0, 4), device=device)
                quality_selected = torch.empty((0,), device=device)

            reference_points_list.append(coords_selected)
            quality_scores_list.append(quality_selected)

        # 7. Padding到统一长度（max_K）
        # 🔥🔥 修改：初始化为 0.5 (图像中心)，而不是 0.0
        # 0.0 经过 inverse_sigmoid 会变成负无穷或极大负数，导致 attention 采样越界和 NaN 梯度
        reference_points = torch.full((BS, max_K, 4), 0.5, device=device)
        # 为 padding 的位置设置合理的默认框：中心位置，小尺寸
        reference_points[..., 2:] = 0.1  # w, h = 0.1

        quality_scores = torch.zeros(BS, max_K, device=device)

        for b in range(BS):
            K = num_queries[b].item()
            K_safe = min(K, actual_k)
            if K_safe > 0:
                reference_points[b, :K_safe] = reference_points_list[b]
                quality_scores[b, :K_safe] = quality_scores_list[b]

        return reference_points, quality_scores


class DynamicQueryModule(nn.Module):
    """
    动态查询机制总模块
    整合：边界预测 + 查询初始化
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

        # 边界预测器
        self.boundary_predictor = LearnableBoundaryPredictor(
            feature_dim=feature_dim,
            num_boundaries=num_boundaries,
            max_objects=max_objects,
            initial_smoothness=initial_smoothness
        )

        # 查询初始化器
        self.query_initializer = QualityAwareQueryInitializer(
            feature_dim=feature_dim,
            max_queries=max(query_levels)
        )

    def forward(self, density_feature, encoder_feature, real_counts=None, training=True):
        """
        前向传播
        输入:
            density_feature: (BS, C, H, W) - CCM输出的密度特征
            encoder_feature: (BS, C, H, W) - Encoder输出的特征
            real_counts: (BS,) - 真实目标数量（仅训练时需要）
            training: bool - 是否训练模式
        输出:
            outputs: dict - 包含边界、查询数量、参考点等信息
        """
        BS = density_feature.shape[0]
        device = density_feature.device

        # 1. 预测边界
        boundaries, raw_boundaries = self.boundary_predictor(density_feature)

        outputs = {
            'boundaries': boundaries,  # (BS, 3)
            'raw_boundaries': raw_boundaries,  # (BS, 3)
        }

        if training and real_counts is not None:
            # 训练模式：计算软区间概率
            interval_probs = self.boundary_predictor.compute_interval_probabilities(
                boundaries, real_counts
            )
            outputs['interval_probs'] = interval_probs  # (BS, 4)

            # 使用真实目标数量确定查询数量（用于训练稳定性）
            # 或者使用期望查询数量
            num_queries = self._get_expected_query_number(interval_probs)

        else:
            # 推理模式：根据预测的目标数量确定查询数量
            # 这里需要从density_feature估计目标数量
            predicted_count = self._estimate_object_count(density_feature)
            num_queries = self.boundary_predictor.get_query_number(
                boundaries, predicted_count, self.query_levels
            )
            outputs['predicted_count'] = predicted_count

        outputs['num_queries'] = num_queries  # (BS,)

        # 2. 初始化查询位置
        try:
            reference_points, quality_scores = self.query_initializer(
                encoder_feature, num_queries
            )
            outputs['reference_points'] = reference_points
            outputs['quality_scores'] = quality_scores
        except Exception as e:
            print(f"Query initialization error: {e}")
            # 使用默认值
            max_K = max(self.query_levels)
            outputs['reference_points'] = torch.rand(
                BS, max_K, 4, device=device
            ) * 0.5 + 0.25
            outputs['quality_scores'] = torch.ones(
                BS, max_K, device=device
            ) * 0.5

        return outputs

    def _get_expected_query_number(self, interval_probs):
        """
        根据软区间概率计算期望查询数量
        """
        BS = interval_probs.shape[0]
        device = interval_probs.device

        query_levels_tensor = torch.tensor(
            self.query_levels, dtype=torch.float32, device=device
        )

        # 期望查询数量 = sum(p_i * K_i)
        expected_queries = (interval_probs * query_levels_tensor).sum(dim=1)

        # 取最接近的查询等级
        num_queries = torch.zeros(BS, dtype=torch.long, device=device)
        for b in range(BS):
            diffs = torch.abs(query_levels_tensor - expected_queries[b])
            num_queries[b] = self.query_levels[torch.argmin(diffs)]

        return num_queries

    def _estimate_object_count(self, density_feature):
        """
        从密度特征估计目标数量（简单求和）
        """
        # 这里可以根据CCM的输出进行估计
        # 简化版本：对密度图求和
        count = density_feature.sum(dim=(1, 2, 3))
        return count.long()

    def update_smoothness(self, epoch, total_epochs):
        """更新平滑系数"""
        self.boundary_predictor.update_smoothness(epoch, total_epochs)


def build_dynamic_query_module(args):
    """工厂函数：构建动态查询模块"""
    return DynamicQueryModule(
        feature_dim=args.hidden_dim,
        num_boundaries=getattr(args, 'num_boundaries', 3),
        max_objects=getattr(args, 'max_objects', 1500),
        query_levels=getattr(args, 'dynamic_query_levels', [300, 500, 900, 1500]),
        initial_smoothness=getattr(args, 'initial_smoothness', 1.0)
    )