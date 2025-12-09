import torch
from dynamic_query import DynamicQueryModule

# 测试动态查询模块
module = DynamicQueryModule(
    feature_dim=256,
    num_boundaries=3,
    max_objects=1500,
    query_levels=[300, 500, 900, 1500]
)

# 模拟输入
BS = 4
density_feat = torch.randn(BS, 256, 32, 32)
encoder_feat = torch.randn(BS, 256, 32, 32)
real_counts = torch.tensor([50, 150, 800, 1500])

# 前向传播
outputs = module(density_feat, encoder_feat, real_counts, training=True)

print("✅ Boundaries:", outputs['boundaries'].shape)
print("✅ Interval Probs:", outputs['interval_probs'].shape)
print("✅ Num Queries:", outputs['num_queries'])
print("✅ Reference Points:", outputs['reference_points'].shape)
print("✅ Quality Scores:", outputs['quality_scores'].shape)