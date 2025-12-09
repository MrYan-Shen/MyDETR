_base_ = ['coco_transformer.py']

num_classes = 9  # 数据集的目标类别数（AI-TOD-V2为8类+1背景）
lr = 0.0001  # 基础学习率
param_dict_type = 'default'  # 参数分组策略，'default'表示默认分组
lr_backbone = 1e-05  # Backbone网络的学习率，通常设置得比整体学习率小，用于微调预训练模型
lr_backbone_names = ['backbone.0']  # 指定哪些模块使用lr_backbone这个学习率
lr_linear_proj_names = ['reference_points', 'sampling_offsets']  # 可变形注意力中线性投影层的学习率应用对象
lr_linear_proj_mult = 0.1  # 上述线性投影层学习率的乘数（lr * 0.1）
ddetr_lr_param = False  # 是否使用Deformable DETR特定的学习率参数设置
batch_size = 1  # 批处理大小。由于模型较大且输入图像大，batch_size设为1是常见选择。
weight_decay = 0.0001  # L2权重衰减，用于防止过拟合
epochs = 24  # 训练总轮数
lr_drop = 11  # 学习率下降的轮次（旧schedule）
save_checkpoint_interval = 1  # 每隔多少epoch保存一次检查点
clip_max_norm = 0.1  # 梯度裁剪的最大范数，用于稳定训练
onecyclelr = False  # 是否使用OneCycle学习率调度器
multi_step_lr = True  # 是否使用多步长学习率调度器
lr_drop_list = [13, 23]  # 多步长调度器下，学习率下降的轮次列表
val_epoch = [23]  # 指定在哪些轮次后进行验证
# dataset_file='aitod_v2'

# 分类阈值
ccm_params = [10, 100, 500]
ccm_cls_num = 4
dynamic_query_list = [300, 500, 900, 1500]
find_unused_parameters = False

# 模型骨架 (Backbone) 与通用DETR参数
modelname = 'dqdetr'  # 模型名称
frozen_weights = None  # 是否冻结某些权重
backbone = 'resnet50'  # 主干网络
use_checkpoint = False  # 是否使用梯度检查点（节省显存，但减慢速度）
dilation = False  # 是否在ResNet最后阶段使用空洞卷积
position_embedding = 'sine'  # 位置编码类型，'sine'表示正弦编码
pe_temperatureH = 20  # 高度方向位置编码的温度参数
pe_temperatureW = 20  # 宽度方向位置编码的温度参数
return_interm_indices = [0, 1, 2, 3]  # 指定返回Backbone哪几个阶段的特征图
backbone_freeze_keywords = None  # 冻结Backbone中包含特定关键词的层
enc_layers = 6  # Transformer编码器层数,原6
dec_layers = 6  # Transformer解码器层数,原6
unic_layers = 0  # 未知
pre_norm = False  # 是否使用Pre-Norm结构（Transformer中）
dim_feedforward = 2048  # Transformer中FFN层的隐藏维度
hidden_dim = 256  # Transformer的特征维度
dropout = 0.0  # Dropout率
nheads = 8  # 多头注意力机制中的头数
num_queries = 900  # **默认的查询数量**（在非动态模式下或作为初始值）
query_dim = 4
num_patterns = 0
pdetr3_bbox_embed_diff_each_layer = False
pdetr3_refHW = -1
random_refpoints_xy = False
fix_refpoints_hw = -1
dabdetr_yolo_like_anchor_update = False
dabdetr_deformable_encoder = False
dabdetr_deformable_decoder = False
use_deformable_box_attn = False
box_attn_type = 'roi_align'
dec_layer_number = None
num_feature_levels = 5  # 使用的多尺度特征图数量
enc_n_points = 4  # 编码器中可变形注意力的参考点数量
dec_n_points = 4  # 解码器中可变形注意力的参考点数量
decoder_layer_noise = False
dln_xy_noise = 0.2
dln_hw_noise = 0.2
add_channel_attention = False
add_pos_value = False
two_stage_type = 'standard'
two_stage_pat_embed = 0
two_stage_add_query_num = 0
two_stage_bbox_embed_share = False
two_stage_class_embed_share = False
two_stage_learn_wh = False
two_stage_default_hw = 0.05
two_stage_keep_all_tokens = False
num_select = 300
transformer_activation = 'relu'
batch_norm_type = 'FrozenBatchNorm2d'
masks = False
aux_loss = True  # 是否使用辅助损失（在每个解码层都计算损失）
set_cost_class = 2.0  # 匈牙利匹配中类别误差的权重
set_cost_bbox = 5.0  # 匈牙利匹配中边界框L1误差的权重
set_cost_giou = 2.0  # 匈牙利匹配中GIoU误差的权重
cls_loss_coef = 1.0  # 最终损失函数中分类损失的系数
mask_loss_coef = 1.0  # 最终损失函数中标记损失的系数
dice_loss_coef = 1.0
bbox_loss_coef = 5.0  # 最终损失函数中边界框L1损失的系数
giou_loss_coef = 2.0  # 最终损失函数中GIoU损失的系数
enc_loss_coef = 1.0
interm_loss_coef = 1.0
no_interm_box_loss = False
focal_alpha = 0.25  # Focal Loss中的alpha参数

decoder_sa_type = 'sa'
matcher_type = 'HungarianMatcher'  # 匹配器类型，匈牙利匹配
decoder_module_seq = ['sa', 'ca', 'ffn']
nms_iou_threshold = -1

dec_pred_bbox_embed_share = True
dec_pred_class_embed_share = True

# for dn
use_dn = True
dn_number = 100
dn_box_noise_scale = 0.4
dn_label_noise_ratio = 0.5
embed_init_tgt = False
dn_labelbook_size = 91
match_unstable_error = True

# for ema
use_ema = False
ema_decay = 0.9997
ema_epoch = 0

use_detached_boxes_dec_out = False

# 🔥 动态查询机制参数
use_dynamic_query = True  # 启用动态查询
num_boundaries = 3  # 边界数量
max_objects = 1500  # 最大目标数量
dynamic_query_levels = [300, 500, 900, 1500]  # 四个查询等级
initial_smoothness = 1.0  # 初始平滑系数

# 损失权重
loss_interval_coef = 1.0  # 区间分类损失权重
loss_boundary_coef = 0.01  # 边界L2正则化权重
