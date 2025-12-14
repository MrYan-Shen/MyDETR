# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""

import math
import os
import sys
from typing import Iterable

from util.utils import slprint, to_device

import torch
import torch.nn as nn
import torch.nn.functional as F
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator

print_freq = 5000
CCM_LOSS = torch.nn.CrossEntropyLoss()
ccm_coeff = 1

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    wo_class_error=False, lr_scheduler=None, args=None, logger=None, ema_m=None):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)

    ccm_params = args.ccm_params

    _cnt = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header, logger=logger):

        samples = samples.to(device)
        ccm_targets = []
        for i in range(len(targets)):
            tgt_num = targets[i]['labels'].shape[0]
            t = 0
            for j in range(len(ccm_params)):
                if tgt_num >= ccm_params[j]:
                    t = j + 1
            ccm_targets.append(t)

        ccm_targets = torch.tensor(ccm_targets, dtype=torch.int64).to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.amp.autocast('cuda',enabled=args.amp):
            if need_tgt_for_training:
                outputs = model(samples, targets)
            else:
                outputs = model(samples)

            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            ccm_loss = CCM_LOSS(outputs['pred_bbox_number'], ccm_targets)
            losses += ccm_coeff * ccm_loss

            # [Debug] 检查 Loss Keys
            if _cnt % 100 == 0 and utils.is_main_process():
                keys_in_loss = [k for k in loss_dict.keys() if 'dq' in k]
                print(f"[DEBUG Engine] Current Step {_cnt}, DQ Losses found: {keys_in_loss}")

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}

        loss_dict_reduced_scaled['ccm_loss'] = ccm_loss
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # --- 添加检查代码 ---
        if _cnt % 100 == 0 and utils.is_main_process():  # 每100个batch检查一次
            for name, param in model.named_parameters():
                if "dynamic_query_module" in name and param.requires_grad:
                    if param.grad is not None:
                        grad_mean = param.grad.abs().mean().item()
                        print(f"[Grad Check] {name}: {grad_mean:.6f}")
                    else:
                        print(f"[Grad Check] ❌ {name} bas NO GRADIENT!")
                    break  # 只检查第一个参数即可证明
        # ------------------

        # amp backward function
        if args.amp:
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # original backward function
            optimizer.zero_grad()
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        if args.onecyclelr:
            lr_scheduler.step()
        if args.use_ema:
            if epoch >= args.ema_epoch:
                ema_m.update(model)

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

    if getattr(criterion, 'loss_weight_decay', False):
        criterion.loss_weight_decay(epoch=epoch)
    if getattr(criterion, 'tuning_matching', False):
        criterion.tuning_matching(epoch)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if getattr(criterion, 'loss_weight_decay', False):
        resstat.update({f'weight_{k}': v for k,v in criterion.weight_dict.items()})
    return resstat


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None, logger=None):
    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    useCats = True
    try:
        useCats = args.useCats
    except:
        useCats = True
    if not useCats:
        print("useCats: {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(useCats))
    coco_evaluator = CocoEvaluator(base_ds, iou_types, useCats=useCats)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    _cnt = 0
    output_state_dict = {} # for debug only
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header, logger=logger):

        samples = samples.to(device)
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        with torch.amp.autocast('cuda',enabled=args.amp):
            if need_tgt_for_training:
                outputs = model(samples, targets)
            else:
                outputs = model(samples)

            loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict


        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}

        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes, outputs['num_select'])


        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}


        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

        if args.save_results:
            # res_score = outputs['res_score']
            # res_label = outputs['res_label']
            # res_bbox = outputs['res_bbox']
            # res_idx = outputs['res_idx']

            for i, (tgt, res, outbbox) in enumerate(zip(targets, results, outputs['pred_boxes'])):
                """
                pred vars:
                    K: number of bbox pred
                    score: Tensor(K),
                    label: list(len: K),
                    bbox: Tensor(K, 4)
                    idx: list(len: K)
                tgt: dict.

                """
                # compare gt and res (after postprocess)
                gt_bbox = tgt['boxes']
                gt_label = tgt['labels']
                gt_info = torch.cat((gt_bbox, gt_label.unsqueeze(-1)), 1)

                # img_h, img_w = tgt['orig_size'].unbind()
                # scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=0)
                # _res_bbox = res['boxes'] / scale_fct
                _res_bbox = outbbox
                _res_prob = res['scores']
                _res_label = res['labels']
                res_info = torch.cat((_res_bbox, _res_prob.unsqueeze(-1), _res_label.unsqueeze(-1)), 1)
                # import ipdb;ipdb.set_trace()

                if 'gt_info' not in output_state_dict:
                    output_state_dict['gt_info'] = []
                output_state_dict['gt_info'].append(gt_info.cpu())

                if 'res_info' not in output_state_dict:
                    output_state_dict['res_info'] = []
                output_state_dict['res_info'].append(res_info.cpu())

            # # for debug only
            # import random
            # if random.random() > 0.7:
            #     print("Now let's break")
            #     break

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

    if args.save_results:
        import os.path as osp

        # output_state_dict['gt_info'] = torch.cat(output_state_dict['gt_info'])
        # output_state_dict['res_info'] = torch.cat(output_state_dict['res_info'])
        savepath = osp.join(args.output_dir, 'results-{}.pkl'.format(utils.get_rank()))
        print("Saving res to {}".format(savepath))
        torch.save(output_state_dict, savepath)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]



    return stats, coco_evaluator
#
#
# """
# Train and eval functions used in main.py
# """
#
# import math
# import os
# import sys
# from typing import Iterable
#
# from util.utils import slprint, to_device
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import util.misc as utils
# from datasets.coco_eval import CocoEvaluator
# from datasets.panoptic_eval import PanopticEvaluator
#
# print_freq = 5000
# CCM_LOSS = torch.nn.CrossEntropyLoss()
# ccm_coeff = 1.0  # 确保是浮点数
#
#
# def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
#                     data_loader: Iterable, optimizer: torch.optim.Optimizer,
#                     device: torch.device, epoch: int, max_norm: float = 1.0,
#                     wo_class_error=False, lr_scheduler=None, args=None, logger=None, ema_m=None):
#     # 1. 模型参数健康检查 (防止由上一个Epoch遗留的NaN)
#     if epoch >= 1:
#         for name, param in model.named_parameters():
#             if param.requires_grad and (torch.isnan(param).any() or torch.isinf(param).any()):
#                 msg = f"❌ CRITICAL: NaN/Inf parameters detected at start of epoch {epoch}: {name}"
#                 print(msg)
#                 if logger: logger.info(msg)
#                 return {}
#
#     # 2. 兼容 DDP 和单机模式
#     model_core = model.module if hasattr(model, 'module') else model
#     model.train()
#     criterion.train()
#
#     # 3. 初始化指标记录器
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
#     if not wo_class_error:
#         metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
#     header = 'Epoch: [{}]'.format(epoch)
#
#     # 初始化混合精度 Scaler
#     scaler = torch.amp.GradScaler('cuda', enabled=args.amp)
#
#     try:
#         need_tgt_for_training = args.use_dn
#     except:
#         need_tgt_for_training = False
#
#     ccm_params = args.ccm_params if hasattr(args, 'ccm_params') else []
#
#     _cnt = 0
#     skip_count = 0
#
#     for samples, targets in metric_logger.log_every(data_loader, print_freq, header, logger=logger):
#         # --- 输入数据检查 ---
#         if torch.isnan(samples.tensors).any() or torch.isinf(samples.tensors).any():
#             print(f"Warning: Batch {_cnt} input contains NaN/Inf, skipping.")
#             skip_count += 1
#             continue
#
#         samples = samples.to(device)
#
#         # 准备 CCM Targets
#         ccm_targets_list = []
#         if ccm_params:
#             for i in range(len(targets)):
#                 tgt_num = targets[i]['labels'].shape[0]
#                 t = 0
#                 for j in range(len(ccm_params)):
#                     if tgt_num >= ccm_params[j]:
#                         t = j + 1
#                 ccm_targets_list.append(t)
#
#         if ccm_targets_list:
#             ccm_targets = torch.tensor(ccm_targets_list, dtype=torch.int64).to(device)
#         else:
#             ccm_targets = None
#
#         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
#
#         # --- 优化器清零 ---
#         optimizer.zero_grad(set_to_none=True)
#
#         # --- 前向传播 ---
#         try:
#             with torch.amp.autocast('cuda', enabled=args.amp):
#                 if need_tgt_for_training:
#                     outputs = model(samples, targets)
#                 else:
#                     outputs = model(samples)
#
#                 # 安全处理 pred_bbox_number (防止 NaN)
#                 if 'pred_bbox_number' in outputs:
#                     outputs['pred_bbox_number'] = torch.nan_to_num(outputs['pred_bbox_number'], nan=0.0).clamp(-10.0,
#                                                                                                                10.0)
#
#                 loss_dict = criterion(outputs, targets)
#                 weight_dict = criterion.weight_dict
#
#                 # 仅累加有效的 Loss
#                 losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys()
#                              if k in weight_dict and torch.isfinite(loss_dict[k]))
#
#                 # 计算 CCM Loss
#                 ccm_loss = torch.tensor(0.0, device=device)
#                 if ccm_targets is not None and 'pred_bbox_number' in outputs:
#                     # ccm_loss = CCM_LOSS(outputs['pred_bbox_number'], ccm_targets)
#                     # if torch.isfinite(ccm_loss):
#                     #     losses += ccm_coeff * ccm_loss
#                     # 确保 pred_bbox_number 没有 NaN
#                     pred_bbox_num = outputs['pred_bbox_number']
#                     if torch.isnan(pred_bbox_num).any():
#                         pred_bbox_num = torch.nan_to_num(pred_bbox_num, nan=0.0)
#
#                     ccm_loss_val = CCM_LOSS(pred_bbox_num, ccm_targets)
#
#                     if torch.isfinite(ccm_loss_val):
#                         # 显式转换 ccm_coeff 为 tensor 避免标量类型推断问题（可选）
#                         ccm_loss = ccm_loss_val
#                         losses += ccm_coeff * ccm_loss
#                     else:
#                         print(f"Warning: CCM Loss is NaN at batch {_cnt}")
#
#                 # 🔥 NaN 熔断机制：如果 Loss 无效，直接跳过，不进行反向传播
#                 # 这保护了模型权重不被污染，同时跳过 backward 防止报错
#                 if not torch.isfinite(losses):
#                     print(f"⚠️ Loss is {losses.item()}, skipping batch {_cnt}")
#                     skip_count += 1
#
#                     # 检查哪个 loss 出了问题
#                     for key, value in loss_dict.items():
#                         if not torch.isfinite(value):
#                             print(f"   {key}: {value.item()}")
#
#         except Exception as e:
#             print(f"❌ Forward pass error at batch {_cnt}: {e}")
#             skip_count += 1
#             continue
#
#         # --- 反向传播 ---
#         try:
#             if args.amp:
#                 # 1. Scale Loss & Backward
#                 scaler.scale(losses).backward()
#
#                 # 2. Unscale & Clip
#                 if max_norm > 0:
#                     scaler.unscale_(optimizer)  # 显式 unscale
#
#                     # 检查梯度范数
#                     total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
#
#                     # 仅打印警告，不跳过流程！
#                     # 如果 total_norm 是 NaN，clip_grad_norm_ 会把所有梯度设为 NaN
#                     # 接着 scaler.step 会检测到 NaN，从而跳过 optimizer.step 并减小 scale 因子
#                     if torch.isnan(total_norm) or torch.isinf(total_norm):
#                         print(
#                             f"⚠️ Warning: Gradients are NaN/Inf (Norm: {total_norm.item()}) at batch {_cnt}. Skipping step.")
#                         scaler.update()  # Update scaler to adjust scale factor
#                         optimizer.zero_grad()  # 清空梯度
#                         continue  # Skip optimizer step
#
#                 # 3. Step & Update (必须总是调用)
#                 scaler.step(optimizer)
#                 scaler.update()  # 重置 scaler 状态，防止 "unscale_ called twice" 错误
#
#             else:
#                 # 非 AMP 模式
#                 losses.backward()
#                 if max_norm > 0:
#                     total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
#                     if torch.isfinite(total_norm):
#                         optimizer.step()
#                     else:
#                         print(f"⚠️ Warning: Gradients are NaN/Inf at batch {_cnt}. Skipping step.")
#                 else:
#                     optimizer.step()
#
#         except Exception as e:
#             print(f"❌ Backward pass error at batch {_cnt}: {e}")
#             skip_count += 1
#             continue
#
#         # --- 学习率调度 ---
#         if args.onecyclelr:
#             lr_scheduler.step()
#         if args.use_ema and epoch >= args.ema_epoch:
#             ema_m.update(model)
#
#         # --- 日志记录 ---
#         loss_dict_reduced = utils.reduce_dict(loss_dict)
#         loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
#         loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
#
#         # 确保日志中的数值也是有限的
#         if torch.isfinite(ccm_loss):
#             loss_dict_reduced_scaled['ccm_loss'] = ccm_loss.item()
#
#         # 安全获取 loss value
#         loss_value = losses.item() if torch.isfinite(losses) else 0.0
#
#         metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
#         if 'class_error' in loss_dict_reduced:
#             metric_logger.update(class_error=loss_dict_reduced['class_error'])
#         metric_logger.update(lr=optimizer.param_groups[0]["lr"])
#
#         _cnt += 1
#         if args.debug and _cnt % 15 == 0:
#             print("BREAK!" * 5)
#             break
#
#     # --- Epoch 结束处理 ---
#     if skip_count > 0:
#         print(f"⚠️  Epoch {epoch} summary: Skipped {skip_count} batches due to instabilities.")
#
#     if getattr(criterion, 'loss_weight_decay', False):
#         criterion.loss_weight_decay(epoch=epoch)
#     if getattr(criterion, 'tuning_matching', False):
#         criterion.tuning_matching(epoch)
#
#     # 动态更新平滑系数
#     if hasattr(model_core, 'transformer') and hasattr(model_core.transformer, 'dynamic_query_module'):
#         model_core.transformer.dynamic_query_module.update_smoothness(epoch, args.epochs)
#         if logger:
#             logger.info(f"Updated smoothness")
#
#     # gather the stats from all processes
#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger)
#     resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
#     if getattr(criterion, 'loss_weight_decay', False):
#         resstat.update({f'weight_{k}': v for k, v in criterion.weight_dict.items()})
#     return resstat
#
#
# @torch.no_grad()
# def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False,
#              args=None, logger=None):
#     try:
#         need_tgt_for_training = args.use_dn
#     except:
#         need_tgt_for_training = False
#
#     model.eval()
#     criterion.eval()
#
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     if not wo_class_error:
#         metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
#     header = 'Test:'
#
#     iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
#     useCats = getattr(args, 'useCats', True)
#
#     coco_evaluator = CocoEvaluator(base_ds, iou_types, useCats=useCats)
#
#     panoptic_evaluator = None
#     if 'panoptic' in postprocessors.keys():
#         panoptic_evaluator = PanopticEvaluator(
#             data_loader.dataset.ann_file,
#             data_loader.dataset.ann_folder,
#             output_dir=os.path.join(output_dir, "panoptic_eval"),
#         )
#
#     output_state_dict = {}
#     for samples, targets in metric_logger.log_every(data_loader, print_freq, header, logger=logger):
#         samples = samples.to(device)
#         targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]
#
#         with torch.amp.autocast('cuda', enabled=args.amp):
#             if need_tgt_for_training:
#                 outputs = model(samples, targets)
#             else:
#                 outputs = model(samples)
#
#             loss_dict = criterion(outputs, targets)
#
#         weight_dict = criterion.weight_dict
#
#         loss_dict_reduced = utils.reduce_dict(loss_dict)
#         loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
#         loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
#
#         metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
#                              **loss_dict_reduced_scaled,
#                              **loss_dict_reduced_unscaled)
#         if 'class_error' in loss_dict_reduced:
#             metric_logger.update(class_error=loss_dict_reduced['class_error'])
#
#         orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
#         results = postprocessors['bbox'](outputs, orig_target_sizes, outputs['num_select'])
#
#         if 'segm' in postprocessors.keys():
#             target_sizes = torch.stack([t["size"] for t in targets], dim=0)
#             results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
#         res = {target['image_id'].item(): output for target, output in zip(targets, results)}
#
#         if coco_evaluator is not None:
#             coco_evaluator.update(res)
#
#         if panoptic_evaluator is not None:
#             res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
#             for i, target in enumerate(targets):
#                 image_id = target["image_id"].item()
#                 file_name = f"{image_id:012d}.png"
#                 res_pano[i]["image_id"] = image_id
#                 res_pano[i]["file_name"] = file_name
#             panoptic_evaluator.update(res_pano)
#
#         if args.save_results:
#             for i, (tgt, res, outbbox) in enumerate(zip(targets, results, outputs['pred_boxes'])):
#                 gt_bbox = tgt['boxes']
#                 gt_label = tgt['labels']
#                 gt_info = torch.cat((gt_bbox, gt_label.unsqueeze(-1)), 1)
#                 _res_bbox = outbbox
#                 _res_prob = res['scores']
#                 _res_label = res['labels']
#                 res_info = torch.cat((_res_bbox, _res_prob.unsqueeze(-1), _res_label.unsqueeze(-1)), 1)
#
#                 if 'gt_info' not in output_state_dict:
#                     output_state_dict['gt_info'] = []
#                 output_state_dict['gt_info'].append(gt_info.cpu())
#
#                 if 'res_info' not in output_state_dict:
#                     output_state_dict['res_info'] = []
#                 output_state_dict['res_info'].append(res_info.cpu())
#
#     if args.save_results:
#         import os.path as osp
#         savepath = osp.join(args.output_dir, 'results-{}.pkl'.format(utils.get_rank()))
#         print("Saving res to {}".format(savepath))
#         torch.save(output_state_dict, savepath)
#
#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger)
#     if coco_evaluator is not None:
#         coco_evaluator.synchronize_between_processes()
#         coco_evaluator.accumulate()
#         coco_evaluator.summarize()
#
#     panoptic_res = None
#     if panoptic_evaluator is not None:
#         panoptic_evaluator.synchronize_between_processes()
#         panoptic_res = panoptic_evaluator.summarize()
#
#     stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
#     if coco_evaluator is not None:
#         if 'bbox' in postprocessors.keys():
#             stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
#         if 'segm' in postprocessors.keys():
#             stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
#     if panoptic_res is not None:
#         stats['PQ_all'] = panoptic_res["All"]
#         stats['PQ_th'] = panoptic_res["Things"]
#         stats['PQ_st'] = panoptic_res["Stuff"]
#
#     return stats, coco_evaluator