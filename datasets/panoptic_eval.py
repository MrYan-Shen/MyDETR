# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import json
import os

import util.misc as utils

try:
    from panopticapi.evaluation import pq_compute
except ImportError:
    pass

# 全景分割评估器
class PanopticEvaluator(object):
    def __init__(self, ann_file, ann_folder, output_dir="panoptic_eval"):
        self.gt_json = ann_file
        self.gt_folder = ann_folder
        if utils.is_main_process():
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
        self.output_dir = output_dir  # 预测结果输出目录
        self.predictions = []  # 存储预测结果的列表

    # 更新评估器状态，添加新预测结果
    # 立即写入PNG数据，避免累积大内存占用；使用pop()移除已处理的字段；使用预测中提供的文件名，确保与标注匹配
    def update(self, predictions):
        for p in predictions:
            with open(os.path.join(self.output_dir, p["file_name"]), "wb") as f:
                f.write(p.pop("png_string"))

        self.predictions += predictions

    # 在分布式环境中同步各进程的预测结果
    def synchronize_between_processes(self):
        # 从所有进程收集预测结果
        all_predictions = utils.all_gather(self.predictions)
        # 合并所有进程的预测
        merged_predictions = []
        for p in all_predictions:
            merged_predictions += p  # 连接各进程的预测列表
        self.predictions = merged_predictions  # 更新内部状态

    # 执行最终评估并返回结果
    def summarize(self):
        if utils.is_main_process():
            json_data = {"annotations": self.predictions}
            # 预测JSON文件路径
            predictions_json = os.path.join(self.output_dir, "predictions.json")
            # 写入预测元数据
            with open(predictions_json, "w") as f:
                f.write(json.dumps(json_data))
            return pq_compute(self.gt_json, predictions_json, gt_folder=self.gt_folder, pred_folder=self.output_dir)
        # 非主进程不执行评估
        return None
