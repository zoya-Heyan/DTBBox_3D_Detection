import torch
import numpy as np

def compute_ap(precision, recall):
    """
    计算 AP
    Args:
        precision: 精度曲线
        recall: 召回曲线
    Returns:
        ap: 平均精度
    """
    # 确保精度曲线是单调递减的
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i+1])

    # 计算 AP
    ap = 0.0
    for i in range(1, len(recall)):
        ap += (recall[i] - recall[i-1]) * precision[i]

    return ap

def evaluate(preds, targets):
    """
    评估模型性能
    Args:
        preds: 预测结果
        targets: 目标值
    Returns:
        metrics: 评估指标
    """
    # 这里实现评估逻辑
    # 简化版本：计算位置误差
    loc_errors = []

    for pred, target in zip(preds, targets):
        if pred["locations"].numel() > 0 and target["locations"].numel() > 0:
            # 假设一对一匹配
            loc_error = torch.norm(pred["locations"] - target["locations"], dim=1).mean()
            loc_errors.append(loc_error.item())

    if loc_errors:
        avg_loc_error = np.mean(loc_errors)
    else:
        avg_loc_error = float('inf')

    metrics = {
        "avg_loc_error": avg_loc_error
    }

    return metrics