import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_loss(preds, targets, config):
    """
    计算损失
    Args:
        preds: 预测结果
        targets: 目标值
        config: 配置
    Returns:
        total_loss: 总损失 (tensor)
    """
    device = preds[0]["locations"].device if preds and preds[0]["locations"].numel() > 0 else torch.device("cpu")
    total_loss = torch.zeros(1, device=device).requires_grad_(True)

    for pred, target in zip(preds, targets):
        if pred["locations"].numel() == 0:
            continue

        # 位置损失
        loc_loss = F.smooth_l1_loss(pred["locations"], target["locations"])
        total_loss = total_loss + config.loss.abs_weight * loc_loss

        # 尺寸损失
        dim_loss = F.smooth_l1_loss(pred["dimensions"], target["dimensions"])
        total_loss = total_loss + config.loss.dim_weight * dim_loss

        # 偏航角损失
        yaw_loss = F.smooth_l1_loss(pred["yaws"], target["yaws"])
        total_loss = total_loss + config.loss.yaw_weight * yaw_loss

    return total_loss.squeeze()