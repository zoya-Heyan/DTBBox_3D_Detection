import torch
import numpy as np

def box_iou(boxes1, boxes2):
    """
    计算边界框的 IoU
    Args:
        boxes1: [N, 4] 边界框，格式为 [x1, y1, x2, y2]
        boxes2: [M, 4] 边界框，格式为 [x1, y1, x2, y2]
    Returns:
        iou: [N, M] IoU 矩阵
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou

def box_nms(boxes, scores, iou_threshold=0.5):
    """
    边界框 NMS
    Args:
        boxes: [N, 4] 边界框，格式为 [x1, y1, x2, y2]
        scores: [N] 置信度得分
        iou_threshold: IoU 阈值
    Returns:
        keep: 保留的边界框索引
    """
    if len(boxes) == 0:
        return []

    # 按得分排序
    order = scores.argsort(descending=True)
    keep = []

    while order.numel() > 0:
        i = order[0]
        keep.append(i)

        if order.numel() == 1:
            break

        # 计算与其他框的 IoU
        iou = box_iou(boxes[i].unsqueeze(0), boxes[order[1:]])[0]

        # 保留 IoU 小于阈值的框
        inds = torch.nonzero(iou < iou_threshold).squeeze(1)
        order = order[inds + 1]

    return keep