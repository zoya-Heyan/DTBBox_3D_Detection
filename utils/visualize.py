import cv2
import numpy as np
import torch
from .geometry import project_3d_to_2d, compute_box_corners

def draw_3d_box(img, corners_2d):
    """
    在图像上绘制 3D 边界框
    Args:
        img: 图像
        corners_2d: [8, 2] 2D 角点
    Returns:
        img: 绘制后的图像
    """
    # 绘制底部
    bottom = [0, 1, 2, 3, 0]
    for i in range(4):
        p1 = tuple(corners_2d[bottom[i]].astype(int))
        p2 = tuple(corners_2d[bottom[i+1]].astype(int))
        cv2.line(img, p1, p2, (0, 255, 0), 2)

    # 绘制顶部
    top = [4, 5, 6, 7, 4]
    for i in range(4):
        p1 = tuple(corners_2d[top[i]].astype(int))
        p2 = tuple(corners_2d[top[i+1]].astype(int))
        cv2.line(img, p1, p2, (0, 255, 0), 2)

    # 绘制连接
    for i in range(4):
        p1 = tuple(corners_2d[i].astype(int))
        p2 = tuple(corners_2d[i+4].astype(int))
        cv2.line(img, p1, p2, (0, 255, 0), 2)

    return img

def visualize_prediction(img, pred, calib):
    """
    可视化预测结果
    Args:
        img: 图像
        pred: 预测结果
        calib: 相机标定矩阵
    Returns:
        img: 可视化后的图像
    """
    img = img.copy()

    for i in range(len(pred["boxes"])):
        # 绘制 2D 边界框
        box = pred["boxes"][i].cpu().numpy()
        h, w = img.shape[:2]
        x1 = int(box[0] * w)
        y1 = int(box[1] * h)
        x2 = int(box[2] * w)
        y2 = int(box[3] * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # 绘制 3D 边界框
        if pred["locations"].numel() > 0:
            dimensions = pred["dimensions"][i].cpu().numpy()
            location = pred["locations"][i].cpu().numpy()
            yaw = pred["yaws"][i].cpu().numpy()

            corners = compute_box_corners(
                torch.tensor(dimensions),
                torch.tensor(location),
                torch.tensor(yaw)
            ).cpu().numpy()

            corners_2d = project_3d_to_2d(
                torch.tensor(corners),
                torch.tensor(calib)
            ).cpu().numpy()

            img = draw_3d_box(img, corners_2d)

    return img