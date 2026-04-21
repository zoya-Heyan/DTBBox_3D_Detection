import torch
import numpy as np

def project_3d_to_2d(points, calib):
    """
    将 3D 点投影到 2D 图像平面
    Args:
        points: [N, 3] 3D 点
        calib: [3, 4] 相机标定矩阵
    Returns:
        points_2d: [N, 2] 2D 点
    """
    # 添加齐次坐标
    points_homo = torch.cat([points, torch.ones(points.shape[0], 1, device=points.device)], dim=1)
    # 投影
    points_2d = torch.matmul(calib, points_homo.T).T
    # 归一化
    points_2d = points_2d[:, :2] / points_2d[:, 2:3]
    return points_2d

def compute_box_corners(dimensions, location, yaw):
    """
    计算 3D 边界框的 8 个角点
    Args:
        dimensions: [3] 3D 尺寸 (h, w, l)
        location: [3] 3D 位置 (x, y, z)
        yaw: 偏航角
    Returns:
        corners: [8, 3] 3D 角点
    """
    h, w, l = dimensions

    # 边界框的中心点到角点的偏移
    corners = torch.tensor([
        [l/2, w/2, 0],
        [l/2, -w/2, 0],
        [-l/2, -w/2, 0],
        [-l/2, w/2, 0],
        [l/2, w/2, h],
        [l/2, -w/2, h],
        [-l/2, -w/2, h],
        [-l/2, w/2, h]
    ], device=location.device)

    # 旋转
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    rotation = torch.tensor([
        [cos_yaw, -sin_yaw, 0],
        [sin_yaw, cos_yaw, 0],
        [0, 0, 1]
    ], device=location.device)
    corners = torch.matmul(corners, rotation.T)

    # 平移
    corners = corners + location

    return corners