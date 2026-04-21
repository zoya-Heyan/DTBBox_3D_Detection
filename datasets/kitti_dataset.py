import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def read_calib(calib_path):
    with open(calib_path, "r") as f:
        lines = f.readlines()

    P2 = None
    for line in lines:
        if line.startswith("P2:"):
            values = line.strip().split()[1:]
            P2 = np.array([float(v) for v in values], dtype=np.float32).reshape(3, 4)
            break

    if P2 is None:
        raise ValueError(f"P2 not found in {calib_path}")

    return P2


def read_label(label_path, allowed_classes=("Car", "Van", "Truck")):
    objects = []

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue

        cls = parts[0]
        if cls not in allowed_classes:
            continue

        # 解析 2D 边界框
        left = float(parts[4])
        top = float(parts[5])
        right = float(parts[6])
        bottom = float(parts[7])

        # 解析 3D 边界框
        height = float(parts[8])
        width = float(parts[9])
        length = float(parts[10])
        x = float(parts[11])
        y = float(parts[12])
        z = float(parts[13])
        yaw = float(parts[14])

        objects.append({
            "class": cls,
            "bbox": [left, top, right, bottom],
            "dimensions": [height, width, length],
            "location": [x, y, z],
            "yaw": yaw
        })

    return objects


class KittiDataset(Dataset):
    def __init__(self, root, split_file, input_size=(512, 512), allowed_classes=("Car", "Van", "Truck")):
        self.root = root
        self.input_size = input_size
        self.allowed_classes = allowed_classes

        # 读取 split 文件
        with open(split_file, "r") as f:
            self.sample_ids = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]

        # 加载图像
        img_path = os.path.join(self.root, "training", "image_2", f"{sample_id}.png")
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")

        orig_h, orig_w = img.shape[:2]

        # 加载标定文件
        calib_path = os.path.join(self.root, "training", "calib", f"{sample_id}.txt")
        P2 = read_calib(calib_path)

        # 加载标签
        label_path = os.path.join(self.root, "training", "label_2", f"{sample_id}.txt")
        objects = read_label(label_path, self.allowed_classes)

        # 预处理图像
        img = cv2.resize(img, self.input_size)
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        img = img / 255.0
        img = torch.from_numpy(img).float()

        # 处理标签
        bboxes = []
        dimensions = []
        locations = []
        yaws = []
        labels = []

        for obj in objects:
            bbox = obj["bbox"]
            # 归一化边界框
            bbox[0] = bbox[0] / orig_w
            bbox[1] = bbox[1] / orig_h
            bbox[2] = bbox[2] / orig_w
            bbox[3] = bbox[3] / orig_h
            bboxes.append(bbox)

            dimensions.append(obj["dimensions"])
            locations.append(obj["location"])
            yaws.append(obj["yaw"])
            labels.append(0)  # 暂时只处理一种类别

        # 处理空样本
        if not bboxes:
            bboxes = np.array([], dtype=np.float32)
            dimensions = np.array([], dtype=np.float32)
            locations = np.array([], dtype=np.float32)
            yaws = np.array([], dtype=np.float32)
            labels = np.array([], dtype=np.int64)
        else:
            bboxes = np.array(bboxes, dtype=np.float32)
            dimensions = np.array(dimensions, dtype=np.float32)
            locations = np.array(locations, dtype=np.float32)
            yaws = np.array(yaws, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)

        return {
            "image": img,
            "bboxes": torch.from_numpy(bboxes),
            "dimensions": torch.from_numpy(dimensions),
            "locations": torch.from_numpy(locations),
            "yaws": torch.from_numpy(yaws),
            "labels": torch.from_numpy(labels),
            "calib": torch.from_numpy(P2),
            "orig_size": torch.tensor([orig_h, orig_w], dtype=torch.int32),
            "sample_id": sample_id
        }