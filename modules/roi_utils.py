import torch
import torch.nn as nn
import torch.nn.functional as F


class RoIPool(nn.Module):
    def __init__(self, output_size):
        super(RoIPool, self).__init__()
        self.output_size = output_size

    def forward(self, features, rois):
        """
        RoI Pooling
        Args:
            features: [B, C, H, W] 特征图
            rois: [B, N, 4] 边界框，格式为 [x1, y1, x2, y2]（归一化坐标）
        Returns:
            pooled_features: [B, N, C, output_size[0], output_size[1]]
        """
        batch_size, channels, height, width = features.shape
        num_rois = rois.shape[1]

        pooled_features = []
        for i in range(batch_size):
            batch_rois = rois[i]
            batch_features = features[i]

            roi_features = []
            for roi in batch_rois:
                # 将归一化坐标转换为绝对坐标
                x1 = int(roi[0] * width)
                y1 = int(roi[1] * height)
                x2 = int(roi[2] * width)
                y2 = int(roi[3] * height)

                # 确保坐标有效
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(width - 1, x2)
                y2 = min(height - 1, y2)

                # 裁剪特征图
                roi_feature = batch_features[:, y1:y2+1, x1:x2+1]

                # 池化
                if roi_feature.numel() == 0:
                    # 处理空区域
                    pooled = torch.zeros(channels, self.output_size[0], self.output_size[1],
                                        device=features.device)
                else:
                    pooled = F.adaptive_max_pool2d(roi_feature.unsqueeze(0), self.output_size)[0]

                roi_features.append(pooled)

            if roi_features:
                pooled_features.append(torch.stack(roi_features))
            else:
                pooled_features.append(torch.empty(0, channels, self.output_size[0], self.output_size[1],
                                               device=features.device))

        return pooled_features