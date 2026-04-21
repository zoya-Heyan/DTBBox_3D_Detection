import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import Backbone
from modules.roi_utils import RoIPool
from modules.pair_graph import PairGraph
from modules.rpfo import RPFO


class DTBoxNet(nn.Module):
    def __init__(self, config):
        super(DTBoxNet, self).__init__()

        # 配置
        self.backbone_name = config.model.backbone
        self.roi_size = config.model.roi_size
        self.fc_dim = config.model.fc_dim
        self.stage = config.get("stage", "baseline")

        # 骨干网络
        self.backbone = Backbone(self.backbone_name)
        backbone_out_channels = self.backbone.out_channels

        # 单目标 RoI 头部
        self.roi_pool = RoIPool(output_size=(self.roi_size, self.roi_size))
        self.fc1 = nn.Linear(backbone_out_channels * self.roi_size * self.roi_size, self.fc_dim)
        self.fc2 = nn.Linear(self.fc_dim, self.fc_dim)

        # 回归头
        self.reg_head = nn.Linear(self.fc_dim, 7)  # 3D 位置 + 3D 尺寸 + 偏航角

        # 双目标 RoI 和相对头部（如果需要）
        if self.stage in ["dtbbox", "relative", "full"]:
            self.pair_graph = PairGraph()

        if self.stage in ["relative", "full"]:
            self.rel_head = nn.Linear(self.fc_dim * 2, 4)  # 相对位置和尺寸

        # R-PFO（测试时使用）
        if self.stage == "full":
            self.rpfo = RPFO(lam=config.rpfo.lam, min_depth=config.rpfo.min_depth)

    def forward(self, images, bboxes):
        # 提取特征
        features = self.backbone(images)

        # 单目标 RoI 处理
        batch_size = images.shape[0]
        device = images.device

        all_preds = []
        for i in range(batch_size):
            batch_bboxes = bboxes[i]
            if batch_bboxes.numel() == 0:
                # 空样本处理
                all_preds.append({
                    "boxes": torch.empty(0, 4, device=device),
                    "dimensions": torch.empty(0, 3, device=device),
                    "locations": torch.empty(0, 3, device=device),
                    "yaws": torch.empty(0, device=device)
                })
                continue

            # RoI Pooling
            roi_features = self.roi_pool(features[i].unsqueeze(0), batch_bboxes.unsqueeze(0))[0]
            roi_features = roi_features.view(roi_features.shape[0], -1)

            # 全连接层
            x = F.relu(self.fc1(roi_features))
            x = F.relu(self.fc2(x))

            # 回归预测
            reg_preds = self.reg_head(x)
            locations = reg_preds[:, :3]
            dimensions = reg_preds[:, 3:6]
            yaws = reg_preds[:, 6]

            # 双目标处理
            if self.stage in ["dtbbox", "relative", "full"]:
                # 构建配对图
                pairs = self.pair_graph(batch_bboxes)
                if pairs.numel() > 0 and self.stage in ["relative", "full"]:
                    # 相对头部预测
                    pair_features = torch.cat([
                        x[pairs[:, 0]],
                        x[pairs[:, 1]]
                    ], dim=1)
                    rel_preds = self.rel_head(pair_features)
                    # 这里可以添加相对预测的处理逻辑

            all_preds.append({
                "boxes": batch_bboxes,
                "dimensions": dimensions,
                "locations": locations,
                "yaws": yaws
            })

        return all_preds

    def inference(self, images, bboxes, calibs):
        # 前向传播
        preds = self.forward(images, bboxes)

        # R-PFO 后处理
        if self.stage == "full":
            for i, pred in enumerate(preds):
                if pred["locations"].numel() > 0:
                    # 应用 R-PFO
                    pred["locations"] = self.rpfo(pred["locations"], calibs[i])

        return preds