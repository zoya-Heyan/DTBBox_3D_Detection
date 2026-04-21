import torch
import torch.nn as nn


class PairGraph(nn.Module):
    def __init__(self):
        super(PairGraph, self).__init__()

    def forward(self, bboxes):
        """
        构建目标对
        Args:
            bboxes: [N, 4] 边界框，格式为 [x1, y1, x2, y2]
        Returns:
            pairs: [M, 2] 目标对索引
        """
        N = bboxes.shape[0]
        if N < 2:
            return torch.empty(0, 2, dtype=torch.long, device=bboxes.device)

        # 生成所有可能的目标对
        pairs = []
        for i in range(N):
            for j in range(i + 1, N):
                pairs.append([i, j])

        return torch.tensor(pairs, dtype=torch.long, device=bboxes.device)