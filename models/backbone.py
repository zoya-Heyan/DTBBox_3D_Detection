import torch
import torch.nn as nn
import torchvision.models as models


class Backbone(nn.Module):
    def __init__(self, backbone_name="resnet18"):
        super(Backbone, self).__init__()

        if backbone_name == "resnet18":
            self.backbone = models.resnet18(pretrained=True)
            self.out_channels = 512
        elif backbone_name == "resnet34":
            self.backbone = models.resnet34(pretrained=True)
            self.out_channels = 512
        elif backbone_name == "resnet50":
            self.backbone = models.resnet50(pretrained=True)
            self.out_channels = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # 移除最后的分类层
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

    def forward(self, x):
        features = self.backbone(x)
        return features