import torch
import torch.nn as nn
from torchvision import models


def build_image_model(
    backbone: str = "swin_t", num_classes: int = 2, pretrained: bool = True
) -> nn.Module:
    if backbone == "swin_s":
        weights = models.Swin_S_Weights.DEFAULT if pretrained else None
        model = models.swin_s(weights=weights)
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
    elif backbone == "swin_b":
        weights = models.Swin_B_Weights.DEFAULT if pretrained else None
        model = models.swin_b(weights=weights)
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
    elif backbone == "swin_t":
        weights = models.Swin_T_Weights.DEFAULT if pretrained else None
        model = models.swin_t(weights=weights)
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
    elif backbone == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
    elif backbone == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(
            f"Unknown backbone: {backbone}. "
            "Use swin_t, swin_s, swin_b, resnet18, or mobilenet_v3_small."
        )
    return model
