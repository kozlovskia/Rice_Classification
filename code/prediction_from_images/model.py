import torch
import torchvision.models as models


class RiceClassifier(torch.nn.Module):
    def __init__(self, backbone_name):
        super().__init__()
        self.backbone_name = backbone_name
        if self.backbone_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=True)
            self.backbone.fc = torch.nn.Linear(in_features=512, out_features=5)
        elif self.backbone_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=True)
            self.backbone.classifier[1] = torch.nn.Linear(in_features=1280, out_features=5)
        else:
            raise ValueError(f'Wrong backbone_name: {backbone_name}!')

    def forward(self, x):
        return self.backbone(x)

    def __str__(self):
        return f"{self.backbone_name=}\n{str(self.backbone)}"
