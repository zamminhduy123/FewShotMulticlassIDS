import torchvision.models as models
import torch.nn as nn

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        # Load a pre-defined ResNet (e.g., ResNet18)
        self.resnet = models.resnet18(pretrained=False)
        # Modify the first convolutional layer to match the input channels and size
        self.resnet.conv1 = nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # Modify the first pooling layer if needed, or remove it
        self.resnet.maxpool = nn.Identity()  # Remove pooling to retain spatial dimensions

        self.embs = nn.Sequential(
            nn.Linear(1000, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
        )

    def forward(self, x):
        x = x.view(x.shape[0], 16, 3, 3)
        x = self.resnet(x)
        x = self.embs(x)
        return x