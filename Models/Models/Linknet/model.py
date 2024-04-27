import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights

class LinkNet(nn.Module):
    def __init__(self, num_classes=1):
        super(LinkNet, self).__init__()
        base = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)  # Using ResNet34 as the backbone
        
        self.in_block = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool
        )

        # Encoder blocks (ResNet layers)
        self.encoder1 = base.layer1
        self.encoder2 = base.layer2
        self.encoder3 = base.layer3
        self.encoder4 = base.layer4

        # Decoder blocks
        self.decoder4 = DecoderBlock(512, 256)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder2 = DecoderBlock(128, 64)
        self.decoder1 = DecoderBlock(64, 64)

        # Final convolution
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Initial block
        x = self.in_block(x)

        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder with skip connections
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        # Final output
        out = self.final_conv(d1)
        return torch.sigmoid(out)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.deconv = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.deconv(x)
        x = self.relu2(x)
        x = self.conv2(x)
        x = self.relu3(x)
        return x

def jaccard_loss(y_true, y_pred):
    intersection = torch.sum(y_true * y_pred)
    union = torch.sum(y_true) + torch.sum(y_pred) - intersection
    return 1 - (intersection + 1) / (union + 1)  # Add 1 to avoid division by zero

# Load pre-trained weights
def load_pretrained_weights(model, weights_path):
    model.load_state_dict(torch.load(weights_path))
