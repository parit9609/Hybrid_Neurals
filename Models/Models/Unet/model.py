import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        # Encoder
        self.conv1 = self.conv_block(3, 64)
        self.conv2 = self.conv_block(64, 128)
        self.conv3 = self.conv_block(128, 256)
        self.conv4 = self.conv_block(256, 512)
        self.conv5 = self.conv_block(512, 1024)
        # Decoder
        self.upconv6 = self.conv_transpose_block(1024, 512)
        self.conv6 = self.conv_block(1024, 512)
        self.upconv7 = self.conv_transpose_block(512, 256)
        self.conv7 = self.conv_block(512, 256)
        self.upconv8 = self.conv_transpose_block(256, 128)
        self.conv8 = self.conv_block(256, 128)
        self.upconv9 = self.conv_transpose_block(128, 64)
        self.conv9 = self.conv_block(128, 64)
        self.conv10 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def conv_transpose_block(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        # Encoder
        conv1 = self.conv1(x)
        pool1 = F.max_pool2d(conv1, kernel_size=2)
        conv2 = self.conv2(pool1)
        pool2 = F.max_pool2d(conv2, kernel_size=2)
        conv3 = self.conv3(pool2)
        pool3 = F.max_pool2d(conv3, kernel_size=2)
        conv4 = self.conv4(pool3)
        pool4 = F.max_pool2d(conv4, kernel_size=2)
        conv5 = self.conv5(pool4)
        # Decoder
        upconv6 = self.upconv6(conv5)
        concat6 = torch.cat([upconv6, conv4], dim=1)
        conv6 = self.conv6(concat6)
        upconv7 = self.upconv7(conv6)
        concat7 = torch.cat([upconv7, conv3], dim=1)
        conv7 = self.conv7(concat7)
        upconv8 = self.upconv8(conv7)
        concat8 = torch.cat([upconv8, conv2], dim=1)
        conv8 = self.conv8(concat8)
        upconv9 = self.upconv9(conv8)
        concat9 = torch.cat([upconv9, conv1], dim=1)
        conv9 = self.conv9(concat9)
        conv10 = self.conv10(conv9)
        output = torch.sigmoid(conv10)
        return output

# Jaccard Loss function
def jaccard_loss(y_true, y_pred):
    intersection = torch.sum(y_true * y_pred)
    union = torch.sum(y_true) + torch.sum(y_pred) - intersection
    return 1 - (intersection + 1) / (union + 1)  # Add 1 to avoid division by zero

# Load pre-trained weights
def load_pretrained_weights(model, weights_path):
    model.load_state_dict(torch.load(weights_path))

