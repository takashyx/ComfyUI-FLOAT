import torch
import torch.nn as nn
from ..basemodel import BaseModel

class StyleEncoder(BaseModel):
    def __init__(self, opt):
        super(StyleEncoder, self).__init__()
        self.opt = opt
        
        # Define the encoder architecture
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
        
        # Final fully connected layer
        self.fc = nn.Linear(512 * 4 * 4, opt.dim_w)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        self.bn5 = nn.BatchNorm2d(512)
        
        # LeakyReLU activation
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        # Input shape: (B, C, H, W)
        x = self.leaky_relu(self.bn1(self.conv1(x)))
        x = self.leaky_relu(self.bn2(self.conv2(x)))
        x = self.leaky_relu(self.bn3(self.conv3(x)))
        x = self.leaky_relu(self.bn4(self.conv4(x)))
        x = self.leaky_relu(self.bn5(self.conv5(x)))
        
        # Flatten and pass through final layer
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x 