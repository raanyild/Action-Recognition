import numpy as np
import torch
import torch.nn as nn


class acnet(nn.Module):
    def __init__(self):
        super(acnet, self).__init__()
        self.color_feature = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=2),
            nn.Conv2d(64, 128, kernel_size=5, padding=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
    
        self.flow_feature = nn.Sequential(
            nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.conv1 = nn.Conv2d(256, 512, kernel_size=3)
        self.r1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(512, 512, kernel_size=3)
        self.r2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=3)
        self.r3 = nn.ReLU(inplace=True)
        
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512 * 4 * 4, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 3)
        )

        
        
    def forward(self, x1, x2):
        x1 = self.color_feature(x1)
        x2 = self.flow_feature(x2)
        x = torch.cat((x1, x2), dim=1)
        x = self.conv1(x)
        x = self.r1(x)
        x = self.conv2(x)
        x = self.r2(x)
        x = self.conv3(x)
        x = self.r3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x