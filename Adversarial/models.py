import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models

class ResNet18(nn.Module):
    name = 'ResNet18'
    def __init__(self, input_ch, output_ch):
        super(ResNet18, self).__init__()
        
        self.resnet = models.resnet18(pretrained=False)
            
        if input_ch!=3:
            self.resnet.conv1 = torch.nn.Conv2d(input_ch, 64, kernel_size=7)
            
        old_n_ch = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(old_n_ch, output_ch)
        
        
        for param in self.resnet.parameters():
            param.requires_grad = True
        
    def forward(self, x):
        out = self.resnet(x)
        return out
    
    
    
class ResNet50(nn.Module):
    name = 'ResNet50'
    def __init__(self, input_ch, output_ch):
        super(ResNet50, self).__init__()
        
        self.resnet = models.resnet50(pretrained=False)
            
        if input_ch!=3:
            self.resnet.conv1 = torch.nn.Conv2d(input_ch, 64, kernel_size=7)
            
        old_n_ch = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(old_n_ch, output_ch)
        
        
        for param in self.resnet.parameters():
            param.requires_grad = True
        
    def forward(self, x):
        out = self.resnet(x)
        return out
    

    

class VGG11(nn.Module):
    name = 'VGG11'
    def __init__(self, input_ch, output_ch):
        super(VGG11, self).__init__()
        
        self.vgg11 = models.vgg11(pretrained=False)
        
        old_n_ch = self.vgg11.classifier[6].in_features
        self.vgg11.classifier[6] = nn.Linear(old_n_ch, output_ch)
        
        for param in self.vgg11.parameters():
            param.requires_grad = True
        
        
        if input_ch!=3:
            self.vgg11.features[0] = torch.nn.Conv2d(input_ch, 64, kernel_size=7)
        
        
    def forward(self, x):
        out = self.vgg11(x)
        return out
    

class ResBlock(nn.Module):
    def __init__(self, input_ch, expand=True):
        super(ResBlock, self).__init__()

        output_ch = input_ch*2 if expand else input_ch

        self.conv_block = nn.Sequential(
            nn.Conv2d(input_ch, output_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(output_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_ch, output_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(output_ch)
        )

        self.downsample = nn.Sequential(
            nn.Conv2d(input_ch, output_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_ch)
        ) if expand else None

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv_block(x)

        if self.downsample:
            identity = self.downsample(x)
        else:
            identity = x

        out = out+identity
        out = self.relu(out)
        return out


class LightResNet(nn.Module):
    """9 layer, suitable for 44x44 image"""
    
    name = 'LightResNet'
    
    def __init__(self, input_ch, output_ch):
        super(LightResNet, self).__init__()
        
        # 第一层conv换成小一点的kernel
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_ch, 64, kernel_size=5, bias=False),   # 40x40x64ch
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # 第一层pooling不用了
        
        self.conv2_block = ResBlock(64, expand=False)   # 20x20x64ch
        self.conv3_block = ResBlock(64)                 # 10x10x128ch
        self.conv4_block = ResBlock(128)                # 5x5x256ch
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))     # 256ch
        self.fc1 = nn.Linear(256, 7)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_block(x)
        x = self.conv3_block(x)
        x = self.conv4_block(x)
        x = self.avgpool(x)
        
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        return x
    
    
    
from torchvision.models import inception

class Inception3(nn.Module):
    name = 'Inception3'
    
    def __init__(self, input_ch, output_ch):
        super(Inception3, self).__init__()
        
        self.inception = inception.Inception3(num_classes=output_ch, aux_logits=False)
        self.inception.Conv2d_1a_3x3 = inception.BasicConv2d(input_ch, 32, kernel_size=3, stride=2)
        
        for param in self.inception.parameters():
            param.requires_grad = True
        
        
    def forward(self, x):
        out = self.inception(x)
        return out
    
    
    
class DenseNet121(nn.Module):
    name = 'DenseNet121'
    def __init__(self, input_ch, output_ch):
        super(DenseNet121, self).__init__()
        
        self.densenet = models.densenet121(pretrained=False, progress=False)
            
        if input_ch!=3:
            self.densenet.features[0] = nn.Conv2d(
                input_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
            
        old_n_ch = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(old_n_ch, output_ch)
        
        for param in self.densenet.parameters():
            param.requires_grad = True
        
    def forward(self, x):
        out = self.densenet(x)
        return out