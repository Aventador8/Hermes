
import torch.nn as nn
import torch
from torchvision.models.resnet import Bottleneck

from Code.model.Intra_task_attention import Intra_task_attention


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Normalize(nn.Module):
    """ Ln normalization copied from
    https://github.com/salesforce/CoMatch
    """
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class ClsEmbeddingHead(nn.Module):
    def __init__(self, in_channel=2048 ,low_dim=128):
        super(ClsEmbeddingHead, self).__init__()
        self.l2norm = Normalize(2)
        self.projection = nn.Sequential(nn.Linear(in_channel, in_channel//2),
                                        nn.LeakyReLU(inplace=True, negative_slope=0.1),
                                        nn.Linear(in_channel//2,low_dim)
                                        )

    def forward(self, x):
        x = self.projection(x)
        x = self.l2norm(x)
        return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):

        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet18(nn.Module):
    def __init__(self, block, layers, num_classes=1000,
                 zero_init_residual=False, device=None,
                 cls_contrast_button=False, consistency_button=False,
                 consistency_dimension=256
                 ):

        super(ResNet18, self).__init__()
        self.cls_contrast_button = cls_contrast_button
        self.consistency_button = consistency_button
        self.consistency_dimension = consistency_dimension

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Stage1 ~ Stage4
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # GAP
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.cls_head = nn.Linear(512 * block.expansion, num_classes)
        self.cls_embed_head = ClsEmbeddingHead(in_channel=512 * block.expansion, low_dim=128)

        self.intra_attention4 = Intra_task_attention(in_channel=64)
        self.confuesed3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.intra_attention3 = Intra_task_attention(in_channel=64)
        self.confuesed2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.intra_attention2 = Intra_task_attention(in_channel=128)
        self.confuesed1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.intra_attention1 = Intra_task_attention(in_channel=256)
        self.interTaskConsistency_projection_class = nn.Sequential(nn.Linear(512,256,bias=False),
                                                                   nn.BatchNorm1d(256),
                                                                   nn.ReLU(inplace=True),
                                                                   nn.Linear(256, self.consistency_dimension, bias=True),
                                                                   nn.BatchNorm1d(self.consistency_dimension),
                                                                   nn.ReLU(inplace=True),
                                                                   ) #N*64


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, seg_out=None):  # 16 * 3 * 256 * 256

        x = self.conv1(x)        # 16 * 64 * 128 * 128
        x = self.bn1(x)
        x = self.relu(x)
        if seg_out is not None:
            x = self.intra_attention4(x, seg_out['x_up3'])     # seg_out['x']: 16, 64 ,128 128

        x = self.maxpool(x)                              # 16 * 64 * 64 * 64
        x = self.layer1(x)                               # 16 * 64 * 64 * 64
        if seg_out is not None:
            seg_feat3 = self.confuesed3(seg_out['x_up2'])  # seg_out['x1'] : 256, 64, 64
            x = self.intra_attention3(x, seg_feat3)   # seg_feat3:   64, 64, 64

        x = self.layer2(x)                                  # 16 * 128 * 32 * 32
        if seg_out is not None:
            seg_feat2 = self.confuesed2(seg_out['x_up1'])       # seg_out['x2'] : 512, 32, 32
            x = self.intra_attention2(x, seg_feat2)    # 128 * 32 * 32

        x = self.layer3(x)                                  # 16 * 256 * 16 * 16
        if seg_out is not None:
            seg_feat1 = self.confuesed1(seg_out['x3'])                     # seg_out['x3'] : 1024, 16, 16
            x = self.intra_attention1(x, seg_feat1)  # 256 * 16 * 16

        x = self.layer4(x)                                  # 16 * 512 * 8 * 8

        x = self.avgpool(x)                                 # 16 * 512 * 1 * 1
        feat = x.view(x.size(0), -1)
        # out = self.fc(feat)
        out = self.cls_head(feat)

        if self.consistency_button:
            consistency_feat = self.interTaskConsistency_projection_class(feat)
        else:
            consistency_feat = None

        if self.cls_contrast_button:
            # embedding layer for contrastive learning
            contrast_feat = self.cls_embed_head(feat)

        else:
            contrast_feat = None

        return {'output': out, 'contrast_embed': contrast_feat, 'consistency': consistency_feat}

def pretrained_resnet18(pretrained=False, device=None, num_classes=2,
                        cls_contrast_button=False, consistency_button=False, consistency_dimension=256,**kwargs):

    model = ResNet18(BasicBlock, [2, 2, 2, 2], num_classes=num_classes,
                     cls_contrast_button=cls_contrast_button, consistency_button=consistency_button, consistency_dimension=consistency_dimension,**kwargs)
    if pretrained:
        model_state = torch.load('/data/zhangpeng/resnet18-5c106cde.pth')
        model.load_state_dict(model_state,strict=False)
        model = model.to(device)
        print("device: ",device)
        print('resnet: ', next(model.parameters()).device)
    return model



