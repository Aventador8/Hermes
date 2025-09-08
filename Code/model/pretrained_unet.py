# -*- coding: utf-8 -*-
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from Code.model.Res2Net_v1b import res2net101_v1b_26w_4s


def generate_saliency(inputs, encoder, optimizer, device):
    encoder.eval()
    inputs2 = copy.copy(inputs)
    inputs2.requires_grad = True


    scores= encoder(inputs2)['output']   # scores are the classification logits,like this[number_class] and don't pass through softmax

    score_max, score_max_index = torch.max(scores, 1)
    score_max.backward(torch.FloatTensor([1.0]*score_max.shape[0]).to(device))
    saliency, _ = torch.max(inputs2.grad.data.abs(),dim=1)

    # saliency = inputs2.grad.data.abs()
    optimizer.zero_grad()
    encoder.train()

    return saliency


class SegEmbeddingHead(nn.Module):
    def __init__(self, dim_in, embed_dim=256, embed='convmlp'):
        super(SegEmbeddingHead, self).__init__()

        if embed == 'linear':
            self.embed = nn.Conv2d(dim_in, embed_dim, kernel_size=1)
        elif embed == 'convmlp':
            self.embed = nn.Sequential(
                nn.Conv2d(dim_in, dim_in, kernel_size=1),
                nn.BatchNorm2d(dim_in),
                nn.ReLU(),
                nn.Conv2d(dim_in, embed_dim, kernel_size=1)
            )

    def forward(self, x):
        return F.normalize(self.embed(x), p=2, dim=1)



class CONV_Block(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        # self.relu = nn.ReLU(inplace = True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        return out


class preUnet(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, device=None,saliency_button=False,
                 seg_pretrained=False, consistency_button=False,seg_contrast_button=False,
                 contrast_feature=None, consistency_dimension=256):
        super().__init__()
        self.device = device
        self.saliency_button = saliency_button
        self.consistency_button = consistency_button
        self.seg_contrast_button = seg_contrast_button
        self.contrast_feature = contrast_feature
        self.consistency_dimension = consistency_dimension

        self.resnet = res2net101_v1b_26w_4s(pretrained=seg_pretrained, device=device)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # self.conv_confused = CONV_Block(1030, 1024, 1024)  # in_channels = 512 + 512 + 6  (1024+6)
        self.conv_up_1 = CONV_Block(1024, 1024, 512)
        # self.conv_up_2_1 = CONV_Block(1028, 512, 512)  # in_channels = 512 + 512 + 4
        self.conv_up_2_2 = CONV_Block(1024, 512, 512)
        self.conv_up_3 = CONV_Block(512, 512, 256)
        self.conv_up_4 = CONV_Block(512, 256, 256)
        self.conv_up_5 = CONV_Block(256, 256, 64)
        self.conv_up_6 = CONV_Block(128, 64, 64)

        self.confused = CONV_Block(516, 512, 512)  # in_channels = 256 + 256 + 4

        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        contrast_channels = {'1':64, '2':256, '3':512, '4':1024}

        # self.seg_contrast_embed_head = SegEmbeddingHead(dim_in=512, embed_dim=256)  # the thrid layer of seg encoder features
        self.seg_contrast_embed_head = SegEmbeddingHead(dim_in=contrast_channels[self.contrast_feature], embed_dim=256)

        # self.interTaskConsistency_seg = nn.Sequential(nn.Conv2d(64,4,3,padding=1), # N * 4 * 512 * 512
        #                                               nn.InstanceNorm2d(4),
        #                                               nn.LeakyReLU(0.2, inplace=True),
        #                                               nn.AdaptiveMaxPool2d((8,8)),
        #                                               nn.Flatten(),
        #                                               nn.Linear(256, 128, bias=False),
        #                                               nn.BatchNorm1d(128),
        #                                               nn.ReLU(inplace=True),
        #                                               nn.Linear(128, 64, bias=True),
        #                                               nn.BatchNorm1d(64),
        #                                               nn.ReLU(inplace=True)
        #                                               ) # N * 256

        self.interTaskConsistency_seg = nn.Sequential(
                                                      nn.AdaptiveAvgPool2d((1, 1)),
                                                      nn.Flatten(),
                                                      nn.Linear(1024, 512, bias=False),
                                                      nn.BatchNorm1d(512),
                                                      nn.ReLU(inplace=True),
                                                      nn.Linear(512, self.consistency_dimension, bias=True),
                                                      nn.BatchNorm1d(self.consistency_dimension),
                                                      nn.ReLU(inplace=True)
                                                      )


    def forward(self, x, cls_model=None, cls_optimizer=None): # bs, 3, 256, 256
        if self.saliency_button:
            saliency = generate_saliency(x, cls_model, cls_optimizer, self.device)
            bridge = torch.cat([x, torch.unsqueeze(saliency,1)], dim=1)
            bridge = nn.functional.interpolate(bridge, scale_factor=0.25, mode='bilinear',
                                               align_corners=True)
        else:
            bridge = None
        x = self.resnet.conv1(x)   # bs, 64 , 112, 112
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x_k = self.resnet.maxpool(x)  # bs, 64,  56, 56

        # ----------- low-level features -------------

        x1 = self.resnet.layer1(x_k)  # bs, 256,  56, 56
        x2 = self.resnet.layer2(x1)  # bs, 512,  28,  28
        x3 = self.resnet.layer3(x2)  # bs, 1024,  14, 14

        x_up_1 = self.conv_up_1(self.up(x3))  # 512,  28, 28
        x_up_1 = self.conv_up_2_2(torch.cat([x2, x_up_1], 1))  # 512, 28, 28

        x_up_2 = self.conv_up_3(self.up(x_up_1))  # 256, 56, 56
        if bridge is not None:
            x_up_2 = self.confused(torch.cat([x1, x_up_2,bridge], 1))
        else:
            x_up_2 = self.conv_up_4(torch.cat([x1, x_up_2], 1))  # 256, 56, 56

        x_up_3 = self.conv_up_5(self.up(x_up_2))  # 64, 112, 112
        x_up_3 = self.conv_up_6(torch.cat([x, x_up_3], 1))  # 64, 112, 112

        x_up_4 = self.up(x_up_3)  # 64, 224, 224
        output = self.final(x_up_4)   # 3, 224, 224

        if self.consistency_button:
            consistency_feat = self.interTaskConsistency_seg(x3)
        else:
            consistency_feat = None
        feature_maps = {'1':x_k, '2':x1, '3':x2, '4':x3}
        if self.seg_contrast_button:
            # seg_contrast_embedding = self.seg_contrast_embed_head(x2)
            seg_contrast_embedding = self.seg_contrast_embed_head(feature_maps[self.contrast_feature])
        else:
            seg_contrast_embedding = None
            # return {'output':output, 'x':x, 'x1':x1, 'x2':x2, 'x3':x3}

        return {'output': output, 'consistency': consistency_feat, 'x_up1': x_up_1,
                'x_up2': x_up_2, 'x_up3': x_up_3, 'x3': x3, 'contrast_embed': seg_contrast_embedding}
