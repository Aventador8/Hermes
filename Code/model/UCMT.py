import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from Code.model.Res2Net_v1b import res2net50_v1b_26w_4s
from segmentation_models_pytorch.base import ClassificationHead
from Code.model.Intra_task_attention import Intra_task_attention
from Code.model.ResNet18 import pretrained_resnet18
from Code.model.pretrained_unet_copy_discard import preUnet


def generate_saliency(inputs, encoder, optimizer, device):
    encoder.eval()
    inputs2 = copy.copy(inputs)
    inputs2.requires_grad = True


    scores= encoder(inputs2)['cls_out']   # scores are the classification logits,like this[number_class] and don't pass through softmax

    score_max, score_max_index = torch.max(scores, 1)
    score_max.backward(torch.FloatTensor([1.0]*score_max.shape[0]).to(device))
    saliency, _ = torch.max(inputs2.grad.data.abs(),dim=1)

    # saliency = inputs2.grad.data.abs()
    optimizer.zero_grad()
    encoder.train()

    return saliency

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

# cls branch
class preResNet50(nn.Module):
    def __init__(self, num_classes=2, device=None,
                 cls_contrast_button=False, cls_pretrained=False,
                 consistency_button=False):
        super().__init__()
        self.cls_contrast_button = cls_contrast_button
        self.consistency_button = consistency_button
        self.encoder = res2net50_v1b_26w_4s(pretrained=cls_pretrained, device=device)
        # self.encoder = resnet18(pretrained=True, device=device)
        self.cls_head = ClassificationHead(in_channels=2048, classes=num_classes)     # benign and malignant
        self.cls_embed_head = ClsEmbeddingHead(in_channel=2048 ,low_dim=128)    # the last layer of cls encoder features

        self.intra_attention1 = Intra_task_attention(in_channel=1024)     # resolution is  24 * 24
        self.intra_attention2 = Intra_task_attention(in_channel=512)      # resolution is  48 * 48
        self.intra_attention3 = Intra_task_attention(in_channel=256)       # resolution is  96 * 96
        self.intra_attention4 = Intra_task_attention(in_channel=64)         # resolution is  128 * 128

        self.interTaskConsistency_projection_class = nn.Sequential(nn.Linear(2048,512,bias=False),
                                                                   nn.BatchNorm1d(512),
                                                                   nn.ReLU(inplace=True),
                                                                   nn.Linear(512, 64, bias=True),
                                                                   nn.BatchNorm1d(64),
                                                                   nn.ReLU(inplace=True),
                                                                   ) #N*64

    def forward(self, x, seg_out=None):  # 16, 3, 384, 384
        #  当前考虑使用unet encoder 的特征进行融合 (效果不行)
        cls_x = self.encoder.conv1(x)  # 16 * 64 * 192 * 192
        cls_x = self.encoder.bn1(cls_x)
        cls_x = self.encoder.relu(cls_x)
        if seg_out is not None:
            cls_x = self.intra_attention4(cls_x, seg_out['x_up3'])
        cls_x_k = self.encoder.maxpool(cls_x)  # 16 * 64 * 96 * 96

        # 192 * 192, 96 *96 , 48 * 48,  24 * 24   进行特征融合(使用Unet decoder的特征) x_up3, x_up2, x_up1, x3

        cls_x1 = self.encoder.layer1(cls_x_k)  # 16 * 256 * 96 * 96
        if seg_out is not None:
            cls_x1 = self.intra_attention3(cls_x1, seg_out['x_up2'])

        cls_x2 = self.encoder.layer2(cls_x1)  # 16 * 512 * 48 * 48
        if seg_out is not None:
            cls_x2 = self.intra_attention2(cls_x2, seg_out['x_up1'])

        cls_x3 = self.encoder.layer3(cls_x2)  # 16 * 1024 * 24 * 24
        if seg_out is not None:
            cls_x3 = self.intra_attention1(cls_x3, seg_out['x3'])

        cls_x4 = self.encoder.layer4(cls_x3)   # 16 * 2048 * 12 * 12

        out = self.cls_head(cls_x4)
        if self.cls_contrast_button:
            # embedding layer for contrastive learning
            embed = F.adaptive_avg_pool2d(cls_x4, 1)
            feat = embed.view(embed.size(0), -1)
            contrast_feat = self.cls_embed_head(feat)
        else:
            contrast_feat = None

        if self.consistency_button:
            embed = F.adaptive_avg_pool2d(cls_x4, 1)
            feat = embed.view(embed.size(0), -1)
            consistency_feat = self.interTaskConsistency_projection_class(feat)
        else:
            consistency_feat = None

        return {'cls_out': out, 'cls_contrast_embed': contrast_feat, 'consistency': consistency_feat}


class Saliency(nn.Module):
    def __init__(self,device=None):
        super().__init__()
        self.device = device

    def forward(self, x, cls_model,optimizer):
        saliency = generate_saliency(x, cls_model, optimizer, self.device)
        bridge = torch.cat([x, torch.unsqueeze(saliency,1)], dim=1)
        bridge = nn.functional.interpolate(bridge, scale_factor=0.125, mode='bilinear',
                                           align_corners=True)
        return bridge


class UNetAndResNet(nn.Module):
    def __init__(self, cls_num_class=2, seg_num_class=2, device=None, ita_button=True,
                 saliency_button=True, lesion_button=True, cls_contrast_button=False,
                 seg_contrast_button=False, cls_pretrained=False, seg_pretrained=False,
                 consistency_button=False):
        super().__init__()
        self.seg_contrast_button = seg_contrast_button
        # self.cls_branch_resnet = preResNet50(num_classer=cls_num_class, device=device,
        #                                    cls_contrast_button=cls_contrast_button, cls_pretrained=cls_pretrained,
        #                                    consistency_button=consistency_button)
        self.cls_branch_resnet = pretrained_resnet18(pretrained=cls_pretrained, num_classes=cls_num_class,
                                                     device=device, cls_contrast_button=cls_contrast_button,
                                                     consistency_button=consistency_button)
        self.seg_branch_network = preUnet(num_classes=seg_num_class, device=device,
                                          seg_pretrained=seg_pretrained, consistency_button=consistency_button)
        self.seg_embed_head = SegEmbeddingHead(dim_in=512, embed_dim=256)    # the thrid layer of seg encoder features
        self.ita_button = ita_button
        self.saliency_button = saliency_button
        self.lesion_button = lesion_button
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.classifier = nn.Linear(1024, cls_num_class)

    def forward(self, x, optimizer=None, device=None):  # 16, 3, 384, 384
        if self.ita_button:
            if self.saliency_button:
                # saliency
                saliency = generate_saliency(x, self.cls_branch_resnet, optimizer, device)
                bridge = torch.cat([x, torch.unsqueeze(saliency,1)], dim=1)      # N * 4 * 384 * 384
                bridge = nn.functional.interpolate(bridge, scale_factor=0.0625, mode='bilinear',
                                                   align_corners=True)  # 下采样8倍   #  N * 4 * 48 * 48   (当前改为下采样16倍)
                # seg branch
                seg_outputs = self.seg_branch_network(x, bridge)
            else:
                seg_outputs = self.seg_branch_network(x)

            # 对比学习   对unet 编码器分辨率为 48 * 48 进行对比
            if self.seg_contrast_button:
                seg_embedding = self.seg_embed_head(seg_outputs['contrast_embed'])
            else:
                seg_embedding = None

            if self.lesion_button:
                cls_outputs = self.cls_branch_resnet(x, seg_outputs)
            else:
                cls_outputs = self.cls_branch_resnet(x)

            if self.saliency_button:
                return {'seg':seg_outputs, 'seg_embed':seg_embedding, 'cls':cls_outputs,
                        'saliency':saliency}
            else:
                return {'seg': seg_outputs, 'seg_embed': seg_embedding, 'cls': cls_outputs}
        else:

            # seg branch
            seg_outputs = self.seg_branch_network(x)
            # 对比学习   对unet 编码器分辨率为 48 * 48 进行对比
            if self.seg_contrast_button:
                seg_embedding = self.seg_embed_head(seg_outputs['contrast_embed'])
            else:
                seg_embedding = None

            # cls branch
            cls_outputs = self.cls_branch_resnet(x)
            return {'seg': seg_outputs, 'cls': cls_outputs, 'seg_embed':seg_embedding}

    def inference_seg(self, x):
        seg = self.seg_branch_network(x)['output']
        # preds = torch.argmax(seg, dim=1).to(torch.float)
        return seg

    def inference_cls(self, x):
        cls = self.cls_branch_resnet(x)['cls_out']
        return cls


if __name__ == '__main__':
    x1 = torch.rand([4, 3, 384, 384]).cuda(2)
    model = UNetAndResNet(cls_num_class=2, seg_num_class=3).cuda(2)
    device = 'cuda:2'
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
    output = model(x1, optimizer, device)
    print(output['cls']['cls_embed'].shape)



