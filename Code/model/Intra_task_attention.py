import torch
from torch import nn



# channel attention means what to focus on
class Channel_attention(nn.Module):
    def __init__(self, in_channel, ratio=4):
        super(Channel_attention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        # to fuse cls and seg  channel informations
        self.conv = nn.Conv2d(in_channels=in_channel*2, out_channels=in_channel, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(in_features=in_channel, out_features=in_channel//ratio)
        self.fc2 = nn.Linear(in_features=in_channel//4, out_features=in_channel)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, cls_feature, seg_feature):
        fused_feature = self.conv(torch.cat([cls_feature, seg_feature], 1))

        b, c, h, w = fused_feature.shape
        avg_pool = self.avg_pool(fused_feature)
        avg_pool = avg_pool.view([b, c])
        avg_out = self.fc2(self.relu(self.fc1(avg_pool)))
        avg_out = self.sigmoid(avg_out)
        avg_out = avg_out.view([b, c, 1, 1])
        avg_out = avg_out * cls_feature

        return avg_out


# spatial channel means where to focus on
class Spatial_attention(nn.Module):
    def __init__(self, kernel_size=3):
        super(Spatial_attention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x_maxpool, _ = torch.max(x, dim=1, keepdim=True)
        x_avgpool = torch.mean(x, dim=1, keepdim=True)

        feat = torch.cat([x_maxpool, x_avgpool], dim=1)
        feat = self.conv(feat)
        feat = self.sigmoid(feat)
        output = feat * x

        return output


class Intra_task_attention(nn.Module):
    def __init__(self, in_channel, ratio=4, kernel_size=3):
        super(Intra_task_attention, self).__init__()

        self.channel_attention = Channel_attention(in_channel=in_channel, ratio=ratio)
        self.spatial_attention = Spatial_attention(kernel_size=kernel_size)


    def forward(self, cls_feature, seg_feature):
        x = self.channel_attention(cls_feature, seg_feature)
        output = self.spatial_attention(x)

        return output






if __name__ == '__main__':
    x1 = torch.rand([4,32,16,16])
    x2 = torch.rand([4,32,16,16])
    model = Intra_task_attention(in_channel=x1.shape[1])
    out = model(x1, x2)

    print(out.shape)


