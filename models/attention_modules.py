import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1, inplace=False)
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=False)
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class Memory_Attention_Aggregation(nn.Module):
    def __init__(self, agg_dim, d_model=512, S=64):
        super(Memory_Attention_Aggregation, self).__init__()
        self.agg_dim = agg_dim
        self.weight = nn.Parameter(torch.Tensor(d_model, d_model))
        self.softmax = nn.Softmax(dim=-1)
        torch.nn.init.kaiming_normal_(self.weight, a=np.sqrt(5))

        self.mk = nn.Linear(d_model, S, bias=False)
        self.mv = nn.Linear(S, d_model, bias=False)
        self.attn_softmax = nn.Softmax(dim=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, hiddens):

        attn = self.mk(hiddens)
        attn = self.attn_softmax(attn)
        attn = attn / torch.sum(attn, dim=2, keepdim=True)
        hiddens = self.mv(attn)
        m = torch.tanh(hiddens)
        alpha = torch.softmax(torch.matmul(m, self.weight), 0)
        roh = torch.mul(hiddens, alpha)
        new_h = torch.sum(roh, 0)
        return new_h

class Auxiliary_Self_Attention_Aggregation(nn.Module):
    def __init__(self, agg_dim):
        super(Auxiliary_Self_Attention_Aggregation, self).__init__()
        self.agg_dim = agg_dim
        self.weight = nn.Parameter(torch.Tensor(agg_dim, 1))
        self.softmax = nn.Softmax(dim=-1)
        torch.nn.init.kaiming_normal_(self.weight, a=np.sqrt(5))
        self.dsc = DepthwiseSeparableConv(in_channels=1, out_channels=1)

    def forward(self, hiddens):
        hiddens = hiddens.unsqueeze(0)
        hiddens = hiddens.permute(1, 0, 2, 3)
        hiddens = self.dsc(hiddens)
        maxpool = torch.max(hiddens, dim=1)[0]
        avgpool = torch.mean(hiddens, dim=1)
        agg_spatial = torch.cat((avgpool, maxpool), dim=1)
        energy = torch.bmm(agg_spatial.permute([0, 2, 1]), agg_spatial)
        attention = self.softmax(energy)
        weighted_feat = torch.bmm(attention, agg_spatial.permute([0, 2, 1]))
        weight = self.weight.unsqueeze(0).repeat([hiddens.size(0), 1, 1])
        agg_feature = torch.bmm(weighted_feat.permute([0, 2, 1]), weight)
        return agg_feature.squeeze(dim=-1)

class EMSA(nn.Module):
    def __init__(self, channels, factor=5):
        super(EMSA, self).__init__()
        self.groups = factor
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        concat_x = torch.cat([x_h, x_w], dim=2)
        hw = self.conv1x1(concat_x)
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x_h_sigmoid = x_h.sigmoid()
        x_w_sigmoid = x_w.permute(0, 1, 3, 2).sigmoid()
        x1 = self.gn(group_x * x_h_sigmoid * x_w_sigmoid)
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)
