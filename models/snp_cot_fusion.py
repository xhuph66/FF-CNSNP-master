#!/user/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
from .CoTnet import CoTAttention

class SNPLayer(nn.Module):
    def __init__(self):
        super(SNPLayer, self).__init__()
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.bn_0 = nn.BatchNorm2d(5)
        self.bn_1 = nn.BatchNorm2d(32)
        self.SFT_scale_conv0 = nn.Conv2d(5, 32, 1)
        self.SFT_scale_conv1 = nn.Conv2d(32, 5, 1)
        self.SFT_shift_conv0 = nn.Conv2d(5, 32, 1)
        self.SFT_shift_conv1 = nn.Conv2d(32, 5, 1)

    def forward(self, x):
        act_scale = self.act(self.bn_0(x[1]))
        act_shift = self.act(self.bn_0(x[1]))
        scale = self.SFT_scale_conv1(self.act(self.bn_1(self.SFT_scale_conv0(act_scale))))
        shift = self.SFT_shift_conv1(self.act(self.bn_1(self.SFT_shift_conv0(act_shift))))
        return x[0] * (scale + 1) + shift


class SNPFusion(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(SNPFusion, self).__init__()

        self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=3,
                               stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, out_ch, kernel_size=3,
                               stride=1, padding=1)
        self.relu = nn.ReLU()

        self.norm_layer1 = nn.GroupNorm(4, 64)
        self.norm_layer2 = nn.GroupNorm(4, 64)
        self.attn_cot = CoTAttention(5)
        self.sft_snp = SNPLayer()

    def forward(self, x):
        fusecat = torch.cat(x, dim=1)

        attn_cot = self.attn_cot(fusecat)

        sft_in = [fusecat, attn_cot]
        sft_out = self.sft_snp(sft_in)
        sft_out = self.conv1(self.norm_layer1(self.relu(sft_out)))
        sft_out = self.conv2(self.norm_layer2(self.relu(sft_out)))

        attn_fuse_snp = ((sft_out).sum(1)).unsqueeze(1)
        return attn_fuse_snp
