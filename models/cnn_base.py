import torch
import torch.nn as nn
import math
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .ode_expand import ODE_Expander
from einops import rearrange
from .cnndec import CNNdec, CNNenc, CNNenc_tiny


class cnn_net(nn.Module):

    def __init__(self, opts):
        super().__init__()
        self.output_size = opts.output_size
        self.input_size = opts.input_size
        self.patch_size = 32

        self.cnn_encoder = CNNenc()
        self.cnn_decoder = CNNdec()
        self.feature_expand = ODE_Expander(1024)
        self.inner_index, self.outer_index=self.get_index()
        self.apply(self._init_weights)


    def get_last_layer(self):
        return self.cnn_decoder.decoder_stage0.weight

    def get_index(self):
        input_query_width=self.input_size//self.patch_size
        output_query_width=self.output_size//self.patch_size
        mask=torch.ones(size=[output_query_width,output_query_width]).long()
        pad_width=(output_query_width-input_query_width)//2
        mask[pad_width:-pad_width,pad_width:-pad_width] = 0
        mask=mask.view(-1)
        return mask==0,mask==1

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}


    def forward(self, samples):
        if type(samples) is not dict:
            samples={'input':samples, 'gt_inner':F.pad(samples,(32,32,32,32))}
        x = samples['input']
        gt_inner = samples['gt_inner']

        en_feature, shortcut = self.cnn_encoder(x)
        src = rearrange(en_feature, 'b c h w-> b (h w) c')
        src = self.feature_expand(src)
        fake = self.cnn_decoder(rearrange(src, 'b (h w) c-> b c h w', h=6, w=6), shortcut)

        p=(192-128)//2
        fake[:, :, p:-p, p:-p] = gt_inner[:, :, p:-p, p:-p]
        return fake

