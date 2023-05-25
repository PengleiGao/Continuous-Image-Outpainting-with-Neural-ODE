import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
#from utils.utils import *
from einops import rearrange


def make_layers(in_channel, out_channel, kernel_size, stride, padding, dilation=1, bias=True, norm=True,
                activation=True, is_relu=False):
    layer = []
    if norm:
        layer.append(nn.InstanceNorm2d(in_channel, affine=True))
    if activation:
        if is_relu:
            layer.append(nn.ReLU())
        else:
            layer.append(nn.LeakyReLU(negative_slope=0.2))
    layer.append(
        nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                  bias=bias))
    return nn.Sequential(*layer)


def make_layers_transpose(in_channel, out_channel, kernel_size, stride, padding, dilation=1, bias=True, norm=True,
                          activation=True, is_relu=False):
    layer = []
    if norm:
        layer.append(nn.InstanceNorm2d(in_channel, affine=True))
    if activation:
        if is_relu:
            layer.append(nn.ReLU())
        else:
            layer.append(nn.LeakyReLU(negative_slope=0.2))
    layer.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding,
                                    dilation=dilation, bias=bias))
    return nn.Sequential(*layer)

def rms_norm(tensor):
    return tensor.pow(2).mean().sqrt()

def make_norm(state):
    state_size = state.numel()
    def norm(aug_state):
        y = aug_state[1:1 + state_size]
        adj_y = aug_state[1 + state_size:1 + 2 * state_size]
        return max(rms_norm(y), rms_norm(adj_y))
    return norm

class identity_block(nn.Module):
    def __init__(self, channels, norm=True, is_relu=False):
        super(identity_block, self).__init__()

        self.conv1 = make_layers(channels[0], channels[1], kernel_size=1, stride=1, padding=0, bias=False, norm=norm,
                                 activation=True, is_relu=is_relu)
        self.conv2 = make_layers(channels[1], channels[2], kernel_size=3, stride=1, padding=1, bias=False, norm=norm,
                                 activation=True, is_relu=is_relu)
        self.conv3 = make_layers(channels[2], channels[3], kernel_size=1, stride=1, padding=0, bias=False, norm=norm,
                                 activation=False)
        self.output = nn.ReLU() if is_relu else nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x + shortcut
        x = self.output(x)
        return x

class convolutional_block(nn.Module):
    def __init__(self, channels, norm=True, is_relu=False):
        super(convolutional_block, self).__init__()

        self.conv1 = make_layers(channels[0], channels[1], kernel_size=1, stride=1, padding=0, bias=False, norm=norm,
                                 activation=True, is_relu=is_relu)
        self.conv2 = make_layers(channels[1], channels[2], kernel_size=3, stride=2, padding=1, bias=False, norm=norm,
                                 activation=True, is_relu=is_relu)
        self.conv3 = make_layers(channels[2], channels[3], kernel_size=1, stride=1, padding=0, bias=False, norm=norm,
                                 activation=False)
        self.shortcut_path = make_layers(channels[0], channels[3], kernel_size=1, stride=2, padding=0, bias=False,
                                         norm=norm, activation=False)
        self.output = nn.ReLU() if is_relu else nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        shortcut = self.shortcut_path(shortcut)
        x = x + shortcut
        x = self.output(x)
        return x



class SHC(nn.Module):
    def __init__(self, channel, norm=True):
        super(SHC, self).__init__()

        self.conv1 = make_layers(channel * 2, int(channel / 2), kernel_size=1, stride=1, padding=0, norm=norm,
                                 activation=True, is_relu=True)
        self.conv2 = make_layers(int(channel / 2), int(channel / 2), kernel_size=3, stride=1, padding=1, norm=norm,
                                 activation=True, is_relu=True)
        self.conv3 = make_layers(int(channel / 2), channel, kernel_size=1, stride=1, padding=0, norm=norm,
                                 activation=False)

    def forward(self, x, shortcut):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x + shortcut
        return x


class GRB(nn.Module):
    def __init__(self, channel, dilation, norm=True):
        super(GRB, self).__init__()

        self.path1 = nn.Sequential(
            make_layers(channel, channel, kernel_size=(3, 1), stride=1, padding=(dilation, 0), dilation=dilation,
                        norm=norm, activation=True, is_relu=True),
            make_layers(channel, channel, kernel_size=(1, 7), stride=1, padding=(0, 3 * dilation), dilation=dilation,
                        norm=norm, activation=False)
        )
        self.path2 = nn.Sequential(
            make_layers(channel, channel, kernel_size=(1, 7), stride=1, padding=(0, 3 * dilation), dilation=dilation,
                        norm=norm, activation=True, is_relu=True),
            make_layers(channel, channel, kernel_size=(3, 1), stride=1, padding=(dilation, 0), dilation=dilation,
                        norm=norm, activation=False)
        )
        self.output = nn.ReLU()

    def forward(self, x):
        x1 = self.path1(x)
        x2 = self.path2(x)
        x = x + x1 + x2
        x = self.output(x)
        return x

class CNNdec(nn.Module):
    def __init__(self):
        super(CNNdec, self).__init__()

        self.GRB4 = GRB(1024, 1)
        self.decoder_stage4 = nn.Sequential(
            identity_block([1024, 256, 256, 1024], is_relu=True),
            identity_block([1024, 256, 256, 1024], is_relu=True),
            make_layers_transpose(1024, 512, kernel_size=4, stride=2, padding=1, bias=False, norm=True, activation=True,
                                  is_relu=True)
        )
        self.SHC4 = SHC(512)
        self.skip4 = nn.Sequential(
            nn.InstanceNorm2d(512, affine=True),
            nn.ReLU()
        )

        ## Stage -3
        # 8*16*512 to 16*32*256
        self.GRB3 = GRB(512, 2)
        self.decoder_stage3 = nn.Sequential(
            identity_block([512, 128, 128, 512], is_relu=True),
            identity_block([512, 128, 128, 512], is_relu=True),
            identity_block([512, 128, 128, 512], is_relu=True),
            make_layers_transpose(512, 256, kernel_size=4, stride=2, padding=1, bias=False, norm=True, activation=True,
                                  is_relu=True)
        )
        self.SHC3 = SHC(256)
        self.skip3 = nn.Sequential(
            nn.InstanceNorm2d(256, affine=True),
            nn.ReLU()
        )

        ## Stage -2
        # 16*32*256 to 32*64*128
        self.GRB2 = GRB(256, 4, norm=True)
        self.decoder_stage2 = nn.Sequential(
            identity_block([256, 64, 64, 256], is_relu=True, norm=True),
            identity_block([256, 64, 64, 256], is_relu=True, norm=True),
            identity_block([256, 64, 64, 256], is_relu=True, norm=True),
            identity_block([256, 64, 64, 256], is_relu=True, norm=True),
            make_layers_transpose(256, 128, kernel_size=4, stride=2, padding=1, bias=False, norm=True, activation=True,
                                  is_relu=True)
        )
        self.SHC2 = SHC(128, norm=True)
        self.skip2 = nn.Sequential(
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU()
        )

        ## Stage -1
        # 32*64*128 to 64*128*64
        self.decoder_stage1 = make_layers_transpose(128, 64, kernel_size=4, stride=2, padding=1, bias=False, norm=True,
                                                    activation=True, is_relu=True)
        self.SHC1 = SHC(64, norm=True)
        self.skip1 = nn.Sequential(
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU()
        )

        ## Stage -0
        # 64*128*64 to 128*256*3
        self.decoder_stage0 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False)
        # 128*256*3

    def forward(self, x, shortcut):
        # stage -4
        out = self.GRB4(x)
        out = self.decoder_stage4(out)
        #mask=torch.ones(size=[12,12]).long()
        #mask[2:10, 2:10] = 0
        sc = out[:,:,2:10, 2:10]
        shortcut_x = shortcut[3]
        merge = torch.cat((sc, shortcut_x), 1)
        merge = self.skip4(self.SHC4(merge, shortcut_x))
        out[:,:,2:10, 2:10] = merge

        # stage -3
        out = self.GRB3(out)
        out = self.decoder_stage3(out)
        sc = out[:, :, 4:20, 4:20]
        shortcut_x = shortcut[2]
        merge = torch.cat((sc, shortcut_x), 1)
        merge = self.skip3(self.SHC3(merge, shortcut_x))
        out[:,:,4:20, 4:20] = merge

        # stage -2
        out = self.GRB2(out)
        out = self.decoder_stage2(out)
        sc = out[:, :, 8:40, 8:40]
        shortcut_x = shortcut[1]
        merge = torch.cat((sc, shortcut_x), 1)
        merge = self.skip2(self.SHC2(merge, shortcut_x))
        out[:,:,8:40, 8:40] = merge

        # stage -1
        out = self.decoder_stage1(out)
        #merge = torch.cat((sc, shortcut_x), 1)
        #merge = self.skip1(self.SHC1(merge, shortcut_x))

        recon = self.decoder_stage0(out)
        return recon

    def forward123(self, x):
        # stage -4
        out = self.GRB4(x)
        out = self.decoder_stage4(out)

        # stage -3
        out = self.GRB3(out)
        out = self.decoder_stage3(out)

        # stage -2
        out = self.GRB2(out)
        out = self.decoder_stage2(out)

        # stage -1
        out = self.decoder_stage1(out)
        recon = self.decoder_stage0(out)
        return recon


class CNNenc(nn.Module):
    def __init__(self):
        super(CNNenc, self).__init__()
        ## Stage 1
        # 128*128*3 to 64*64*64
        self.encoder_stage1_conv1 = make_layers(3, 64, kernel_size=4, stride=2, padding=1, bias=False, norm=False,
                                                activation=True, is_relu=False)

        # 64*64*64 to 32*32*128
        self.encoder_stage1_conv2 = make_layers(64, 128, kernel_size=4, stride=2, padding=1, bias=False, norm=False,
                                                activation=True, is_relu=False)

        ## Stage 2
        # 32*32*128 to 16*16*256
        self.encoder_stage2 = nn.Sequential(
            convolutional_block([128, 64, 64, 256], norm=False),
            identity_block([256, 64, 64, 256], norm=False),
            identity_block([256, 64, 64, 256], norm=False)
        )

        ## Stage 3
        # 32*32*256 to 8*8*512
        self.encoder_stage3 = nn.Sequential(
            convolutional_block([256, 128, 128, 512]),
            identity_block([512, 128, 128, 512]),
            identity_block([512, 128, 128, 512]),
            identity_block([512, 128, 128, 512])
        )

        ## Stage 4
        # 8*8*512 to 4*4*1024
        self.encoder_stage4 = nn.Sequential(
            convolutional_block([512, 256, 256, 1024]),
            identity_block([1024, 256, 256, 1024]),
            identity_block([1024, 256, 256, 1024]),
            identity_block([1024, 256, 256, 1024]),
            identity_block([1024, 256, 256, 1024])
        )

    def forward(self, x):
        shortcut = []
        x = self.encoder_stage1_conv1(x)
        shortcut.append(x)
        x = self.encoder_stage1_conv2(x)
        shortcut.append(x)
        x = self.encoder_stage2(x)
        shortcut.append(x)
        x = self.encoder_stage3(x)
        shortcut.append(x)
        x = self.encoder_stage4(x)
        shortcut.append(x)

        return x, shortcut


class CNNenc_tiny(nn.Module):
    def __init__(self):
        super(CNNenc_tiny, self).__init__()
        ## Stage 1
        # 128*128*3 to 64*64*64
        self.encoder_stage1_conv1 = make_layers(3, 48, kernel_size=4, stride=2, padding=1, bias=False, norm=False,
                                                activation=True, is_relu=False)

        # 64*64*64 to 32*32*128
        self.encoder_stage1_conv2 = make_layers(48, 96, kernel_size=4, stride=2, padding=1, bias=False, norm=False,
                                                activation=True, is_relu=False)

        ## Stage 2
        # 32*32*128 to 16*16*256
        self.encoder_stage2 = nn.Sequential(
            convolutional_block([96, 48, 48, 192], norm=False),
            identity_block([192, 48, 48, 192], norm=False),
            identity_block([192, 48, 48, 192], norm=False)
        )

        ## Stage 3
        # 32*32*256 to 8*8*512
        self.encoder_stage3 = nn.Sequential(
            convolutional_block([192, 96, 96, 384]),
            identity_block([384, 96, 96, 384]),
            identity_block([384, 96, 96, 384]),
            identity_block([384, 96, 96, 384])
        )

        ## Stage 4
        # 8*8*512 to 4*4*1024
        self.encoder_stage4 = nn.Sequential(
            convolutional_block([384, 192, 192, 768]),
            identity_block([768, 192, 192, 768]),
            identity_block([768, 192, 192, 768]),
            identity_block([768, 192, 192, 768]),
            identity_block([768, 192, 192, 768])
        )

    def forward(self, x):
        shortcut = []
        x = self.encoder_stage1_conv1(x)
        shortcut.append(x)
        x = self.encoder_stage1_conv2(x)
        shortcut.append(x)
        x = self.encoder_stage2(x)
        shortcut.append(x)
        x = self.encoder_stage3(x)
        shortcut.append(x)
        x = self.encoder_stage4(x)
        shortcut.append(x)

        return x, shortcut