import torch.nn as nn
import torch


class FFCSE_block(nn.Module):
    def __init__(self, channels, ratio_g):
        super(FFCSE_block, self).__init__()
        channels = channels.shape[1]
        in_cg = int(channels * ratio_g)
        in_cl = channels - in_cg
        r = 4

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(channels, channels // r, kernel_size=1, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv_a2l = None if in_cl == 0 else nn.Conv2d(channels // r, in_cl, kernel_size=1, bias=True)
        self.conv_a2g = None if in_cg == 0 else nn.Conv2d(channels // r, in_cg, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x if type(x) is tuple else (x, 0)
        id_l, id_g = x

        x = id_l if type(id_g) is int else torch.cat([id_l, id_g], dim=1)
        x = self.avgpool(x)
        x = self.relu1(self.conv1(x))

        x_l = 0 if self.conv_a2l is None else id_l * self.sigmoid(self.conv_a2l(x))
        x_g = 0 if self.conv_a2g is None else id_g * self.sigmoid(self.conv_a2g(x))
        return x_l + x_g


class FFCSE_block1(nn.Module):
    def __init__(self, channels):
        super(FFCSE_block1, self).__init__()
        channels = channels.shape[1]

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

        self.relu1 = nn.ReLU(inplace=True)

        self.conv1_1 = nn.Conv2d(channels, channels // 2, kernel_size=1, bias=True)

        self.conv_a2l = nn.Conv2d(32, 32, kernel_size=1, bias=True)
        self.conv_a2l_1 = nn.Conv2d(32, 16, kernel_size=1, bias=True)

        self.conv_a2g = nn.Conv2d(32, 32, kernel_size=1, bias=True)
        self.conv_a2g_1 = nn.Conv2d(32, 16, kernel_size=1, bias=True)

        self.fft_norm = 'ortho'

    def forward(self, x):
        x = x if type(x) is tuple else (x, 0)
        id_l, id_g = x

        x = id_l if type(id_g) is int else torch.cat([id_l, id_g], dim=1)
        x9 = self.conv1(self.relu1(self.conv1(x)))
        x10 = self.conv1_1(self.relu1(self.conv1(x)))

        x8 = self.avgpool(x9)

        x1 = self.conv_a2l_1(self.relu1(self.conv_a2l(x8)))
        x2 = self.conv_a2g_1(self.relu1(self.conv_a2g(x8)))

        fft_dim = (-2, -1)
        ffted = torch.fft.rfftn(x8, dim=fft_dim, norm=self.fft_norm)

        split_size = 16
        x3, x4 = torch.split(ffted, [split_size, split_size], dim=1)

        x1 = x1 * x3
        x2 = x2 * x4
        ffted = x1 + x2
        ifft_shape_slice = x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)
        output = output * x10
        output = output + self.conv1_1(x)

        return output


class CNNSE_block(nn.Module):
    def __init__(self, channels, r=8):
        super(CNNSE_block, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        channels = channels.shape[1]
        self.conv1 = nn.Conv2d(channels, channels // r, kernel_size=1, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels // r, channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        a = self.avgpool(x)
        a = self.conv1(a)
        a = self.relu1(a)
        a = self.conv2(a)
        x = x * self.sigmoid(a)
        return x


class PAM(nn.Module):

    def __init__(self, in_dim):

        super(PAM, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_dim*2,
            out_channels=2,
            kernel_size=3,
            padding=1
        )

        self.v_rgb = nn.Parameter(torch.randn((1, in_dim, 1, 1)), requires_grad=True)
        self.v_freq = nn.Parameter(torch.randn((1, in_dim, 1, 1)), requires_grad=True)

    def forward(self, rgb, freq):

        attmap = self.conv(torch.cat((rgb, freq), 1))
        attmap = torch.sigmoid(attmap)

        rgb = attmap[:, 0:1, :, :] * rgb * self.v_rgb
        freq = attmap[:, 1:, :, :] * freq * self.v_freq
        out = rgb + freq

        return out
