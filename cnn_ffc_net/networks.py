from collections import OrderedDict

import torch
import torch.nn as nn

from cnn_ffc_net.ffc import FFC_BN_ACT, ConcatTupleLayer
from cnn_ffc_net.attentionblock import FFCSE_block, CNNSE_block


def get_model(model_name, in_channels=3, ratio=0.5):
    if model_name == "cnn_cnn":
        model = cnn_ffc(in_channels, ffc=False)
    elif model_name == "cnn_ffc":
        model = cnn_ffc(in_channels, ffc=True, ratio_in=ratio)
    else:
        print("Model name not found")
        assert False

    return model


class cnn_ffc(nn.Module):

    def __init__(self, in_channels=3, init_features=8, ratio_in=0.5, ffc=True):
        super(cnn_ffc, self).__init__()

        self.ffc = ffc
        self.ratio_in = ratio_in

        features = init_features

        self.encoder1 = cnn_ffc._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = cnn_ffc._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = cnn_ffc._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = cnn_ffc._block(features * 4, features * 4, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        if ffc:
            self.encoder1_f = FFC_BN_ACT(in_channels, features, kernel_size=1, ratio_gin=0, ratio_gout=ratio_in)
            self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_f = FFC_BN_ACT(features, features * 2, kernel_size=1, ratio_gin=ratio_in, ratio_gout=ratio_in)
            self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_f = FFC_BN_ACT(features * 2, features * 4, kernel_size=1, ratio_gin=ratio_in, ratio_gout=ratio_in)
            self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_f = FFC_BN_ACT(features * 4, features * 4, kernel_size=1, ratio_gin=ratio_in, ratio_gout=ratio_in)
            self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)

        else:
            self.encoder1_f = cnn_ffc._block(in_channels, features, name="enc1_2")
            self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_f = cnn_ffc._block(features, features * 2, name="enc2_2")
            self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_f = cnn_ffc._block(features * 2, features * 4, name="enc3_2")
            self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_f = cnn_ffc._block(features * 4, features * 4, name="enc4_2")
            self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = cnn_ffc._block(features * 8, features * 16, name="bottleneck")

        self.softmax = nn.Softmax2d()
        self.catLayer = ConcatTupleLayer()

    def apply_fft(self, inp, batch):
        ffted = torch.fft.fftn(inp)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        return ffted

    def forward(self, x):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        enc1 = self.encoder1(x)
        attenc1 = CNNSE_block(enc1).to(device)
        attenc1 = attenc1(enc1)
        enc1 = enc1 + attenc1

        enc2 = self.encoder2(self.pool1(enc1))
        attenc2 = CNNSE_block(enc2).to(device)
        attenc2 = attenc2(enc2)
        enc2 = enc2 + attenc2

        enc3 = self.encoder3(self.pool2(enc2))
        attenc3 = CNNSE_block(enc3).to(device)
        attenc3 = attenc3(enc3)
        enc3 = enc3 + attenc3

        enc4 = self.encoder4(self.pool3(enc3))
        enc4_2 = self.pool4(enc4)

        if self.ffc:
            enc1_f = self.encoder1_f(x)
            enc1_l, enc1_g = enc1_f
            attenc1_l = FFCSE_block(enc1_l, 0).to(device)
            attenc1_l = attenc1_l(enc1_l)
            attenc1_g = FFCSE_block(enc1_g, 0).to(device)
            attenc1_g = attenc1_g(enc1_g)

            enc2_f = self.encoder2_f((self.pool1_f(attenc1_l), self.pool1_f(attenc1_g)))
            enc2_l, enc2_g = enc2_f
            attenc2_l = FFCSE_block(enc2_l, 0).to(device)
            attenc2_l = attenc2_l(enc2_l)
            attenc2_g = FFCSE_block(enc2_g, 0).to(device)
            attenc2_g = attenc2_g(enc2_g)

            enc3_f = self.encoder3_f((self.pool2_f(attenc2_l), self.pool2_f(attenc2_g)))
            enc3_l, enc3_g = enc3_f
            attenc3_l = FFCSE_block(enc3_l, 0).to(device)
            attenc3_l = attenc3_l(enc3_l)
            attenc3_g = FFCSE_block(enc3_g, 0).to(device)
            attenc3_g = attenc3_g(enc3_g)

            enc4_f = self.encoder4_f((self.pool3_f(attenc3_l), self.pool3_f(attenc3_g)))
            enc4_l, enc4_g = enc4_f

            enc4_f2 = self.catLayer((self.pool4_f(enc4_l), self.pool4_f(enc4_g)))

        else:
            enc1_f = self.encoder1_f(x)
            enc2_f = self.encoder2_f(self.pool1_f(enc1_f))
            enc3_f = self.encoder3_f(self.pool2_f(enc2_f))
            enc4_f = self.encoder4_f(self.pool3_f(enc3_f))
            enc4_f2 = self.pool4(enc4_f)

        attenc4_2 = CNNSE_block(enc4_2).to(device)
        attenc4_2 = attenc4_2(enc4_2)

        attenc4_f2 = FFCSE_block(enc4_f2, 0).to(device)
        attenc4_f2 = attenc4_f2(enc4_f2)

        enc4_2 = enc4_2 + attenc4_2
        enc4_f2 = enc4_f2 + attenc4_f2
        bottleneck = torch.cat((enc4_2, enc4_f2), 1)

        bottleneck = self.bottleneck(bottleneck)

        return bottleneck

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


class FFCNet(nn.Module):

    def __init__(self, in_channels=3, init_features=8, ratio_in=0.5, ffc=True, cat_merge=True):
        super(FFCNet, self).__init__()

        self.ffc = ffc
        self.ratio_in = ratio_in
        self.cat_merge = cat_merge

        features = init_features

        # FFC #
        self.encoder1_f = FFC_BN_ACT(in_channels, features, kernel_size=1, ratio_gin=0, ratio_gout=ratio_in)
        self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2_f = FFC_BN_ACT(features, features * 2, kernel_size=1, ratio_gin=ratio_in, ratio_gout=ratio_in)
        self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3_f = FFC_BN_ACT(features * 2, features * 4, kernel_size=1, ratio_gin=ratio_in, ratio_gout=ratio_in)
        self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4_f = FFC_BN_ACT(features * 4, features * 16, kernel_size=1, ratio_gin=ratio_in, ratio_gout=ratio_in)
        self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc4_f2 = nn.Linear(4, 4)

        self.softmax = nn.Softmax2d()
        self.catLayer = ConcatTupleLayer()

    def apply_fft(self, inp, batch):
        ffted = torch.fft.fftn(inp)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        return ffted

    def forward(self, x):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        enc1_f = self.encoder1_f(x)
        enc1_l, enc1_g = enc1_f
        attenc1_l = FFCSE_block(enc1_l, 0).to(device)
        attenc1_l = attenc1_l(enc1_l)
        attenc1_g = FFCSE_block(enc1_g, 0).to(device)
        attenc1_g = attenc1_g(enc1_g)

        enc2_f = self.encoder2_f((self.pool1_f(attenc1_l), self.pool1_f(attenc1_g)))
        enc2_l, enc2_g = enc2_f
        attenc2_l = FFCSE_block(enc2_l, 0).to(device)
        attenc2_l = attenc2_l(enc2_l)
        attenc2_g = FFCSE_block(enc2_g, 0).to(device)
        attenc2_g = attenc2_g(enc2_g)

        enc3_f = self.encoder3_f((self.pool2_f(attenc2_l), self.pool2_f(attenc2_g)))
        enc3_l, enc3_g = enc3_f
        attenc3_l = FFCSE_block(enc3_l, 0).to(device)
        attenc3_l = attenc3_l(enc3_l)
        attenc3_g = FFCSE_block(enc3_g, 0).to(device)
        attenc3_g = attenc3_g(enc3_g)

        enc4_f = self.encoder4_f((self.pool3_f(attenc3_l), self.pool3_f(attenc3_g)))
        enc4_l, enc4_g = enc4_f

        enc4_f2 = self.catLayer((self.pool4_f(enc4_l), self.pool4_f(enc4_g)))

        attenc4_f2 = FFCSE_block(enc4_f2, 0).to(device)
        attenc4_f2 = attenc4_f2(enc4_f2)

        enc4_f2 = enc4_f2 + attenc4_f2

        return enc4_f2


class FFCNett(nn.Module):

    def __init__(self, in_channels=3, init_features=8, ratio_in=0.5, ffc=True, cat_merge=True):
        super(FFCNett, self).__init__()

        self.ffc = ffc
        self.ratio_in = ratio_in
        self.cat_merge = cat_merge

        features = init_features

        # FFC #
        self.encoder1_f = FFC_BN_ACT(in_channels, features, kernel_size=1, ratio_gin=0, ratio_gout=ratio_in)
        self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2_f = FFC_BN_ACT(features, features * 2, kernel_size=1, ratio_gin=ratio_in, ratio_gout=ratio_in)
        self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3_f = FFC_BN_ACT(features * 2, features * 4, kernel_size=1, ratio_gin=ratio_in, ratio_gout=ratio_in)
        self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4_f = FFC_BN_ACT(features * 4, features * 16, kernel_size=1, ratio_gin=ratio_in, ratio_gout=ratio_in)
        self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc4_f2 = nn.Linear(4, 4)

        self.softmax = nn.Softmax2d()
        self.catLayer = ConcatTupleLayer()

    def apply_fft(self, inp, batch):
        ffted = torch.fft.fftn(inp)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        return ffted

    def forward(self, x):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        enc1_f = self.encoder1_f(x)
        enc1_l, enc1_g = enc1_f
        attenc1_l = FFCSE_block(enc1_l, 0).to(device)
        attenc1_l = attenc1_l(enc1_l)
        attenc1_g = FFCSE_block(enc1_g, 0).to(device)
        attenc1_g = attenc1_g(enc1_g)

        enc2_f = self.encoder2_f((self.pool1_f(attenc1_l), self.pool1_f(attenc1_g)))
        enc2_l, enc2_g = enc2_f
        attenc2_l = FFCSE_block(enc2_l, 0).to(device)
        attenc2_l = attenc2_l(enc2_l)
        attenc2_g = FFCSE_block(enc2_g, 0).to(device)
        attenc2_g = attenc2_g(enc2_g)

        enc3_f = self.encoder3_f((self.pool2_f(attenc2_l), self.pool2_f(attenc2_g)))
        enc3_l, enc3_g = enc3_f
        attenc3_l = FFCSE_block(enc3_l, 0).to(device)
        attenc3_l = attenc3_l(enc3_l)
        attenc3_g = FFCSE_block(enc3_g, 0).to(device)
        attenc3_g = attenc3_g(enc3_g)

        enc4_f = self.encoder4_f((self.pool3_f(attenc3_l), self.pool3_f(attenc3_g)))
        enc4_l, enc4_g = enc4_f

        enc4_f2 = self.catLayer((self.pool4_f(enc4_l), self.pool4_f(enc4_g)))

        attenc4_f2 = FFCSE_block(enc4_f2, 0).to(device)
        attenc4_f2 = attenc4_f2(enc4_f2)

        enc4_f2 = enc4_f2 + attenc4_f2

        return enc4_f2
