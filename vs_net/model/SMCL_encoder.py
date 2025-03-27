from collections import OrderedDict

import torch
import torch.nn as nn

from cnn_ffc_net.ffc import FFC_BN_ACT, ConcatTupleLayer
from cnn_ffc_net.networks import cnn_ffc


class ImageEncoder(nn.Module):
    def __init__(self, backbone="cnn_ffc", pretrained=True, latent_size=128, pretrained_weights_path="pretrained_weights_path"):

        super().__init__()

        self.model = cnn_ffc()

        if pretrained_weights_path:
            self.load_pretrained_weights(pretrained_weights_path)

        self.register_buffer("latent", torch.empty(1, 1), persistent=False)
        self.latent_size = latent_size

        if latent_size != 2048:
            self.fc = nn.Linear(2048, latent_size)

    def index(self, uv, cam_z=None, image_size=(), z_bounds=()):
        return self.latent.unsqueeze(-1).expand(-1, -1, uv.shape[1])

    def forward(self, x):
        x = x.to(device=self.latent.device)
        x = self.model(x)

        if self.latent_size != 2048:
            x = x.view(x.size(0), -1)
            x = self.fc(x)

        self.latent = x

        return self.latent

    def load_pretrained_weights(self, pretrained_weights_path):
        pretrained_dict = torch.load(pretrained_weights_path)
        model_dict = self.model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)

    @classmethod
    def from_conf(cls, conf):
        return cls(
            conf.get_string("backbone"),
            pretrained=conf.get_bool("pretrained", True),
            latent_size=conf.get_int("latent_size", 128),
            pretrained_weights_path=conf.get_string("pretrained_weights_path")
        )
