from torch import nn
import torch

import torch.autograd.profiler as profiler
import vs_net.util


class ResnetBlockFC(nn.Module):
    def __init__(self, size_in, size_out=None, size_h=None, beta=0.0):
        super().__init__()
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out

        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)

        nn.init.constant_(self.fc_0.bias, 0.0)
        nn.init.kaiming_normal_(self.fc_0.weight, a=0, mode="fan_in")
        nn.init.constant_(self.fc_1.bias, 0.0)
        nn.init.zeros_(self.fc_1.weight)

        if beta > 0:
            self.activation = nn.Softplus(beta=beta)
        else:
            self.activation = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
            nn.init.constant_(self.shortcut.bias, 0.0)
            nn.init.kaiming_normal_(self.shortcut.weight, a=0, mode="fan_in")

    def forward(self, x):
        with profiler.record_function("resblock"):
            net = self.fc_0(self.activation(x))
            dx = self.fc_1(self.activation(net))

            if self.shortcut is not None:
                x_s = self.shortcut(x)
            else:
                x_s = x
            return x_s + dx


class ResnetFC(nn.Module):
    def __init__(
        self,
        d_in,
        d_out=4,
        n_blocks=5,
        d_latent=0,
        d_hidden=128,
        beta=0.0,
        combine_layer=1000,
        combine_type="average",
        use_spade=False,
    ):

        super().__init__()
        if d_in > 0:
            self.lin_in = nn.Linear(d_in, d_hidden)
            nn.init.constant_(self.lin_in.bias, 0.0)
            nn.init.kaiming_normal_(self.lin_in.weight, a=0, mode="fan_in")

        self.lin_out = nn.Linear(d_hidden, d_out)
        nn.init.constant_(self.lin_out.bias, 0.0)
        nn.init.kaiming_normal_(self.lin_out.weight, a=0, mode="fan_in")

        self.n_blocks = n_blocks
        self.d_latent = d_latent
        self.d_in = d_in
        self.d_out = d_out
        self.d_hidden = d_hidden

        self.combine_layer = combine_layer
        self.combine_type = combine_type
        self.use_spade = use_spade

        self.blocks = nn.ModuleList([ResnetBlockFC(d_hidden, beta=beta) for i in range(n_blocks)])

        if d_latent != 0:
            n_lin_z = min(combine_layer, n_blocks)
            self.lin_z = nn.ModuleList([nn.Linear(d_latent, d_hidden) for i in range(n_lin_z)])
            for i in range(n_lin_z):
                nn.init.constant_(self.lin_z[i].bias, 0.0)
                nn.init.kaiming_normal_(self.lin_z[i].weight, a=0, mode="fan_in")

            if self.use_spade:
                self.scale_z = nn.ModuleList([nn.Linear(d_latent, d_hidden) for _ in range(n_lin_z)])
                for i in range(n_lin_z):
                    nn.init.constant_(self.scale_z[i].bias, 0.0)
                    nn.init.kaiming_normal_(self.scale_z[i].weight, a=0, mode="fan_in")

        if beta > 0:
            self.activation = nn.Softplus(beta=beta)
        else:
            self.activation = nn.ReLU()

    def forward(self, zx, combine_inner_dims=(1,), combine_index=None, dim_size=None):
        with profiler.record_function("resnetfc_infer"):
            assert zx.size(-1) == self.d_latent + self.d_in
            if self.d_latent > 0:
                z = zx[..., : self.d_latent]
                x = zx[..., self.d_latent :]
            else:
                x = zx
            if self.d_in > 0:
                x = self.lin_in(x)
            else:
                x = torch.zeros(self.d_hidden, device=zx.device)

            for blkid in range(self.n_blocks):
                if blkid == self.combine_layer:
                    x = vs_net.util.combine_interleaved(x, combine_inner_dims, self.combine_type)

                if self.d_latent > 0 and blkid < self.combine_layer:
                    tz = self.lin_z[blkid](z)
                    if self.use_spade:
                        sz = self.scale_z[blkid](z)
                        x = sz * x + tz
                    else:
                        x = x + tz

                x = self.blocks[blkid](x)
            out = self.lin_out(self.activation(x))
            return out

    @classmethod
    def from_conf(cls, conf, d_in, **kwargs):
        return cls(
            d_in,
            n_blocks=conf.get_int("n_blocks", 5),
            d_hidden=conf.get_int("d_hidden", 512),
            beta=conf.get_float("beta", 0.0),
            combine_layer=conf.get_int("combine_layer", 1000),
            combine_type=conf.get_string("combine_type", "average"),
            use_spade=conf.get_bool("use_spade", False),
            **kwargs
        )
