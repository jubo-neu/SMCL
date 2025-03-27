import torch
from torch import nn
import numpy as np
import vs_net.util


class ImplicitNet(nn.Module):
    def __init__(
        self,
        d_in,
        dims,
        skip_in=(),
        d_out=4,
        geometric_init=True,
        radius_init=0.3,
        beta=0.0,
        output_init_gain=2.0,
        num_position_inputs=3,
        sdf_scale=1.0,
        dim_excludes_skip=False,
        combine_layer=1000,
        combine_type="average",
    ):

        super().__init__()

        dims = [d_in] + dims + [d_out]
        if dim_excludes_skip:
            for i in range(1, len(dims) - 1):
                if i in skip_in:
                    dims[i] += d_in

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.dims = dims
        self.combine_layer = combine_layer
        self.combine_type = combine_type

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in skip_in:
                out_dim = dims[layer + 1] - d_in
            else:
                out_dim = dims[layer + 1]
            lin = nn.Linear(dims[layer], out_dim)

            if geometric_init:
                if layer == self.num_layers - 2:
                    nn.init.normal_(lin.weight[0], mean=-np.sqrt(np.pi) / np.sqrt(dims[layer]) * sdf_scale, std=0.00001)
                    nn.init.constant_(lin.bias[0], radius_init)
                    if d_out > 1:
                        nn.init.normal_(lin.weight[1:], mean=0.0, std=output_init_gain)
                        nn.init.constant_(lin.bias[1:], 0.0)
                else:
                    nn.init.constant_(lin.bias, 0.0)
                    nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                if d_in > num_position_inputs and (layer == 0 or layer in skip_in):
                    nn.init.constant_(lin.weight[:, -d_in + num_position_inputs :], 0.0)
            else:
                nn.init.constant_(lin.bias, 0.0)
                nn.init.kaiming_normal_(lin.weight, a=0, mode="fan_in")

            setattr(self, "lin" + str(layer), lin)

        if beta > 0:
            self.activation = nn.Softplus(beta=beta)
        else:
            self.activation = nn.ReLU()

    def forward(self, x, combine_inner_dims=(1,)):
        x_init = x
        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))

            if layer == self.combine_layer:
                x = vs_net.util.combine_interleaved(x, combine_inner_dims, self.combine_type)
                x_init = vs_net.util.combine_interleaved(x_init, combine_inner_dims, self.combine_type)

            if layer < self.combine_layer and layer in self.skip_in:
                x = torch.cat([x, x_init], -1) / np.sqrt(2)

            x = lin(x)
            if layer < self.num_layers - 2:
                x = self.activation(x)

        return x

    @classmethod
    def from_conf(cls, conf, d_in, **kwargs):
        return cls(
            d_in,
            conf.get_list("dims"),
            skip_in=conf.get_list("skip_in"),
            beta=conf.get_float("beta", 0.0),
            dim_excludes_skip=conf.get_bool("dim_excludes_skip", False),
            combine_layer=conf.get_int("combine_layer", 1000),
            combine_type=conf.get_string("combine_type", "average"),
            **kwargs
        )
