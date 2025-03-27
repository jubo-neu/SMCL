import torch
import torch.nn.functional as F
import vs_net.util
import torch.autograd.profiler as profiler
from torch.nn import DataParallel
from dotmap import DotMap


class _RenderWrapper(torch.nn.Module):
    def __init__(self, net, renderer, simple_output):
        super().__init__()
        self.net = net
        self.renderer = renderer
        self.simple_output = simple_output

    def forward(self, rays, want_weights=False):
        if rays.shape[0] == 0:
            return (torch.zeros(0, 3, device=rays.device), torch.zeros(0, device=rays.device))

        outputs = self.renderer(self.net, rays, want_weights=want_weights and not self.simple_output)
        if self.simple_output:
            if self.renderer.using_fine:
                rgb = outputs.fine.rgb
                depth = outputs.fine.depth
            else:
                rgb = outputs.coarse.rgb
                depth = outputs.coarse.depth
            return rgb, depth
        else:
            return outputs.toDict()


class NeRFRenderer(torch.nn.Module):
    def __init__(
        self,
        n_coarse=128,
        n_fine=0,
        n_fine_depth=0,
        noise_std=0.0,
        depth_std=0.01,
        eval_batch_size=100000,
        white_bkgd=False,
        lindisp=False,
        sched=None,
    ):
        super().__init__()
        self.n_coarse = n_coarse
        self.n_fine = n_fine
        self.n_fine_depth = n_fine_depth

        self.noise_std = noise_std
        self.depth_std = depth_std

        self.eval_batch_size = eval_batch_size
        self.white_bkgd = white_bkgd
        self.lindisp = lindisp
        if lindisp:
            print("Using linear displacement rays")
        self.using_fine = n_fine > 0
        self.sched = sched
        if sched is not None and len(sched) == 0:
            self.sched = None
        self.register_buffer("iter_idx", torch.tensor(0, dtype=torch.long), persistent=True)
        self.register_buffer("last_sched", torch.tensor(0, dtype=torch.long), persistent=True)

    def sample_coarse(self, rays):
        device = rays.device
        near, far = rays[:, -2:-1], rays[:, -1:]

        step = 1.0 / self.n_coarse
        B = rays.shape[0]
        z_steps = torch.linspace(0, 1 - step, self.n_coarse, device=device)
        z_steps = z_steps.unsqueeze(0).repeat(B, 1)
        z_steps += torch.rand_like(z_steps) * step
        if not self.lindisp:
            return near * (1 - z_steps) + far * z_steps
        else:
            return 1 / (1 / near * (1 - z_steps) + 1 / far * z_steps)

        return near * (1 - z_steps) + far * z_steps

    def sample_fine(self, rays, weights):
        device = rays.device
        B = rays.shape[0]

        weights = weights.detach() + 1e-5
        pdf = weights / torch.sum(weights, -1, keepdim=True)
        cdf = torch.cumsum(pdf, -1)
        cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], -1)

        u = torch.rand(B, self.n_fine - self.n_fine_depth, dtype=torch.float32, device=device)
        inds = torch.searchsorted(cdf, u, right=True).float() - 1.0
        inds = torch.clamp_min(inds, 0.0)

        z_steps = (inds + torch.rand_like(inds)) / self.n_coarse

        near, far = rays[:, -2:-1], rays[:, -1:]
        if not self.lindisp:
            z_samp = near * (1 - z_steps) + far * z_steps
        else:
            z_samp = 1 / (1 / near * (1 - z_steps) + 1 / far * z_steps)
        return z_samp

    def sample_fine_depth(self, rays, depth):
        z_samp = depth.unsqueeze(1).repeat((1, self.n_fine_depth))
        z_samp += torch.randn_like(z_samp) * self.depth_std
        z_samp = torch.max(torch.min(z_samp, rays[:, -1:]), rays[:, -2:-1])
        return z_samp

    def composite(self, model, rays, z_samp, coarse=True, sb=0):
        with profiler.record_function("renderer_composite"):
            B, K = z_samp.shape

            deltas = z_samp[:, 1:] - z_samp[:, :-1]

            delta_inf = rays[:, -1:] - z_samp[:, -1:]
            deltas = torch.cat([deltas, delta_inf], -1)

            points = rays[:, None, :3] + z_samp.unsqueeze(2) * rays[:, None, 3:6]
            points = points.reshape(-1, 3)

            use_viewdirs = hasattr(model, "use_viewdirs") and model.use_viewdirs

            val_all = []
            if sb > 0:
                points = points.reshape(sb, -1, 3)
                eval_batch_size = (self.eval_batch_size - 1) // sb + 1
                eval_batch_dim = 1
            else:
                eval_batch_size = self.eval_batch_size
                eval_batch_dim = 0

            split_points = torch.split(points, eval_batch_size, dim=eval_batch_dim)
            if use_viewdirs:
                dim1 = K
                viewdirs = rays[:, None, 3:6].expand(-1, dim1, -1)
                if sb > 0:
                    viewdirs = viewdirs.reshape(sb, -1, 3)
                else:
                    viewdirs = viewdirs.reshape(-1, 3)
                split_viewdirs = torch.split(viewdirs, eval_batch_size, dim=eval_batch_dim)
                for pnts, dirs in zip(split_points, split_viewdirs):
                    val_all.append(model(pnts, coarse=coarse, viewdirs=dirs))
            else:
                for pnts in split_points:
                    val_all.append(model(pnts, coarse=coarse))
            points = None
            viewdirs = None
            out = torch.cat(val_all, dim=eval_batch_dim)
            out = out.reshape(B, K, -1)

            rgbs = out[..., :3]
            sigmas = out[..., 3]
            if self.training and self.noise_std > 0.0:
                sigmas = sigmas + torch.randn_like(sigmas) * self.noise_std

            alphas = 1 - torch.exp(-deltas * torch.relu(sigmas))
            deltas = None
            sigmas = None
            alphas_shifted = torch.cat([torch.ones_like(alphas[:, :1]), 1 - alphas + 1e-10], -1)
            T = torch.cumprod(alphas_shifted, -1)
            weights = alphas * T[:, :-1]
            alphas = None
            alphas_shifted = None

            rgb_final = torch.sum(weights.unsqueeze(-1) * rgbs, -2)
            depth_final = torch.sum(weights * z_samp, -1)
            if self.white_bkgd:
                pix_alpha = weights.sum(dim=1)
                rgb_final = rgb_final + 1 - pix_alpha.unsqueeze(-1)
            return weights, rgb_final, depth_final

    def forward(self, model, rays, want_weights=False):

        with profiler.record_function("renderer_forward"):
            if self.sched is not None and self.last_sched.item() > 0:
                self.n_coarse = self.sched[1][self.last_sched.item() - 1]
                self.n_fine = self.sched[2][self.last_sched.item() - 1]

            assert len(rays.shape) == 3
            superbatch_size = rays.shape[0]
            rays = rays.reshape(-1, 8)

            z_coarse = self.sample_coarse(rays)
            coarse_composite = self.composite(model, rays, z_coarse, coarse=True, sb=superbatch_size)

            outputs = DotMap(coarse=self._format_outputs(coarse_composite, superbatch_size, want_weights=want_weights))

            if self.using_fine:
                all_samps = [z_coarse]
                if self.n_fine - self.n_fine_depth > 0:
                    all_samps.append(self.sample_fine(rays, coarse_composite[0].detach()))
                if self.n_fine_depth > 0:
                    all_samps.append(self.sample_fine_depth(rays, coarse_composite[2]))
                z_combine = torch.cat(all_samps, dim=-1)
                z_combine_sorted, argsort = torch.sort(z_combine, dim=-1)
                fine_composite = self.composite(model, rays, z_combine_sorted, coarse=False, sb=superbatch_size)
                outputs.fine = self._format_outputs(fine_composite, superbatch_size, want_weights=want_weights)

            return outputs

    def _format_outputs(self, rendered_outputs, superbatch_size, want_weights=False):
        weights, rgb, depth = rendered_outputs
        if superbatch_size > 0:
            rgb = rgb.reshape(superbatch_size, -1, 3)
            depth = depth.reshape(superbatch_size, -1)
            weights = weights.reshape(superbatch_size, -1, weights.shape[-1])
        ret_dict = DotMap(rgb=rgb, depth=depth)
        if want_weights:
            ret_dict.weights = weights
        return ret_dict

    def sched_step(self, steps=1):
        if self.sched is None:
            return
        self.iter_idx += steps
        while (self.last_sched.item() < len(self.sched[0]) and self.iter_idx.item() >= self.sched[0][self.last_sched.item()]):
            self.n_coarse = self.sched[1][self.last_sched.item()]
            self.n_fine = self.sched[2][self.last_sched.item()]
            print(
                "INFO: NeRF sampling resolution changed on schedule ==> c",
                self.n_coarse,
                "f",
                self.n_fine,
            )
            self.last_sched += 1

    @classmethod
    def from_conf(cls, conf, white_bkgd=False, lindisp=False, eval_batch_size=100000):
        return cls(
            conf.get_int("n_coarse", 128),
            conf.get_int("n_fine", 0),
            n_fine_depth=conf.get_int("n_fine_depth", 0),
            noise_std=conf.get_float("noise_std", 0.0),
            depth_std=conf.get_float("depth_std", 0.01),
            white_bkgd=conf.get_float("white_bkgd", white_bkgd),
            lindisp=lindisp,
            eval_batch_size=conf.get_int("eval_batch_size", eval_batch_size),
            sched=conf.get_list("sched", None),
        )

    def bind_parallel(self, net, gpus=None, simple_output=False):
        wrapped = _RenderWrapper(net, self, simple_output=simple_output)
        if gpus is not None and len(gpus) > 1:
            print("Using multi-GPU", gpus)
            wrapped = torch.nn.DataParallel(wrapped, gpus, dim=1)
        return wrapped
