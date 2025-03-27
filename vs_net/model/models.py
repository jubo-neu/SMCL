import torch
from .SMCL_encoder import ImageEncoder
from .code_1 import PositionalEncoding
from .model_util import make_encoder, make_mlp
import torch.autograd.profiler as profiler
from vs_net.util import repeat_interleave
import os
import os.path as osp
import warnings


class SMCLNet(torch.nn.Module):
    def __init__(self, conf, stop_encoder_grad=False):
        super().__init__()
        self.encoder = make_encoder(conf["encoder"])
        self.use_encoder = conf.get_bool("use_encoder", True)
        self.use_xyz = conf.get_bool("use_xyz", False)

        assert self.use_encoder or self.use_xyz

        self.normalize_z = conf.get_bool("normalize_z", True)
        self.stop_encoder_grad = (stop_encoder_grad)
        self.use_code = conf.get_bool("use_code", False)
        self.use_code_viewdirs = conf.get_bool("use_code_viewdirs", True)
        self.use_viewdirs = conf.get_bool("use_viewdirs", False)
        self.use_global_encoder = conf.get_bool("use_global_encoder", False)

        d_latent = self.encoder.latent_size if self.use_encoder else 0
        d_in = 3 if self.use_xyz else 1

        if self.use_viewdirs and self.use_code_viewdirs:
            d_in += 3
        if self.use_code and d_in > 0:
            self.code = PositionalEncoding.from_conf(conf["code"], d_in=d_in)
            d_in = self.code.d_out
        if self.use_viewdirs and not self.use_code_viewdirs:
            d_in += 3

        if self.use_global_encoder:
            self.global_encoder = ImageEncoder.from_conf(conf["global_encoder"])
            self.global_latent_size = self.global_encoder.latent_size
            d_latent += self.global_latent_size

        d_out = 4

        self.latent_size = self.encoder.latent_size
        self.mlp_coarse = make_mlp(conf["mlp_coarse"], d_in, d_latent, d_out=d_out)
        self.mlp_fine = make_mlp(
            conf["mlp_fine"], d_in, d_latent, d_out=d_out, allow_empty=True
        )
        self.register_buffer("poses", torch.empty(1, 3, 4), persistent=False)
        self.register_buffer("image_shape", torch.empty(2), persistent=False)

        self.d_in = d_in
        self.d_out = d_out
        self.d_latent = d_latent
        self.register_buffer("focal", torch.empty(1, 2), persistent=False)
        self.register_buffer("c", torch.empty(1, 2), persistent=False)

        self.num_objs = 0
        self.num_views_per_obj = 1

    def encode(self, images, poses, focal, z_bounds=None, c=None):
        self.num_objs = images.size(0)
        if len(images.shape) == 5:
            assert len(poses.shape) == 4
            assert poses.size(1) == images.size(1)
            self.num_views_per_obj = images.size(1)
            images = images.reshape(-1, *images.shape[2:])
            poses = poses.reshape(-1, 4, 4)
        else:
            self.num_views_per_obj = 1

        self.encoder(images)
        rot = poses[:, :3, :3].transpose(1, 2)
        trans = -torch.bmm(rot, poses[:, :3, 3:])
        self.poses = torch.cat((rot, trans), dim=-1)

        self.image_shape[0] = images.shape[-1]
        self.image_shape[1] = images.shape[-2]

        if len(focal.shape) == 0:
            focal = focal[None, None].repeat((1, 2))
        elif len(focal.shape) == 1:
            focal = focal.unsqueeze(-1).repeat((1, 2))
        else:
            focal = focal.clone()
        self.focal = focal.float()
        self.focal[..., 1] *= -1.0

        if c is None:
            c = (self.image_shape * 0.5).unsqueeze(0)
        elif len(c.shape) == 0:
            c = c[None, None].repeat((1, 2))
        elif len(c.shape) == 1:
            c = c.unsqueeze(-1).repeat((1, 2))
        self.c = c

        if self.use_global_encoder:
            self.global_encoder(images)

    def forward(self, xyz, coarse=True, viewdirs=None, far=False):
        with profiler.record_function("model_inference"):
            SB, B, _ = xyz.shape
            NS = self.num_views_per_obj

            xyz = repeat_interleave(xyz, NS)
            xyz_rot = torch.matmul(self.poses[:, None, :3, :3], xyz.unsqueeze(-1))[..., 0]
            xyz = xyz_rot + self.poses[:, None, :3, 3]

            if self.d_in > 0:
                if self.use_xyz:
                    if self.normalize_z:
                        z_feature = xyz_rot.reshape(-1, 3)
                    else:
                        z_feature = xyz.reshape(-1, 3)
                else:
                    if self.normalize_z:
                        z_feature = -xyz_rot[..., 2].reshape(-1, 1)
                    else:
                        z_feature = -xyz[..., 2].reshape(-1, 1)

                if self.use_code and not self.use_code_viewdirs:
                    z_feature = self.code(z_feature)

                if self.use_viewdirs:
                    assert viewdirs is not None
                    viewdirs = viewdirs.reshape(SB, B, 3, 1)
                    viewdirs = repeat_interleave(viewdirs, NS)
                    viewdirs = torch.matmul(self.poses[:, None, :3, :3], viewdirs)
                    viewdirs = viewdirs.reshape(-1, 3)
                    z_feature = torch.cat((z_feature, viewdirs), dim=1)

                if self.use_code and self.use_code_viewdirs:
                    z_feature = self.code(z_feature)

                mlp_input = z_feature

            if self.use_encoder:
                uv = -xyz[:, :, :2] / xyz[:, :, 2:]
                uv *= repeat_interleave(self.focal.unsqueeze(1), NS if self.focal.shape[0] > 1 else 1)
                uv += repeat_interleave(self.c.unsqueeze(1), NS if self.c.shape[0] > 1 else 1)
                latent = self.encoder.index(uv, None, self.image_shape)

                if self.stop_encoder_grad:
                    latent = latent.detach()
                latent = latent.transpose(1, 2).reshape(-1, self.latent_size)

                if self.d_in == 0:
                    mlp_input = latent
                else:
                    mlp_input = torch.cat((latent, z_feature), dim=-1)

            if self.use_global_encoder:
                global_latent = self.global_encoder.latent
                assert mlp_input.shape[0] % global_latent.shape[0] == 0
                num_repeats = mlp_input.shape[0] // global_latent.shape[0]
                global_latent = repeat_interleave(global_latent, num_repeats)
                mlp_input = torch.cat((global_latent, mlp_input), dim=-1)

            combine_index = None
            dim_size = None

            if coarse or self.mlp_fine is None:
                mlp_output = self.mlp_coarse(
                    mlp_input,
                    combine_inner_dims=(self.num_views_per_obj, B),
                    combine_index=combine_index,
                    dim_size=dim_size,
                )
            else:
                mlp_output = self.mlp_fine(
                    mlp_input,
                    combine_inner_dims=(self.num_views_per_obj, B),
                    combine_index=combine_index,
                    dim_size=dim_size,
                )

            mlp_output = mlp_output.reshape(-1, B, self.d_out)

            rgb = mlp_output[..., :3]
            sigma = mlp_output[..., 3:4]

            output_list = [torch.sigmoid(rgb), torch.relu(sigma)]
            output = torch.cat(output_list, dim=-1)
            output = output.reshape(SB, B, -1)
        return output

    def load_weights(self, args, opt_init=False, strict=True, device=None):
        # TODO: make backups
        if opt_init and not args.resume:
            return
        ckpt_name = ("SMCL_init" if opt_init or not args.resume else "SMCL_latest")
        model_path = "%s/%s/%s" % (args.checkpoints_path, args.name, ckpt_name)

        if device is None:
            device = self.poses.device

        if os.path.exists(model_path):
            print("Load", model_path)
            self.load_state_dict(torch.load(model_path, map_location=device), strict=strict)
        elif not opt_init:
            warnings.warn(
                (
                    "WARNING: {} does not exist, not loaded!! Model will be re-initialized.\n"
                    + "If you are trying to load a pretrained model, STOP since it's not in the right place. "
                    + "If training, unless you are start in a new experiment, please remember to pass --resume."
                ).format(model_path)
            )
        return self

    def save_weights(self, args, opt_init=False):
        from shutil import copyfile

        ckpt_name = "SMCL_init" if opt_init else "SMCL_latest"
        backup_name = "SMCL_init_backup" if opt_init else "SMCL_backup"

        ckpt_path = osp.join(args.checkpoints_path, args.name, ckpt_name)
        ckpt_backup_path = osp.join(args.checkpoints_path, args.name, backup_name)

        if osp.exists(ckpt_path):
            copyfile(ckpt_path, ckpt_backup_path)
        torch.save(self.state_dict(), ckpt_path)
        return self
