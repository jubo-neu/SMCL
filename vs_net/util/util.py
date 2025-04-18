import cv2
import numpy as np
import torch
from torchvision import transforms
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import functools
import math
import warnings


def image_float_to_uint8(img):
    vmin = np.min(img)
    vmax = np.max(img)
    if vmax - vmin < 1e-10:
        vmax += 1e-10
    img = (img - vmin) / (vmax - vmin)
    img *= 255.0
    return img.astype(np.uint8)


def cmap(img, color_map=cv2.COLORMAP_HOT):
    return cv2.applyColorMap(image_float_to_uint8(img), color_map)


def batched_index_select_nd(t, inds):
    return t.gather(1, inds[(...,) + (None,) * (len(t.shape) - 2)].expand(-1, -1, *t.shape[2:]))


def batched_index_select_nd_last(t, inds):
    dummy = inds.unsqueeze(-2).expand(*inds.shape[:-1], t.size(-2), inds.size(-1))
    out = t.gather(-1, dummy)
    return out


def repeat_interleave(input, repeats, dim=0):
    output = input.unsqueeze(1).expand(-1, repeats, *input.shape[1:])
    return output.reshape(-1, *input.shape[1:])


def get_image_to_tensor_balanced(image_size=0):
    ops = []
    if image_size > 0:
        ops.append(transforms.Resize(image_size))
    ops.extend([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])
    return transforms.Compose(ops)


def get_mask_to_tensor():
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0,), (1.0,))])


def homogeneous(points):
    return F.pad(points, (0, 1), "constant", 1.0)


def gen_grid(*args, ij_indexing=False):
    return torch.from_numpy(np.vstack(np.meshgrid(*(np.linspace(lo, hi, sz, dtype=np.float32) for lo, hi, sz in args), indexing="ij" if ij_indexing else "xy")).reshape(len(args), -1).T)


def unproj_map(width, height, f, c=None, device="cpu"):
    if c is None:
        c = [width * 0.5, height * 0.5]
    else:
        c = c.squeeze()
    if isinstance(f, float):
        f = [f, f]
    elif len(f.shape) == 0:
        f = f[None].expand(2)
    elif len(f.shape) == 1:
        f = f.expand(2)
    Y, X = torch.meshgrid(torch.arange(height, dtype=torch.float32) - float(c[1]), torch.arange(width, dtype=torch.float32) - float(c[0]))
    X = X.to(device=device) / float(f[0])
    Y = Y.to(device=device) / float(f[1])
    Z = torch.ones_like(X)
    unproj = torch.stack((X, -Y, -Z), dim=-1)
    unproj /= torch.norm(unproj, dim=-1).unsqueeze(-1)
    return unproj


def coord_from_blender(dtype=torch.float32, device="cpu"):
    return torch.tensor([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=dtype, device=device)


def coord_to_blender(dtype=torch.float32, device="cpu"):
    return torch.tensor([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=dtype, device=device)


def look_at(origin, target, world_up=np.array([0, 1, 0], dtype=np.float32)):
    back = origin - target
    back /= np.linalg.norm(back)
    right = np.cross(world_up, back)
    right /= np.linalg.norm(right)
    up = np.cross(back, right)

    cam_to_world = np.empty((4, 4), dtype=np.float32)
    cam_to_world[:3, 0] = right
    cam_to_world[:3, 1] = up
    cam_to_world[:3, 2] = back
    cam_to_world[:3, 3] = origin
    cam_to_world[3, :] = [0, 0, 0, 1]
    return cam_to_world


def get_cuda(gpu_id):
    return (
        torch.device("cuda:%d" % gpu_id)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )


def masked_sample(masks, num_pix, prop_inside, thresh=0.5):
    num_inside = int(num_pix * prop_inside + 0.5)
    num_outside = num_pix - num_inside
    inside = (masks >= thresh).nonzero(as_tuple=False)
    outside = (masks < thresh).nonzero(as_tuple=False)

    pix_inside = inside[torch.randint(0, inside.shape[0], (num_inside,))]
    pix_outside = outside[torch.randint(0, outside.shape[0], (num_outside,))]
    pix = torch.cat((pix_inside, pix_outside))
    return pix


def bbox_sample(bboxes, num_pix):
    image_ids = torch.randint(0, bboxes.shape[0], (num_pix,))
    pix_bboxes = bboxes[image_ids]
    x = (torch.rand(num_pix) * (pix_bboxes[:, 2] + 1 - pix_bboxes[:, 0]) + pix_bboxes[:, 0]).long()
    y = (torch.rand(num_pix) * (pix_bboxes[:, 3] + 1 - pix_bboxes[:, 1]) + pix_bboxes[:, 1]).long()
    pix = torch.stack((image_ids, y, x), dim=-1)
    return pix


def gen_rays(poses, width, height, focal, z_near, z_far, c=None, ndc=False):
    num_images = poses.shape[0]
    device = poses.device
    cam_unproj_map = (unproj_map(width, height, focal.squeeze(), c=c, device=device).unsqueeze(0).repeat(num_images, 1, 1, 1))
    cam_centers = poses[:, None, None, :3, 3].expand(-1, height, width, -1)
    cam_raydir = torch.matmul(
        poses[:, None, None, :3, :3], cam_unproj_map.unsqueeze(-1)
    )[:, :, :, :, 0]
    if ndc:
        if not (z_near == 0 and z_far == 1):
            warnings.warn("dataset z near and z_far not compatible with NDC, setting them to 0, 1 NOW")
        z_near, z_far = 0.0, 1.0
        cam_centers, cam_raydir = ndc_rays(width, height, focal, 1.0, cam_centers, cam_raydir)

    cam_nears = (torch.tensor(z_near, device=device).view(1, 1, 1, 1).expand(num_images, height, width, -1))
    cam_fars = (torch.tensor(z_far, device=device).view(1, 1, 1, 1).expand(num_images, height, width, -1))
    return torch.cat((cam_centers, cam_raydir, cam_nears, cam_fars), dim=-1)


def trans_t(t):
    return torch.tensor(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, t],
            [0, 0, 0, 1],
        ],
        dtype=torch.float32,
    )


def rot_phi(phi):
    return torch.tensor(
        [
            [1, 0, 0, 0],
            [0, np.cos(phi), -np.sin(phi), 0],
            [0, np.sin(phi), np.cos(phi), 0],
            [0, 0, 0, 1],
        ],
        dtype=torch.float32,
    )


def rot_theta(th):
    return torch.tensor(
        [
            [np.cos(th), 0, -np.sin(th), 0],
            [0, 1, 0, 0],
            [np.sin(th), 0, np.cos(th), 0],
            [0, 0, 0, 1],
        ],
        dtype=torch.float32,
    )


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180.0 * np.pi) @ c2w
    c2w = rot_theta(theta / 180.0 * np.pi) @ c2w
    c2w = (
        torch.tensor(
            [
                [-1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1]
            ],
            dtype=torch.float32,
        )
        @ c2w
    )
    return c2w


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_norm_layer(norm_type="instance", group_norm_groups=32):
    if norm_type == "batch":
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == "instance":
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == "group":
        norm_layer = functools.partial(nn.GroupNorm, group_norm_groups)
    elif norm_type == "none":
        norm_layer = None
    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
    return norm_layer


def make_conv_2d(
    dim_in,
    dim_out,
    padding_type="reflect",
    norm_layer=None,
    activation=None,
    kernel_size=3,
    use_bias=False,
    stride=1,
    no_pad=False,
    zero_init=False,
):
    conv_block = []
    amt = kernel_size // 2
    if stride > 1 and not no_pad:
        raise NotImplementedError("Padding with stride > 1 not supported, use same_pad_conv2d")

    if amt > 0 and not no_pad:
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(amt)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(amt)]
        elif padding_type == "zero":
            conv_block += [nn.ZeroPad2d(amt)]
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)

    conv_block.append(nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, bias=use_bias, stride=stride))
    if zero_init:
        nn.init.zeros_(conv_block[-1].weight)
    if norm_layer is not None:
        conv_block.append(norm_layer(dim_out))

    if activation is not None:
        conv_block.append(activation)
    return nn.Sequential(*conv_block)


def calc_same_pad_conv2d(t_shape, kernel_size=3, stride=1):
    in_height, in_width = t_shape[-2:]
    out_height = math.ceil(in_height / stride)
    out_width = math.ceil(in_width / stride)

    pad_along_height = max((out_height - 1) * stride + kernel_size - in_height, 0)
    pad_along_width = max((out_width - 1) * stride + kernel_size - in_width, 0)
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left
    return pad_left, pad_right, pad_top, pad_bottom


def same_pad_conv2d(t, padding_type="reflect", kernel_size=3, stride=1, layer=None):
    if layer is not None:
        if isinstance(layer, nn.Sequential):
            layer = next(layer.children())
        kernel_size = layer.kernel_size[0]
        stride = layer.stride[0]
    return F.pad(t, calc_same_pad_conv2d(t.shape, kernel_size, stride), mode=padding_type)


def same_unpad_deconv2d(t, kernel_size=3, stride=1, layer=None):
    if layer is not None:
        if isinstance(layer, nn.Sequential):
            layer = next(layer.children())
        kernel_size = layer.kernel_size[0]
        stride = layer.stride[0]
    h_scaled = (t.shape[-2] - 1) * stride
    w_scaled = (t.shape[-1] - 1) * stride
    pad_left, pad_right, pad_top, pad_bottom = calc_same_pad_conv2d((h_scaled, w_scaled), kernel_size, stride)
    if pad_right == 0:
        pad_right = -10000
    if pad_bottom == 0:
        pad_bottom = -10000
    return t[..., pad_top:-pad_bottom, pad_left:-pad_right]


def combine_interleaved(t, inner_dims=(1,), agg_type="average"):
    if len(inner_dims) == 1 and inner_dims[0] == 1:
        return t
    t = t.reshape(-1, *inner_dims, *t.shape[1:])
    if agg_type == "average":
        t = torch.mean(t, dim=1)
    elif agg_type == "max":
        t = torch.max(t, dim=1)[0]
    else:
        raise NotImplementedError("Unsupported combine type " + agg_type)
    return t


def psnr(pred, target):
    mse = ((pred - target) ** 2).mean()
    psnr = -10 * math.log10(mse)
    return psnr


def quat_to_rot(q):
    batch_size, _ = q.shape
    q = F.normalize(q, dim=1)
    R = torch.ones((batch_size, 3, 3), device=q.device)
    qr = q[:, 0]
    qi = q[:, 1]
    qj = q[:, 2]
    qk = q[:, 3]
    R[:, 0, 0] = 1 - 2 * (qj ** 2 + qk ** 2)
    R[:, 0, 1] = 2 * (qj * qi - qk * qr)
    R[:, 0, 2] = 2 * (qi * qk + qr * qj)
    R[:, 1, 0] = 2 * (qj * qi + qk * qr)
    R[:, 1, 1] = 1 - 2 * (qi ** 2 + qk ** 2)
    R[:, 1, 2] = 2 * (qj * qk - qi * qr)
    R[:, 2, 0] = 2 * (qk * qi - qj * qr)
    R[:, 2, 1] = 2 * (qj * qk + qi * qr)
    R[:, 2, 2] = 1 - 2 * (qi ** 2 + qj ** 2)
    return R


def rot_to_quat(R):
    batch_size, _, _ = R.shape
    q = torch.ones((batch_size, 4), device=R.device)

    R00 = R[:, 0, 0]
    R01 = R[:, 0, 1]
    R02 = R[:, 0, 2]
    R10 = R[:, 1, 0]
    R11 = R[:, 1, 1]
    R12 = R[:, 1, 2]
    R20 = R[:, 2, 0]
    R21 = R[:, 2, 1]
    R22 = R[:, 2, 2]

    q[:, 0] = torch.sqrt(1.0 + R00 + R11 + R22) / 2
    q[:, 1] = (R21 - R12) / (4 * q[:, 0])
    q[:, 2] = (R02 - R20) / (4 * q[:, 0])
    q[:, 3] = (R10 - R01) / (4 * q[:, 0])
    return q


def get_module(net):
    if isinstance(net, torch.nn.DataParallel):
        return net.module
    else:
        return net
