import torch.nn as nn
from .model import SMCL_point
from .lpips import LPNet


class BasicLoss(nn.Module):
    def __init__(self, losses_and_weights):
        super(BasicLoss, self).__init__()
        self.losses_and_weights = losses_and_weights

    def forward(self, pred, target):
        loss = 0
        for name_and_weight, loss_func in self.losses_and_weights.items():
            name, weight = name_and_weight.split('/')
            cur_loss = loss_func(pred, target)
            loss += float(weight) * cur_loss
        return loss


def get_model(args, device='cuda'):
    return SMCL_point(args, device=device)


def get_loss(args, bias=1.0):
    losses = nn.ModuleDict()
    for loss_name, weight in args.items():
        if weight > 0:
            if loss_name == "mse":
                losses[loss_name + "/" + str(format(weight, '.0e'))] = nn.MSELoss()
            elif loss_name == "l1":
                losses[loss_name + "/" + str(format(weight, '.0e'))] = nn.L1Loss()
            elif loss_name == "lpips":
                lpips = LPNet()
                lpips.eval()
                losses[loss_name + "/" + str(format(weight, '.0e'))] = lpips
            elif loss_name == "lpips_alex":
                lpips = lpips.LPIPS()
                lpips.eval()
                losses[loss_name + "/" + str(format(weight, '.0e'))] = lpips
            else:
                raise NotImplementedError('loss [{:s}] is not supported'.format(loss_name))
    return BasicLoss(losses)
