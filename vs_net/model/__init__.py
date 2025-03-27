from .models import SMCLNet


def make_model(conf, *args, **kwargs):
    model_type = conf.get_string("type", "SMCL")
    if model_type == "SMCL":
        net = SMCLNet(conf, *args, **kwargs)
    else:
        raise NotImplementedError("Unsupported model type", model_type)
    return net
