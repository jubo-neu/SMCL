from .MultiObjectDataset import MultiObjectDataset
from .DVRDataset import DVRDataset
from .SRNDataset import SRNDataset

from .data_util import ColorJitterDataset


def get_split_dataset(dataset_type, datadir, want_split="all", training=True, **kwargs):
    dset_class, train_aug = None, None
    flags, train_aug_flags = {}, {}

    if dataset_type == "srn":
        dset_class = SRNDataset
    elif dataset_type == "multi_obj":
        dset_class = MultiObjectDataset
    elif dataset_type.startswith("dvr"):
        dset_class = DVRDataset
        if dataset_type == "dvr_gen":
            flags["list_prefix"] = "gen_"
        elif dataset_type == "dvr_dtu":
            flags["list_prefix"] = "new_"
            if training:
                flags["max_imgs"] = 49
            flags["sub_format"] = "dtu"
            flags["scale_focal"] = False
            flags["z_near"] = 0.1
            flags["z_far"] = 5.0
            train_aug = ColorJitterDataset
            train_aug_flags = {"extra_inherit_attrs": ["sub_format"]}
    else:
        raise NotImplementedError("Unsupported dataset type", dataset_type)

    want_train = want_split != "val" and want_split != "test"
    want_val = want_split != "train" and want_split != "test"
    want_test = want_split != "train" and want_split != "val"

    if want_train:
        train_set = dset_class(datadir, stage="train", **flags, **kwargs)
        if train_aug is not None:
            train_set = train_aug(train_set, **train_aug_flags)

    if want_val:
        val_set = dset_class(datadir, stage="val", **flags, **kwargs)

    if want_test:
        test_set = dset_class(datadir, stage="test", **flags, **kwargs)

    if want_split == "train":
        return train_set
    elif want_split == "val":
        return val_set
    elif want_split == "test":
        return test_set
    return train_set, val_set, test_set
