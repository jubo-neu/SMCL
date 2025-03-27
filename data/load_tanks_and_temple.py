import numpy as np
import os
import imageio
from PIL import Image

blender2opencv = np.array(
    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])


def get_instrinsic(filepath):
    try:
        intrinsic = np.loadtxt(filepath).astype(np.float32)[:3, :3]
        return intrinsic
    except ValueError:
        pass

    with open(filepath, 'r') as file:
        f, cx, cy, _ = map(float, file.readline().split())
    fy = fx = f

    intrinsic = np.array([[fx, 0., cx],
                          [0., fy, cy],
                          [0., 0, 1]])
    return intrinsic


def load_tanksandtemple_data(basedir, factor=1, split="train", read_offline=True, tgtH=1280, tgtW=2176):
    colordir = os.path.join(basedir, "rgb")
    posedir = os.path.join(basedir, "pose")
    train_image_paths = [f for f in os.listdir(colordir) if os.path.isfile(os.path.join(colordir, f)) and f.startswith("0")]
    test_image_paths = [f for f in os.listdir(colordir) if os.path.isfile(os.path.join(colordir, f)) and f.startswith("1")]

    if split == "train":
        image_paths = train_image_paths
    elif split == "test":
        image_paths = test_image_paths
    else:
        raise ValueError("Unknown split: {}".format(split))

    image_paths = sorted(image_paths, key=lambda x: int(x.split(".")[0].split("_")[-1]))

    images = []
    poses = []
    out_image_paths = []

    intrinsic = get_instrinsic(os.path.join(basedir, "intrinsics.txt"))
    fx, _, cx = intrinsic[0]
    _, fy, cy = intrinsic[1]

    for i, img_path in enumerate(image_paths):
        image_path = os.path.abspath(os.path.join(colordir, img_path))
        out_image_paths.append(image_path)

        if read_offline:
            image = imageio.imread(image_path)
            H, W = image.shape[:2]
            if factor != 1:
                image = Image.fromarray(image).resize((tgtW // factor, tgtH // factor))
            images.append((np.array(image) / 255.).astype(np.float32))
        elif i == 0:
            image = imageio.imread(image_path)
            H, W = image.shape[:2]
            if factor != 1:
                image = Image.fromarray(image).resize((tgtW // factor, tgtH // factor))
            images.append((np.array(image) / 255.).astype(np.float32))

        pose_path = os.path.join(posedir, img_path.replace(".png", ".txt"))
        pose = np.loadtxt(pose_path).astype(np.float32)
        pose = pose @ blender2opencv
        poses.append(pose)

    images = np.stack(images, 0)
    poses = np.stack(poses, 0)

    realH, realW = images.shape[1:3]
    fx = fx * (realW / W)
    fy = fy * (realH / H)

    return images, poses, [realH, realW, fx, fy], out_image_paths
