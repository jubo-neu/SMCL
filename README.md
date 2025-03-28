# SMCL
## [Paper] A Spatial-spectral Multi-view Contrastive Learning Framework for Scene Representation Learning
These are the official code and dataset releases, intended to assist with peer review.

![Teaser](https://github.com/jubo-neu/SMCL/blob/main/teaser.png)

## Highlights
- Combines FFT with contrastive learning to enhance spectral feature extraction capabilities.
- R-MSA mechanism improves feature updating and 3D space information processing.
- Complex scene representation by integrating spatial and spectral domain features.
- Enhances application scope by integrating point clouds and images into a unified framework.

## TODO
- [x] Training code and training datasets.
- [x] Test code and pretrained models.
- [x] Scripts, tools and configuration files.

### News
- 2025-3-30: All files have been released.

## Preparation for training
1. To start, we prefer creating the environment using conda:
```
conda env create -f SMCL_env.yml
conda activate SMCL
```
2. Getting the data
- Download the complete Synthetic Dataset [NMR](https://s3.eu-central-1.amazonaws.com/avg-projects/differentiable_volumetric_rendering/data/NMR_Dataset.zip), [SRN](https://drive.google.com/drive/folders/1PsT3uKwqHHD2bEEHkIXB99AlIjtmrEiR).
- Download the complete Real-world Dataset [NeRF Sync](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1), [TanksAndTemples](https://www.tanksandtemples.org/).

## Training
1. If you want to train our model on synthetic datasets, you need to create a Mir-180 dataset of the desired category according to our paper's method. We provide various small scripts in the `tools` folder that facilitate the implementation of our code. For creating datasets, `datamaker.py`, `namereader.py` and `txt.py` can effectively help you.

- (Optional) We provide a mini [Mir-180_Dataset](https://onedrive.live.com/?cid=6DE4E04ABF455D96&id=6DE4E04ABF455D96%21sbc6b662b9ae6481a92e9c720b29c8f3e&parId=root&o=OneUp) as an example. Ensuring that you have the following files：
```bash
SMCL
|-- Mir-180_Dataset
    |-- display
        |-- For_Synthesis
        |-- test
        |-- train
        |-- val
    |-- ...
        |-- For_Synthesis
        |-- test
        |-- train
        |-- val
    |-- ...
```

- After making your own dataset, then execute the command:
```bash
python .../train/train.py -n your category
                          -c .../conf/Mir_180.conf
                          -D .../Mir-180_Dataset/your category/For_Synthesis
```

2. If you want to train our model on real-world datasets, you need to create a sparse point cloud and first perform point feature extraction operations. These data processing scripts also in the `tools` folder, including:
```bash
-- pointdownsampling.py # Downsampling operation of point cloud
-- pointvisual.py # Visualization operation of point cloud
-- ply.py # Convert point cloud from .ply to .pcd format
-- pcd.py # Convert point cloud from .pcd to .pth format
```

- (Optional) We also provide sparse point clouds for `Hotdog` and `Caterpillar` in the [point cloud](https://onedrive.live.com/?id=6DE4E04ABF455D96%21s8cbb5ec9cd734e9783be74932e017cf8&cid=6DE4E04ABF455D96) as examples. And you can download a mini [Real-world_Dataset](https://onedrive.live.com/?cid=6DE4E04ABF455D96&id=6DE4E04ABF455D96%21s0b74b63888fb44189fdf0fbd1cb0c4ca&parId=root&o=OneUp). Ensuring that you have the following files：
```bash
SMCL
|-- point_cloud
    |-- caterpillar 10000.pcd
    |-- hotdog 10000.pcd
    |-- ...
|-- Real-world_Dataset
    |-- Caterpillar
        |-- pose
        |-- rgb
        |-- bbox.txt
        |-- intrinsics.txt
        |-- test_traj.txt
    |-- hotdog
        |-- test
        |-- train
        |-- val
        |-- .DS_Store
        |-- transforms_test.json
        |-- transforms_train.json
        |-- transforms_val.json
    |-- ...
```

- After making your own dataset, then execute the command:
```bash
python train.py --opt conf/your category/your category.yml
```

## Preparation for test
We provide some categories as test targets. To strat:
1. Download scene representation module checkpoints [here](https://onedrive.live.com/?id=6DE4E04ABF455D96%21s4c703031efcb49d6ba06e31fd7100a55&cid=6DE4E04ABF455D96).
2. Download view synthesizer checkpoints [here](https://onedrive.live.com/?id=6DE4E04ABF455D96%21s6859c4a61d6449c2bebf38ac6772ae3e&cid=6DE4E04ABF455D96).

- Make sure you have the following models:
```bash
SMCL
|-- cnn_ffc_weights
    |-- cnn_ffc_display_0.05.pt
    |-- encoder_caterpillar.pth
    |-- encoder_hotdog.pth
    |-- ...
|-- checkpoints
    |-- Caterpillar
        |-- model_250000.pth
    |-- display_exp
        |-- SMCL_latest
    |-- hotdog
        |-- model_250000.pth
    |-- ...
```

3. We provide these mini [Mir-180_Dataset](https://onedrive.live.com/?cid=6DE4E04ABF455D96&id=6DE4E04ABF455D96%21sbc6b662b9ae6481a92e9c720b29c8f3e&parId=root&o=OneUp), [Real-world_Dataset](https://onedrive.live.com/?cid=6DE4E04ABF455D96&id=6DE4E04ABF455D96%21s0b74b63888fb44189fdf0fbd1cb0c4ca&parId=root&o=OneUp) and [point cloud](https://onedrive.live.com/?id=6DE4E04ABF455D96%21s8cbb5ec9cd734e9783be74932e017cf8&cid=6DE4E04ABF455D96) as examples.

- Make sure you have the following files:
```bash
SMCL
|-- Mir-180_Dataset
    |-- display
    |-- ...
|-- Real-world_Dataset
    |-- Caterpillar
    |-- hotdog
    |-- ...
|-- point_cloud
    |-- caterpillar 10000.pcd
    |-- hotdog 10000.pcd
    |-- ...
```

## Test
1. If you want to test the NVS performance of our complete model on synthetic datasets, please execute the command:
```bash
python .../eval.py -D .../Mir-180_Dataset/your category/For_Synthesis
                   -n your category
                   -L .../viewlist/your category.txt
                   --multicat
                   -O eval_out/your category
```
- This will allow you to use the provided checkpoint to render visualization results and obtain PSNR and SSIM. 

```bash
python .../calc_metrics.py -D .../Mir-180_Dataset/your category/For_Synthesis
                           --multicat
                           -O .../eval_out/your category
```
- This will give you the final three metrics.

2. If you want to test the NVS performance of our complete model on real-world datasets, please execute the command:
```bash
python test.py --opt conf/tanks_and_temple/your category.yml
```

## Citation
If you find this repository useful in your project, please cite the following work:
```bash
Not available at the moment.
```

## Contact us
If you have any questions, please contact us [jbchen@stumail.neu.edu.cn](jbchen@stumail.neu.edu.cn).
