# SMCL
## [Paper] A Spatial-spectral Multi-view Contrastive Learning Framework for Scene Representation Learning
These are the official code releases.
- Graphical Abstract:
<img src="https://github.com/jubo-neu/SMCL/blob/main/new%20teaser.png?raw=true" alt="Teaser Image">

## Results
- Partial visualizations:

<img height="100" src="https://github.com/jubo-neu/SMCL/blob/main/gif/bench.gif"> </a>
<img height="100" src="https://github.com/jubo-neu/SMCL/blob/main/gif/car.gif"> </a>
<img height="100" src="https://github.com/jubo-neu/SMCL/blob/main/gif/display.gif"> </a> 
<img height="100" src="https://github.com/jubo-neu/SMCL/blob/main/gif/plane.gif"> </a> 
<img height="100" src="https://github.com/jubo-neu/SMCL/blob/main/gif/sofa.gif"> </a> 

<img height="100" src="https://github.com/jubo-neu/SMCL/blob/main/gif/nerfsync_chair.gif"> </a> 
<img height="100" src="https://github.com/jubo-neu/SMCL/blob/main/gif/nerfsync_drums.gif"> </a> 
<img height="100" src="https://github.com/jubo-neu/SMCL/blob/main/gif/nerfsync_hotdog.gif"> </a> 
<img height="100" src="https://github.com/jubo-neu/SMCL/blob/main/gif/nerfsync_lego.gif"> </a> 
<img height="100" src="https://github.com/jubo-neu/SMCL/blob/main/gif/nerfsync_ship.gif"> </a> 

<img height="100" src="https://github.com/jubo-neu/SMCL/blob/main/gif/tt_Truck.gif"> </a> 
<img height="100" src="https://github.com/jubo-neu/SMCL/blob/main/gif/tt_caterpillar.gif"> </a> 
<img height="100" src="https://github.com/jubo-neu/SMCL/blob/main/gif/tt_family.gif"> </a> 

## Highlights
- Combines FFT with contrastive learning to enhance spectral feature extraction capabilities.
- R-MSA mechanism improves feature updating and 3D space information processing.
- Complex scene representation by integrating spatial and spectral domain features.
- Enhances application scope by integrating point clouds and images into a unified framework.

## TODO
- [x] Training code and datasets.
- [x] Test code and models.
- [x] Scripts and configuration files.

### News
- 2025-10-27: Accepted by Information Fusion.
- 2025-5-9: Preprint is available on Elsevier's SSRN eLibrary.
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
1. If you want to train our model on synthetic datasets, you need to create a Mir-180 dataset of the desired category according to our paper's method.

- Ensuring that you have the following files:
```bash
SMCL
|-- Mir-180_Dataset
    |-- display
        |-- For_Synthesis
        |-- test
        |-- train
        |-- val
    |-- other category
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

2. If you want to train our model on real-world datasets, you need to create a sparse point cloud and first perform point feature extraction operations.

- Ensuring that you have the following files:
```bash
SMCL
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
We provide some categories as test targets. To start:
1. Download scene representation module checkpoints [here](https://onedrive.live.com/?id=6DE4E04ABF455D96%21s4c703031efcb49d6ba06e31fd7100a55&cid=6DE4E04ABF455D96).
2. Download view synthesizer checkpoints [here](https://onedrive.live.com/?id=6DE4E04ABF455D96%21s6859c4a61d6449c2bebf38ac6772ae3e&cid=6DE4E04ABF455D96).

- Make sure you have the following models:
```bash
SMCL
|-- cnn_ffc_weights
    |-- encoder_display.pt
    |-- encoder_caterpillar.pth
    |-- encoder_hotdog.pth
    |-- ...
|-- checkpoints
    |-- Caterpillar
        |-- model.pth
    |-- display_exp
        |-- model.pth
    |-- hotdog
        |-- model.pth
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
# Avaliable online
https://doi.org/10.1016/j.inffus.2025.103889
```

## Contact us
If you have any questions, please contact us [jbchen@stumail.neu.edu.cn](jbchen@stumail.neu.edu.cn).
