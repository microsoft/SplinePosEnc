# Spline Positional Encoding

This repository contains the code for our paper published in IJCAI 2021:
- [Spline Positional Encoding for Learning 3D Implicit Signed Distance Fields](https://arxiv.org/abs/2106.01553)
- Peng-Shuai Wang, Yang Liu, Yu-Qi Yang, and Xin Tong

## Install

The code has been tested on Ubuntu 16.04/18.04, please follow the following instructions to install the requirements.

```bash
  conda create --name spe python=3.7
  conda activate spe
  conda install  pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.2 -c pytorch
  pip install -r requirements.txt
```

## Data

For the task of SDF reconstruction from a point cloud, SDF regression and image regression, please download the data from this [link](https://www.dropbox.com/s/b2ow4b5ahsr5wqg/data.zip?dl=0) and then unzip it to the folder `data`.

For the shape space learning, please download the data from the official website of [Dfaust](http://dfaust.is.tue.mpg.de/downloads), and extract the meshes with the code provided by Dfaust to the target folder, denoted as `<dfaust_folder>`.
Then download the training and testing data list file from this [link](https://www.dropbox.com/s/cmd311jfhzjbfk7/dfaust.zip?dl=0) and unzip the list file to the folder `data`.
After these 2 steps, run the following command to generate the data for training and testing: `python scripts/dfaust.py  --root_folder <dfaust_folder>`


## Tasks

### Reconstruct SDFs from a point cloud
- Run the following command: `bash scripts/run_train_sdf.sh`


### Regress images
- Run the following command: `python scripts/run_regress_img.py`
<!-- The figure in our paper is `div2k_002` -->


### Regress SDFs
- Run the following command: `bash scripts/run_regress_sdf.sh`


### Train SDF Space
- Run the following command: `bash scripts/run_shape_space.sh`.
  The training process is relatively slow, we provide the trained weights [here](https://www.dropbox.com/s/keoq8ni752gdbi0/our_all_c33_dfaust_final.pth?dl=0).

- Run the following command to test the trained shape space: `python scripts/run_sdf_space_test.py`
