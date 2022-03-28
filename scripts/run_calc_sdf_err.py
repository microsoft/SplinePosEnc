import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--alias', type=str, required=True)
parser.add_argument('--name', type=str, required=True)
parser.add_argument('--epoch', type=int, required=True)
args = parser.parse_args()

alias = args.alias
name = args.name
epoch = args.epoch


cmds = f'''
 python calc_error.py \
  --filename_gt data/shapes/{name}_gt_mesh_normalized.sdf.npy \
  --filename_pred logs/sdf/{name}/{alias}_c003/mesh/{alias}_c003_{epoch:04}_sdf.npy \
  --metric sdf \
  --filename_out results/{alias}_sdf.csv 2>&1

 python calc_error.py \
  --filename_gt data/shapes/{name}_gt_mesh_normalized.sdf.npy \
  --filename_pred logs/sdf/{name}/{alias}_c009/mesh/{alias}_c009_{epoch:04}_sdf.npy \
  --metric sdf \
  --filename_out results/{alias}_sdf.csv 2>&1

 python calc_error.py \
  --filename_gt data/shapes/{name}_gt_mesh_normalized.sdf.npy \
  --filename_pred logs/sdf/{name}/{alias}_c033/mesh/{alias}_c033_{epoch:04}_sdf.npy \
  --metric sdf \
  --filename_out results/{alias}_sdf.csv 2>&1

 python calc_error.py \
  --filename_gt data/shapes/{name}_gt_mesh_normalized.sdf.npy \
  --filename_pred logs/sdf/{name}/{alias}_c129/mesh/{alias}_c129_{epoch:04}_sdf.npy \
  --metric sdf \
  --filename_out results/{alias}_sdf.csv 2>&1

 python calc_error.py \
  --filename_gt data/shapes/{name}_gt_mesh_normalized.sdf.npy \
  --filename_pred logs/sdf/{name}/{alias}_c257/mesh/{alias}_c257_{epoch:04}_sdf.npy \
  --metric sdf \
  --filename_out results/{alias}_sdf.csv 2>&1

 # siren
 python calc_error.py \
  --filename_gt data/shapes/{name}_gt_mesh_normalized.sdf.npy \
  --filename_pred logs/sdf/{name}/{alias}_siren/mesh/{alias}_siren_{epoch:04}_sdf.npy \
  --metric sdf \
  --filename_out results/{alias}_sdf.csv 2>&1

 # solftplus
 python calc_error.py \
  --filename_gt data/shapes/{name}_gt_mesh_normalized.sdf.npy \
  --filename_pred logs/sdf/{name}/{alias}_softplus/mesh/{alias}_softplus_{epoch:04}_sdf.npy \
  --metric sdf \
  --filename_out results/{alias}_sdf.csv 2>&1

 # fourier
 python calc_error.py \
  --filename_gt data/shapes/{name}_gt_mesh_normalized.sdf.npy \
  --filename_pred logs/sdf/{name}/{alias}_fourier_f128_s4/mesh/{alias}_fourier_{epoch:04}_sdf.npy \
  --metric sdf \
  --filename_out results/{alias}_sdf.csv 2>&1
'''

 # print(cmds)
os.system(cmds)
