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
  --filename_gt data/shapes/{name}_gt_mesh_normalized.ply \
  --filename_pred logs/sdf/{name}/{alias}_c003/mesh/{alias}_c003_{epoch:04}.ply \
  --metric chamfer \
  --point_num 320000 \
  --filename_out results/{alias}.csv

 python calc_error.py \
  --filename_gt data/shapes/{name}_gt_mesh_normalized.ply \
  --filename_pred logs/sdf/{name}/{alias}_c009/mesh/{alias}_c009_{epoch:04}.ply \
  --metric chamfer \
  --point_num 320000 \
  --filename_out results/{alias}.csv

 python calc_error.py \
  --filename_gt data/shapes/{name}_gt_mesh_normalized.ply \
  --filename_pred logs/sdf/{name}/{alias}_c033/mesh/{alias}_c033_{epoch:04}.ply \
  --metric chamfer \
  --point_num 320000 \
  --filename_out results/{alias}.csv

 python calc_error.py \
  --filename_gt data/shapes/{name}_gt_mesh_normalized.ply \
  --filename_pred logs/sdf/{name}/{alias}_c129/mesh/{alias}_c129_{epoch:04}.ply \
  --metric chamfer \
  --point_num 320000 \
  --filename_out results/{alias}.csv

 python calc_error.py \
  --filename_gt data/shapes/{name}_gt_mesh_normalized.ply \
  --filename_pred logs/sdf/{name}/{alias}_c257/mesh/{alias}_c257_{epoch:04}.ply \
  --metric chamfer \
  --point_num 320000 \
  --filename_out results/{alias}.csv

 # Siren
 python calc_error.py \
  --filename_gt data/shapes/{name}_gt_mesh_normalized.ply \
  --filename_pred logs/sdf/{name}/{alias}_siren/mesh/{alias}_siren_{epoch:04}.ply \
  --metric chamfer \
  --point_num 320000 \
  --filename_out results/{alias}.csv

 # Igr
 python calc_error.py \
  --filename_gt data/shapes/{name}_gt_mesh_normalized.ply \
  --filename_pred logs/sdf/{name}/{alias}_softplus/mesh/{alias}_softplus_{epoch:04}.ply \
  --metric chamfer \
  --point_num 320000 \
  --filename_out results/{alias}.csv

 # fourier
 python calc_error.py \
  --filename_gt data/shapes/{name}_gt_mesh_normalized.ply \
  --filename_pred logs/sdf/{name}/{alias}_fourier_f128_s4/mesh/{alias}_fourier_{epoch:04}.ply \
  --metric chamfer \
  --point_num 320000 \
  --filename_out results/{alias}.csv
'''

print(cmds + '\n')
os.system(cmds)
