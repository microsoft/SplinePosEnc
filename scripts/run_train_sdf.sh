#!/bin/bash

python train_sdf.py  \
  SOLVER.gpu 0,  \
  SOLVER.logdir logs/sdf/sphere/0406_sphere_c003 \
  SOLVER.alias 0406_sphere_c003  \
  SOLVER.num_epochs 2000  \
  SOLVER.upsample_size 9 \
  MODEL.name optpos  \
  MODEL.activation softplus  \
  MODEL.in_features 3  \
  MODEL.out_features 1  \
  MODEL.code_num 3  \
  MODEL.code_channel 64 \
  DATA.train.filename data/shapes/sphere.xyz \
  DATA.train.pc_num 10000 \
  DATA.train.scale 0.7 \
  LOSS.sphere_loss True


python scripts/run_train_sdf.py --alias 0406_armadillo --name armadillo --pc_num 15000 --gpu 0
python scripts/run_calc_chamfer.py --alias 0406_armadillo --name armadillo --epoch 8000
python scripts/run_calc_sdf_err.py --alias 0406_armadillo --name armadillo --epoch 8000

python scripts/run_train_sdf.py --alias 0406_bimba --name bimba --pc_num 20000 --gpu 0
python scripts/run_calc_chamfer.py --alias 0406_bimba --name bimba --epoch 8000
python scripts/run_calc_sdf_err.py --alias 0406_bimba --name bimba --epoch 8000

python scripts/run_train_sdf.py --alias 0406_bunny --name bunny --pc_num 20000 --gpu 0
python scripts/run_calc_chamfer.py --alias 0406_bunny --name bunny --epoch 8000
python scripts/run_calc_sdf_err.py --alias 0406_bunny --name bunny --epoch 8000

python scripts/run_train_sdf.py --alias 0406_dragon --name dragon  --pc_num 20000 --gpu 0
python scripts/run_calc_chamfer.py --alias 0406_dragon --name dragon --epoch 8000
python scripts/run_calc_sdf_err.py --alias 0406_dragon --name dragon --epoch 8000

python scripts/run_train_sdf.py --alias 0406_dfaust_m --name dfaust_m  --pc_num 20000 --gpu 0
python scripts/run_calc_chamfer.py --alias 0406_dfaust_m --name dfaust_m --epoch 8000
python scripts/run_calc_sdf_err.py --alias 0406_dfaust_m --name dfaust_m --epoch 8000

python scripts/run_train_sdf.py --alias 0406_dfaust_m --name dfaust_f  --pc_num 20000 --gpu 0
python scripts/run_calc_chamfer.py --alias 0406_dfaust_m --name dfaust_m --epoch 8000
python scripts/run_calc_sdf_err.py --alias 0406_dfaust_m --name dfaust_m --epoch 8000

python scripts/run_train_sdf.py --alias 0406_fandisk --name fandisk  --pc_num 20000 --gpu 0
python scripts/run_calc_chamfer.py --alias 0406_fandisk --name fandisk --epoch 8000
python scripts/run_calc_sdf_err.py --alias 0406_fandisk --name fandisk --epoch 8000

python scripts/run_train_sdf.py --alias 0406_gargoyle --name gargoyle  --pc_num 20000 --gpu 0
python scripts/run_calc_chamfer.py --alias 0406_gargoyle --name gargoyle --epoch 8000
python scripts/run_calc_sdf_err.py --alias 0406_gargoyle --name gargoyle --epoch 8000
