#!/bin/bash

python train_sdf_space.py  \
  MODEL.name optpos  \
  MODEL.activation softplus  \
  MODEL.code_num 3 \
  MODEL.shape_num 6258  \
  MODEL.code_channel 64  \
  MODEL.num_hidden_layers 3 \
  DATA.train.batch_size 8  \
  DATA.train.filename data/dfaust/dfaust_train_all.txt  \
  DATA.train.root_folder data/dfaust/points \
  DATA.train.pc_num 10000  \
  DATA.train.in_memory False \
  SOLVER.gpu 0,  \
  SOLVER.test_every_epoch 10 \
  SOLVER.alias 0406_our_all_c3 \
  SOLVER.logdir logs/dfaust/0406_our_all_c3  \
  SOLVER.num_epochs 1000  \
  SOLVER.run train \
  SOLVER.upsample_size 9 \
  SOLVER.sphere_init logs/sdf/sphere/0406_sphere_c003/checkpoints/model_final.pth


python train_sdf_space.py  \
  MODEL.name optpos  \
  MODEL.activation softplus  \
  MODEL.code_num 9  \
  MODEL.shape_num 6258  \
  MODEL.code_channel 64  \
  MODEL.num_hidden_layers 3 \
  DATA.train.batch_size 8   \
  DATA.train.filename  data/dfaust/dfaust_train_all.txt   \
  DATA.train.root_folder data/dfaust/points \
  DATA.train.pc_num 10000   \
  DATA.train.in_memory False  \
  SOLVER.gpu 0,  \
  SOLVER.test_every_epoch 10   \
  SOLVER.alias 0406_our_all_c9   \
  SOLVER.logdir logs/dfaust/0406_our_all_c9   \
  SOLVER.num_epochs 1000   \
  SOLVER.run train  \
  SOLVER.upsample_size 33 \
  SOLVER.ckpt logs/dfaust/0406_our_all_c3/checkpoints/model_final_upsample_009.pth


python train_sdf_space.py  \
  MODEL.name optpos  \
  MODEL.activation softplus  \
  MODEL.code_num 33  \
  MODEL.shape_num 6258  \
  MODEL.code_channel 64  \
  MODEL.num_hidden_layers 3 \
  DATA.train.batch_size 8   \
  DATA.train.filename  data/dfaust/dfaust_train_all.txt   \
  DATA.train.root_folder data/dfaust/points \
  DATA.train.pc_num 10000   \
  DATA.train.in_memory False  \
  SOLVER.gpu 0,  \
  SOLVER.test_every_epoch 10   \
  SOLVER.alias 0406_our_all_c33   \
  SOLVER.logdir logs/dfaust/0406_our_all_c33   \
  SOLVER.num_epochs 1000   \
  SOLVER.run train  \
  SOLVER.ckpt logs/dfaust/0406_our_all_c9/checkpoints/model_final_upsample_033.pth

