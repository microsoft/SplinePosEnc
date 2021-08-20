import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--alias', type=str, required=True)
parser.add_argument('--name', type=str, required=True)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--pc_num', type=int, default=200000)
parser.add_argument('--dry-run', action='store_true')
args = parser.parse_args()

alias = args.alias
name = args.name
gpu = args.gpu
pc_num = args.pc_num

cmds = f'''
 # optpos
 python regress_sdf.py  \
  SOLVER.gpu {gpu},  \
  SOLVER.logdir logs/sdf/{name}/{alias}_c003  \
  SOLVER.alias {alias}_c003  \
  SOLVER.num_epochs 400  \
  SOLVER.test_every_epoch 10  \
  SOLVER.upsample_size 9  \
  MODEL.name optpos  \
  MODEL.projs 16  \
  MODEL.activation softplus  \
  MODEL.code_num 3  \
  MODEL.code_channel 64  \
  DATA.train.filename data/shapes/{name}_gt_mesh_normalized.sdf.npy  \
  DATA.train.pc_num {pc_num}  \
  DATA.train.normalize False 

 python regress_sdf.py  \
  SOLVER.gpu {gpu},  \
  SOLVER.logdir logs/sdf/{name}/{alias}_c009  \
  SOLVER.alias {alias}_c009  \
  SOLVER.num_epochs 400  \
  SOLVER.test_every_epoch 10  \
  SOLVER.upsample_size 33  \
  MODEL.name optpos  \
  MODEL.projs 16  \
  MODEL.activation softplus  \
  MODEL.code_num 9  \
  MODEL.code_channel 64  \
  DATA.train.filename data/shapes/{name}_gt_mesh_normalized.sdf.npy  \
  DATA.train.pc_num {pc_num}  \
  DATA.train.normalize False  \
  SOLVER.ckpt logs/sdf/{name}/{alias}_c003/checkpoints/model_final_upsample_009.pth

 python regress_sdf.py  \
  SOLVER.gpu {gpu},  \
  SOLVER.logdir logs/sdf/{name}/{alias}_c033  \
  SOLVER.alias {alias}_c033  \
  SOLVER.num_epochs 400  \
  SOLVER.test_every_epoch 10  \
  SOLVER.upsample_size 129  \
  MODEL.name optpos  \
  MODEL.projs 16  \
  MODEL.activation softplus  \
  MODEL.code_num 33  \
  MODEL.code_channel 64  \
  DATA.train.filename data/shapes/{name}_gt_mesh_normalized.sdf.npy  \
  DATA.train.pc_num {pc_num}  \
  DATA.train.normalize False  \
  SOLVER.ckpt logs/sdf/{name}/{alias}_c009/checkpoints/model_final_upsample_033.pth

 python regress_sdf.py  \
  SOLVER.gpu {gpu},  \
  SOLVER.logdir logs/sdf/{name}/{alias}_c129  \
  SOLVER.alias {alias}_c129  \
  SOLVER.num_epochs 400  \
  SOLVER.test_every_epoch 10  \
  SOLVER.upsample_size 257  \
  MODEL.name optpos  \
  MODEL.projs 16  \
  MODEL.activation softplus  \
  MODEL.code_num 129  \
  MODEL.code_channel 64  \
  DATA.train.filename data/shapes/{name}_gt_mesh_normalized.sdf.npy  \
  DATA.train.pc_num {pc_num}  \
  DATA.train.normalize False  \
  SOLVER.ckpt logs/sdf/{name}/{alias}_c033/checkpoints/model_final_upsample_129.pth


 # softplus
 python regress_sdf.py  \
  SOLVER.gpu {gpu},  \
  SOLVER.logdir logs/sdf/{name}/{alias}_softplus  \
  SOLVER.alias {alias}_softplus  \
  SOLVER.num_epochs 400  \
  SOLVER.test_every_epoch 10  \
  MODEL.name mlp  \
  MODEL.activation softplus  \
  DATA.train.filename data/shapes/{name}_gt_mesh_normalized.sdf.npy  \
  DATA.train.pc_num {pc_num}  \
  DATA.train.normalize False 


 # siren
 python regress_sdf.py  \
  SOLVER.gpu {gpu},  \
  SOLVER.logdir logs/sdf/{name}/{alias}_siren  \
  SOLVER.alias {alias}_siren  \
  SOLVER.num_epochs 400  \
  SOLVER.test_every_epoch 10  \
  MODEL.name siren  \
  DATA.train.filename data/shapes/{name}_gt_mesh_normalized.sdf.npy  \
  DATA.train.pc_num {pc_num}  \
  DATA.train.normalize False 


 # fpe
 python regress_sdf.py  \
  SOLVER.gpu {gpu},  \
  SOLVER.logdir logs/sdf/{name}/{alias}_fourier_f128_s4  \
  SOLVER.alias 0111_gargoyle_fourier  \
  SOLVER.num_epochs 400  \
  SOLVER.test_every_epoch 10  \
  MODEL.name randfourier  \
  MODEL.activation softplus  \
  MODEL.scale 4 \
  MODEL.num_frequencies 128 \
  DATA.train.filename data/shapes/{name}_gt_mesh_normalized.sdf.npy  \
  DATA.train.pc_num {pc_num}  \
  DATA.train.normalize False 
'''

print(cmds)
if not args.dry_run:
  os.system(cmds)
