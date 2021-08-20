import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--alias', type=str, required=True)
parser.add_argument('--name', type=str, required=True)
parser.add_argument('--pc_num', type=str, required=True)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--dry-run', action='store_true')
args = parser.parse_args()

alias = args.alias
name = args.name
gpu = args.gpu
pc_num = args.pc_num

cmds = f'''
 # optpos
 python train_sdf.py  \
  SOLVER.gpu {gpu},  \
  SOLVER.logdir logs/sdf/{name}/{alias}_c003  \
  SOLVER.alias {alias}_c003  \
  SOLVER.num_epochs 8000  \
  SOLVER.upsample_size 9  \
  MODEL.name optpos  \
  MODEL.activation softplus  \
  MODEL.code_num 3  \
  MODEL.code_channel 64  \
  DATA.train.filename data/shapes/{name}_normalized.xyz  \
  DATA.train.pc_num {pc_num}  \
  DATA.train.normalize False  \
  SOLVER.ckpt logs/sdf/sphere/0406_sphere_c003/checkpoints/model_final.pth

 python train_sdf.py  \
  SOLVER.gpu {gpu},  \
  SOLVER.logdir logs/sdf/{name}/{alias}_c009  \
  SOLVER.alias {alias}_c009  \
  SOLVER.num_epochs 8000  \
  SOLVER.upsample_size 33  \
  MODEL.name optpos  \
  MODEL.activation softplus  \
  MODEL.code_num 9  \
  MODEL.code_channel 64  \
  DATA.train.filename data/shapes/{name}_normalized.xyz  \
  DATA.train.pc_num {pc_num}  \
  DATA.train.normalize False  \
  SOLVER.ckpt logs/sdf/{name}/{alias}_c003/checkpoints/model_final_upsample_009.pth

 python train_sdf.py  \
  SOLVER.gpu {gpu},  \
  SOLVER.logdir logs/sdf/{name}/{alias}_c033  \
  SOLVER.alias {alias}_c033  \
  SOLVER.num_epochs 8000  \
  SOLVER.upsample_size 129  \
  MODEL.name optpos  \
  MODEL.activation softplus  \
  MODEL.code_num 33  \
  MODEL.code_channel 64  \
  DATA.train.filename data/shapes/{name}_normalized.xyz  \
  DATA.train.pc_num {pc_num}  \
  DATA.train.normalize False  \
  SOLVER.ckpt logs/sdf/{name}/{alias}_c009/checkpoints/model_final_upsample_033.pth

 python train_sdf.py  \
  SOLVER.gpu {gpu},  \
  SOLVER.logdir logs/sdf/{name}/{alias}_c129  \
  SOLVER.alias {alias}_c129  \
  SOLVER.num_epochs 8000  \
  SOLVER.upsample_size 257  \
  MODEL.name optpos  \
  MODEL.activation softplus  \
  MODEL.code_num 129  \
  MODEL.code_channel 64  \
  DATA.train.filename data/shapes/{name}_normalized.xyz  \
  DATA.train.pc_num {pc_num}  \
  DATA.train.normalize False  \
  SOLVER.ckpt logs/sdf/{name}/{alias}_c033/checkpoints/model_final_upsample_129.pth


 python train_sdf.py  \
  SOLVER.gpu {gpu},  \
  SOLVER.logdir logs/sdf/{name}/{alias}_c257  \
  SOLVER.alias {alias}_c257  \
  SOLVER.num_epochs 8000  \
  SOLVER.upsample_size 513  \
  MODEL.name optpos  \
  MODEL.activation softplus  \
  MODEL.code_num 257  \
  MODEL.code_channel 64  \
  DATA.train.filename data/shapes/{name}_normalized.xyz  \
  DATA.train.pc_num {pc_num}  \
  DATA.train.normalize False  \
  SOLVER.ckpt logs/sdf/{name}/{alias}_c129/checkpoints/model_final_upsample_257.pth


 # softplus
 python train_sdf.py  \
  SOLVER.gpu {gpu},  \
  SOLVER.logdir logs/sdf/{name}/{alias}_softplus  \
  SOLVER.alias {alias}_softplus  \
  SOLVER.num_epochs 8000  \
  MODEL.name mlp  \
  MODEL.hidden_features 256  \
  MODEL.activation softplus  \
  DATA.train.filename data/shapes/{name}_normalized.xyz  \
  DATA.train.pc_num {pc_num}  \
  DATA.train.normalize False


 # relu
 python train_sdf.py  \
  SOLVER.gpu {gpu},  \
  SOLVER.logdir logs/sdf/{name}/{alias}_relu  \
  SOLVER.alias {alias}_relu  \
  SOLVER.num_epochs 8000  \
  MODEL.name mlp  \
  MODEL.hidden_features 256  \
  MODEL.activation relu  \
  DATA.train.filename data/shapes/{name}_normalized.xyz  \
  DATA.train.pc_num {pc_num}  \
  DATA.train.normalize False 


 # siren
 python train_sdf.py  \
  SOLVER.gpu {gpu},  \
  SOLVER.logdir logs/sdf/{name}/{alias}_siren  \
  SOLVER.alias {alias}_siren  \
  SOLVER.num_epochs 8000  \
  MODEL.name siren  \
  MODEL.hidden_features 256  \
  DATA.train.filename data/shapes/{name}_normalized.xyz  \
  DATA.train.pc_num {pc_num}  \
  DATA.train.normalize False 


 # nerf
 python train_sdf.py  \
  SOLVER.gpu {gpu},  \
  SOLVER.logdir logs/sdf/{name}/{alias}_nerf  \
  SOLVER.alias {alias}_nerf  \
  SOLVER.num_epochs 8000  \
  MODEL.name nerf  \
  MODEL.activation softplus  \
  MODEL.hidden_features 256  \
  DATA.train.filename data/shapes/{name}_normalized.xyz  \
  DATA.train.pc_num {pc_num}  \
  DATA.train.normalize False 


 # fourier
 python train_sdf.py  \
  SOLVER.gpu {gpu},  \
  SOLVER.logdir logs/sdf/{name}/{alias}_fourier_f128_s4  \
  SOLVER.alias {alias}_fourier  \
  SOLVER.num_epochs 8000  \
  MODEL.name randfourier  \
  MODEL.activation softplus  \
  MODEL.hidden_features 256  \
  MODEL.scale 4  \
  MODEL.num_frequencies 128  \
  DATA.train.filename data/shapes/{name}_normalized.xyz  \
  DATA.train.pc_num {pc_num}  \
  DATA.train.normalize False 
'''

print(cmds)
if not args.dry_run:
  os.system(cmds)
