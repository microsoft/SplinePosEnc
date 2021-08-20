import os
import torch
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str,
                    default='logs/dfaust/0406_our_all_c33/checkpoints/model_final.pth')
parser.add_argument('--filelist', type=str,
                    default='data/dfaust/dfaust_test_small.txt')
parser.add_argument('--logdir', type=str,
                    default='logs/dfaust/0406_our_all_c33_test')
parser.add_argument('--shape_num', type=int, default=6258)

args = parser.parse_args()
ckpt = args.ckpt
logdir = args.logdir
if not os.path.exists(logdir):
  os.makedirs(logdir)

trained_dict = torch.load(ckpt)
shape_code = trained_dict['pos_enc.shape_code']
shape_code = shape_code.view(shape_code.shape[0], args.shape_num, -1).mean(1)
trained_dict['pos_enc.shape_code'] = shape_code
ckpt = os.path.join(args.logdir, os.path.basename(ckpt))
torch.save(trained_dict, ckpt)

with open(args.filelist, 'r') as fid:
  lines = fid.readlines()

for line in tqdm(lines, ncols=80):
  line = line.strip()
  cmds = ['python', 'test_sdf_space.py',
          'MODEL.name', 'optpos',
          'MODEL.activation', 'softplus',
          'MODEL.code_num', '33',
          'MODEL.code_channel', '64',
          'MODEL.shape_num', '1',
          'DATA.train.filename', 'data/dfaust/points/{}.npy'.format(line),
          'DATA.train.pc_num', '50000',
          'DATA.train.normalize', 'False',
          'SOLVER.gpu', '0,',
          'SOLVER.logdir', logdir,
          'SOLVER.num_epochs', '400',
          'SOLVER.alias', line,
          'SOLVER.test_every_epoch', '20',
          'SOLVER.ckpt', ckpt]
  cmd = ' '.join(cmds)
  tqdm.write(cmd)
  os.system(cmd)
