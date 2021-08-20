import os
import torch
import numpy as np
from tqdm import tqdm
from config import parse_args
from models import make_mlp_model
from losses import sdf_mae 
from utils import write_sdf_summary, create_mesh
from datasets import SDFVolume
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# configs
FLAGS = parse_args()
print('\n' + FLAGS.SYS.cmds)

# dataset
flags_data = FLAGS.DATA.train
sdf_dataset = SDFVolume(flags_data.filename, flags_data.pc_num)
dataloader = DataLoader(sdf_dataset, shuffle=True,
                        batch_size=1, pin_memory=True, num_workers=0)

# model
model = make_mlp_model(FLAGS.MODEL)
print(model)
model.cuda()

# load checkpoints
flags_solver = FLAGS.SOLVER
if flags_solver.ckpt:
  print('loading checkpoint %s' % flags_solver.ckpt)
  model.load_state_dict(torch.load(flags_solver.ckpt))
optim = torch.optim.Adam(lr=flags_solver.learning_rate, params=model.parameters())

# summaries
logdir = flags_solver.logdir
writer = SummaryWriter(logdir)
ckpt_dir = os.path.join(logdir, 'checkpoints')
mesh_dir = os.path.join(logdir, 'mesh')
if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
if not os.path.exists(mesh_dir): os.makedirs(mesh_dir)


# train
def train_step(global_step):
  model.train()
  avg_loss = []
  for i, data in enumerate(dataloader):
    coords, sdf_gt = data[0].cuda(), data[1].cuda()

    sdf = model(coords)
    losses = sdf_mae(sdf, sdf_gt, min=-0.1, max=0.1)
    total_loss = losses['total_train_loss']

    optim.zero_grad()
    total_loss.backward()
    optim.step()

    for k, v in losses.items():
      writer.add_scalar(k, v.detach().cpu().item(), global_step + i)
    avg_loss.append(total_loss.detach().cpu().item())
  return np.mean(avg_loss)


# test
def test_step(epoch=-1, save_sdf=False):
  model.eval()
  filename = os.path.join(mesh_dir, '%s_%04d.ply' % (flags_solver.alias, epoch)) 
  tqdm.write("Epoch %d, Extract mesh: %s" % (epoch, filename))
  create_mesh(model, filename, N=flags_solver.resolution, 
              save_sdf=save_sdf, level=flags_solver.level_set)


# run
def train():
  num = len(dataloader)
  for epoch in tqdm(range(flags_solver.num_epochs), ncols=80):
    global_step = epoch * num
    if epoch % flags_solver.test_every_epoch == 0:
      write_sdf_summary(model, writer, global_step)
      ckpt_name = os.path.join(ckpt_dir, 'model_%05d.pth' % epoch)
      torch.save(model.state_dict(), ckpt_name)
      test_step(epoch, save_sdf=False)
    train_loss = train_step(global_step)
    if epoch % 30 == 0:
      tqdm.write("Epoch %d, Total loss %0.6f" % (epoch, train_loss))

  ckpt_name = os.path.join(ckpt_dir, 'model_final.pth')
  torch.save(model.state_dict(), ckpt_name)
  test_step(flags_solver.num_epochs, save_sdf=True)
  upsample_code()

# test
def test():
  epoch = flags_solver.num_epochs
  test_step(epoch, save_sdf=True)

# upsample the hidden code
def upsample_code():
  size = flags_solver.upsample_size
  if size < 0: return

  # upsample
  model_dict = model.state_dict()
  with torch.no_grad():
    code = model.pos_enc.upsample(size)
  model_dict['pos_enc.shape_code'] = code

  # save checkpoints
  ckpt_name = os.path.join(ckpt_dir, 'model_final_upsample_%03d.pth' % size)
  torch.save(model_dict, ckpt_name)


if __name__ == '__main__':
  eval('{}()'.format(flags_solver.run))
