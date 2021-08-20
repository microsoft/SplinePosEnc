import os
import torch
import numpy as np
from tqdm import tqdm
from config import parse_args
from models import MLPSpace
from losses import sdf_loss
from datasets import PointCloud
from torch.utils.data import DataLoader
from utils import create_mesh

# configs
FLAGS = parse_args()

# dataset
flags_data = FLAGS.DATA.train
sdf_dataset = PointCloud(flags_data.filename, flags_data.pc_num, **flags_data)
dataloader = DataLoader(sdf_dataset, shuffle=True, batch_size=1, 
                        pin_memory=True, num_workers=0)

# model
model = MLPSpace(**FLAGS.MODEL)
print(model)
model.cuda()

# load checkpoints
flags_solver = FLAGS.SOLVER
if flags_solver.ckpt:
  print('loading checkpoint %s' % flags_solver.ckpt)
  model.load_state_dict(torch.load(flags_solver.ckpt))

# optmizer
lr = flags_solver.learning_rate
optim = torch.optim.Adam(lr=lr, params=model.pos_enc.parameters())

# summaries
logdir = flags_solver.logdir
# ckpt_dir = os.path.join(logdir, 'checkpoints')
# if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
# writer = SummaryWriter(logdir)


# train
def train_step(model_train, global_step):
  model_train.train()
  avg_loss = []
  for i, data in enumerate(dataloader):
    coords = data[0].cuda().requires_grad_()
    sdf_gt, normal_gt = data[1].cuda(), data[2].cuda()

    sdf = model_train(coords)
    losses = sdf_loss(sdf, coords, sdf_gt, normal_gt,
                      normal_weight=FLAGS.LOSS.normal_weight,
                      grad_weight=FLAGS.LOSS.grad_weight)
    total_loss = losses['total_train_loss']

    optim.zero_grad()
    total_loss.backward()
    optim.step()

    # for k, v in losses.items():
    #   writer.add_scalar(k, v.detach().cpu().item(), global_step + i)
    avg_loss.append(total_loss.detach().cpu().item())
  return np.mean(avg_loss)


# test
def test_step(epoch=0, save_sdf=True):
  model.eval()
  filename = os.path.join(logdir, 'mesh', '%s.ply' % flags_solver.alias)
  output_path = os.path.dirname(filename)
  if not os.path.exists(output_path): os.makedirs(output_path)
  create_mesh(model, filename, N=flags_solver.resolution, 
              save_sdf=save_sdf, level=flags_solver.level_set)


# run train
def train():
  model_train = model
  if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model_train = torch.nn.DataParallel(model)

  num = len(dataloader)
  for epoch in tqdm(range(flags_solver.start_epoch, flags_solver.num_epochs), ncols=80):
    global_step = epoch * num
    # if epoch % flags_solver.test_every_epoch == 0:
    #   write_sdf_summary(model, writer, global_step)
    #   ckpt_name = os.path.join(ckpt_dir, 'model_%05d.pth' % epoch)
    #   torch.save(model.state_dict(), ckpt_name)
    #   ckpt_name = os.path.join(ckpt_dir, 'solver_%05d.pth' % epoch)
    #   torch.save(optim.state_dict(), ckpt_name)
    #   test_step(epoch, save_sdf=False)
    train_loss = train_step(model_train, global_step)
    tqdm.write("Epoch %d, Total loss %0.6f" % (epoch, train_loss))  
  filename = os.path.join(logdir, 'mesh', '%s.pth' % flags_solver.alias)
  ckpt_path = os.path.dirname(filename)
  if not os.path.exists(ckpt_path): os.makedirs(ckpt_path)
  torch.save(model.state_dict(), filename)
  test_step(epoch, save_sdf=True)

# run test
def test():
  num = FLAGS.MODEL.shape_num
  for i in tqdm(range(num), ncols=80):
    test_step(idx=i, save_sdf=True)


if __name__ == '__main__':
  eval('{}()'.format(flags_solver.run))
