import os
import torch
import numpy as np
from tqdm import tqdm
from config import parse_args
from models import MLPSpace
from losses import sdf_loss
from utils import write_sdf_summary, create_mesh
from datasets import DFaustDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from functools import partial

# configs
FLAGS = parse_args()

# dataset
flags_data = FLAGS.DATA.train
dfaust_dataset = DFaustDataset(**flags_data)
dataloader = DataLoader(dfaust_dataset, batch_size=flags_data.batch_size,
                        num_workers=24, shuffle=True, pin_memory=True,
                        drop_last=True)

# model
model = MLPSpace(**FLAGS.MODEL)
print(model)
model.cuda()

# load checkpoints
flags_solver = FLAGS.SOLVER
if flags_solver.ckpt:
  print('loading checkpoint %s' % flags_solver.ckpt)
  model.load_state_dict(torch.load(flags_solver.ckpt))

# init from sphere
if FLAGS.MODEL.name == 'optpos' and flags_solver.sphere_init:
  print('Init from sphere, load: %s' % flags_solver.sphere_init)
  trained_dict = torch.load(flags_solver.sphere_init)
  shape_num = FLAGS.MODEL.shape_num
  shape_code = trained_dict.pop('pos_enc.shape_code')
  trained_dict['pos_enc.shape_code'] = shape_code.repeat(1, shape_num)
  model_dict = model.state_dict()
  model_dict.update(trained_dict)
  model.load_state_dict(model_dict)
if FLAGS.MODEL.name == 'mlp' and flags_solver.sphere_init:
  net = model.net.net
  for i in range(len(net)-1):
    weight, bias = net[i].linear.weight, net[i].linear.bias
    torch.nn.init.normal_(weight, 0.0, np.sqrt(2 / weight.shape[0]))
    torch.nn.init.constant_(bias, 0.0)
  weight, bias = net[-1].linear.weight, net[-1].linear.bias
  torch.nn.init.constant_(bias, -0.6)
  torch.nn.init.normal_(weight, mean=np.sqrt(np.pi / weight.shape[1]), std=1e-5)

# optmizer
lr = flags_solver.learning_rate
optim = torch.optim.Adam(lr=lr, params=model.parameters())
if flags_solver.optim_ckpt:
  print('loading checkpoint %s' % flags_solver.optim_ckpt)
  optim.load_state_dict(torch.load(flags_solver.optim_ckpt))


# summaries
logdir = flags_solver.logdir
ckpt_dir = os.path.join(logdir, 'checkpoints')
writer = SummaryWriter(logdir)
if not os.path.exists(ckpt_dir):
  os.makedirs(ckpt_dir)


# latent code regularization
def shape_code_reg(idx):
  shape_code = model.pos_enc.get_shape_code(idx)
  code_loss = shape_code.square().mean()  # or sum()
  return code_loss

# train
def train_step(model_train, global_step):
  model_train.train()
  avg_loss = []
  for i, data in enumerate(dataloader):
    coords = data[0].cuda().requires_grad_()
    sdf_gt, normal_gt, idx = data[1].cuda(), data[2].cuda(), data[3].cuda()

    sdf = model_train(coords, idx)
    losses = sdf_loss(sdf, coords, sdf_gt, normal_gt,
                      normal_weight=FLAGS.LOSS.normal_weight,
                      grad_weight=FLAGS.LOSS.grad_weight)
    total_train_loss = losses['total_train_loss']

    # latent code regularization
    code_loss = shape_code_reg(idx)
    total_loss = total_train_loss + code_loss * 1e-4

    optim.zero_grad()
    total_loss.backward()
    optim.step()

    # tqdm.write("step %d" % (global_step + i))
    for k, v in losses.items():
      writer.add_scalar(k, v.detach().cpu().item(), global_step + i)
    writer.add_scalar('latent', code_loss.detach().cpu().item(), global_step+1)
    avg_loss.append(total_loss.detach().cpu().item())
  return np.mean(avg_loss)


# test
def test_step(epoch=0, idx=None, save_sdf=True):
  model.eval()
  if idx is None:
    idx = np.random.randint(len(dfaust_dataset))
  output_path = os.path.join(logdir, 'mesh')
  if not os.path.exists(output_path): os.makedirs(output_path)
  filename = '%s_%04d_%04d.ply' % (flags_solver.alias, epoch, idx)
  filename = os.path.join(output_path, filename)
  model_test = partial(model, idx=idx)
  create_mesh(model_test, filename, N=flags_solver.resolution,
              save_sdf=save_sdf, level=flags_solver.level_set)


# run train
def train():
  model_train = model
  if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model_train = torch.nn.DataParallel(model)  # use multiple gpus

  num = len(dataloader)
  rng = range(flags_solver.start_epoch, flags_solver.num_epochs)
  for epoch in tqdm(rng, ncols=80):
    global_step = epoch * num
    if epoch % flags_solver.test_every_epoch == 0:
      write_sdf_summary(model, writer, global_step)
      save_state(filename='model_%05d' % epoch)
      test_step(epoch, save_sdf=False)
    train_loss = train_step(model_train, global_step)
    tqdm.write("Epoch %d, Total loss %0.6f" % (epoch, train_loss))
  save_state(filename='model_final')
  upsample_code()


# run test
def test():
  num = FLAGS.MODEL.shape_num
  for i in tqdm(range(num), ncols=80):
    test_step(idx=i, save_sdf=True)


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


# save model and solver state
def save_state(filename):
  model_dict = model.state_dict()
  ckpt_name = os.path.join(ckpt_dir, filename + '.pth')
  torch.save(model_dict, ckpt_name)

  ckpt_name = os.path.join(ckpt_dir, filename + '.mean.pth')
  model_dict['pos_enc.shape_code'] = model.pos_enc.get_mean_code()
  torch.save(model_dict, ckpt_name)

  ckpt_name = os.path.join(ckpt_dir, filename + '.solver.pth')
  torch.save(optim.state_dict(), ckpt_name)


if __name__ == '__main__':
  eval('{}()'.format(flags_solver.run))