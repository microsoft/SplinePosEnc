import os
import torch
import imageio
import numpy as np
from tqdm import tqdm
from config import parse_args
from models import make_mlp_model
from losses import img_mse, img_psnr
from datasets import SingleImage
from torch.utils.tensorboard import SummaryWriter

def downsample(input, factor=1):
  if factor == 1: return input # directly return
  channel = input.size(-1)
  output = input.view(img_size, img_size, channel)
  output = output[::factor, ::factor, :].reshape(1, -1, channel)
  return output

# configs
FLAGS = parse_args()
print('\n' + FLAGS.SYS.cmds)

# dataset
flags_data = FLAGS.DATA.train
single_image = SingleImage(flags_data.filename)
img_size = single_image.resolution
data = single_image[0]
coords, img_gt = data[0].cuda(), data[1].cuda()
coords_train, img_train = downsample(coords), downsample(img_gt)

# model
model = make_mlp_model(FLAGS.MODEL)
print(model)
model.cuda()

# solver
flags_solver = FLAGS.SOLVER
optim = torch.optim.Adam(lr=flags_solver.learning_rate, params=model.parameters())

# summaries
logdir = flags_solver.logdir
writer = SummaryWriter(logdir)
img_dir = os.path.join(logdir, 'img')
ckpt_dir = os.path.join(logdir, 'checkpoints')
if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
if not os.path.exists(img_dir): os.makedirs(img_dir)
img_name = os.path.join(img_dir, flags_solver.alias)
fid = open(os.path.join(logdir, 'summaries.csv'), 'w')
tqdm.write("Epoch, Loss, PSNR", fid)

# run train and test
for epoch in tqdm(range(flags_solver.num_epochs+1), ncols=80):
  # train
  model.train()
  img_pred = model(coords_train)
  train_loss = img_mse(img_pred, img_train)

  # optimize
  optim.zero_grad()
  train_loss.backward()
  optim.step()

  # test and write summaries
  if epoch % flags_solver.test_every_epoch == 0:
    model.eval()
    with torch.no_grad():
      img_pred = model(coords)
    img_pred = img_pred.clamp(0.0, 1.0) # clip pixel velues to [0, 1]
    test_loss = img_mse(img_pred, img_gt).item()
    psnr = img_psnr(test_loss)

    writer.add_scalar('train_loss', train_loss.item(), epoch)
    writer.add_scalar('total_loss', test_loss, epoch)
    writer.add_scalar('psnr', psnr, epoch)

    img_pred = img_pred.view(img_size, img_size, -1).detach().cpu().numpy()
    img_pred = (img_pred * 255).astype(np.uint8)
    writer.add_image('img', img_pred, global_step=epoch, dataformats='HWC')
    imageio.imwrite(img_name + '_%04d.png' % epoch, img_pred)

    ckpt_name = os.path.join(ckpt_dir, 'model_%05d.pth' % epoch)
    torch.save(model.state_dict(), ckpt_name)
    tqdm.write("%d, %0.6f, %0.6f" % (epoch, test_loss, psnr), fid)
    tqdm.write("Epoch %d, Loss %0.6f, PSNR %0.6f" % (epoch, test_loss, psnr))

fid.close()