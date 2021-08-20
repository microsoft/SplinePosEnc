import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import skimage.measure
import trimesh
from scipy.spatial import cKDTree


def get_mgrid(size, dim=2, offset=0.5, r=-1):
  '''
  Example: 
  >>> get_mgrid(3, dim=2, offset=0.5, r=1)  
      array([[-0.667, -0.667],
             [-0.667,  0.   ],
             [-0.667,  0.667],
             [ 0.   , -0.667],
             [ 0.   ,  0.   ],
             [ 0.   ,  0.667],
             [ 0.667, -0.667],
             [ 0.667,  0.   ],
             [ 0.667,  0.667]], dtype=float32)
  '''
  coords = np.arange(0, size, dtype=np.float32)
  coords = (coords + offset) * 2 / size - 1  # [0, size] -> [-1, 1]
  output = np.meshgrid(*[coords]*dim, indexing='ij')
  output = np.stack(output[::r], -1)
  output = output.reshape(size**dim, dim)
  return output


def lin2img(tensor):
  batch_size, num_samples, channels = tensor.shape
  size = np.sqrt(num_samples).astype(int)
  return tensor.permute(0, 2, 1).view(batch_size, channels, size, size)


def make_contour_plot(array_2d, mode='log'):
  fig, ax = plt.subplots(figsize=(2.75, 2.75), dpi=300)

  if(mode == 'log'):
    num_levels = 6
    levels_pos = np.logspace(-2, 0, num=num_levels)  # logspace
    levels_neg = -1. * levels_pos[::-1]
    levels = np.concatenate((levels_neg, np.zeros((0)), levels_pos), axis=0)
    colors = plt.get_cmap("Spectral")(np.linspace(0., 1., num=num_levels*2+1))
  elif(mode == 'lin'):
    num_levels = 10
    levels = np.linspace(-.5, .5, num=num_levels)
    colors = plt.get_cmap("Spectral")(np.linspace(0., 1., num=num_levels))
  else:
    raise NotImplementedError

  sample = np.flipud(array_2d)
  CS = ax.contourf(sample, levels=levels, colors=colors)
  cbar = fig.colorbar(CS)

  ax.contour(sample, levels=levels, colors='k', linewidths=0.1)
  ax.contour(sample, levels=[0], colors='k', linewidths=0.3)
  ax.axis('off')
  return fig


def write_sdf_summary(model, writer, global_step):
  model.eval()
  coords_2d = get_mgrid(size=256, dim=2, offset=0, r=1)
  coords_2d = torch.from_numpy(coords_2d)
  with torch.no_grad():
    zeros = torch.zeros_like(coords_2d[:, :1])
    ones = torch.ones_like(coords_2d[:, :1])
    names = ['train_yz_sdf_slice', 'train_xz_sdf_slice', 'train_xy_sdf_slice']
    coords = [torch.cat((zeros, coords_2d), dim=-1),
              torch.cat((coords_2d[:, :1], zeros, coords_2d[:, -1:]), dim=-1),
              torch.cat((coords_2d, -0.75*ones), dim=-1)]
    for name, coord in zip(names, coords):
      coord = coord.unsqueeze(0).cuda()
      sdf_values = model(coord)
      sdf_values = lin2img(sdf_values).squeeze().cpu().numpy()
      fig = make_contour_plot(sdf_values)
      writer.add_figure(name, fig, global_step=global_step)


def calc_sdf(model, N=256, max_batch=32**3):
  # generate samples
  num_samples = N ** 3
  samples = get_mgrid(N, dim=3, offset=0, r=1)
  samples = torch.from_numpy(samples)
  sdf_values = torch.zeros(num_samples)  

  # forward
  head = 0
  while head < num_samples:
    tail = min(head + max_batch, num_samples)
    sample_subset = samples[head:tail, :].cuda().unsqueeze(0)
    pred = model(sample_subset).squeeze().detach().cpu()
    sdf_values[head:tail] = pred
    head += max_batch
  sdf_values = sdf_values.reshape(N, N, N).numpy()
  return sdf_values


def create_mesh(model, filename, N=256, max_batch=32**3, level=0,
                save_sdf=False, **kwargs):
  # marching cubes
  sdf_values = calc_sdf(model, N, max_batch)
  vtx, faces = np.zeros((0, 3)), np.zeros((0, 3))
  try:
    vtx, faces, _, _ = skimage.measure.marching_cubes_lewiner(sdf_values, level)
  except:
    pass
  if vtx.size == 0 or faces.size == 0:
    print('Warning from marching cubes: Empty mesh!')
    return

  # normalize vtx
  voxel_size = 2.0 / N
  voxel_origin = np.array([-1, -1, -1])
  vtx = vtx * voxel_size + voxel_origin

  # save to ply and npy
  mesh = trimesh.Trimesh(vtx, faces)
  mesh.export(filename)
  if save_sdf:
    np.save(filename[:-4] + "_sdf.npy", sdf_values)


def calc_sdf_err(filename_gt, filename_pred):
  scale = 1.0e2  # scale the result for better display
  sdf_gt = np.load(filename_gt)
  sdf = np.load(filename_pred)
  err = np.abs(sdf - sdf_gt).mean() * scale
  return err


def calc_chamfer(filename_gt, filename_pred, point_num):
  scale = 1.0e5  # scale the result for better display
  np.random.seed(101)

  mesh_a = trimesh.load(filename_gt)
  points_a, _ = trimesh.sample.sample_surface(mesh_a, point_num)
  mesh_b = trimesh.load(filename_pred)
  points_b, _ = trimesh.sample.sample_surface(mesh_b, point_num)

  kdtree_a = cKDTree(points_a)
  dist_a, _ = kdtree_a.query(points_b)
  chamfer_a = np.mean(np.square(dist_a)) * scale

  kdtree_b = cKDTree(points_b)
  dist_b, _ = kdtree_b.query(points_a)
  chamfer_b = np.mean(np.square(dist_b)) * scale
  return chamfer_a, chamfer_b
