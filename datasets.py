import os
import torch
from torch.utils.data import Dataset
import imageio
import numpy as np
from utils import get_mgrid

def rescale_img(img, tmin=-1.0, tmax=1.0):
  img = np.clip(img, tmin, tmax)
  img = ((img + 1.0) * 255.0).astype(np.uint8)
  return img


def sample_points(coords, normals, on_surface_samples):
  point_cloud_size = coords.shape[0]
  on_surface_samples = min(on_surface_samples, point_cloud_size)
  off_surface_samples = on_surface_samples

  # random coords on the surface
  rand_idx = np.random.choice(point_cloud_size, size=on_surface_samples)
  on_surface_coords = coords[rand_idx, :]
  on_surface_normals = normals[rand_idx, :]
  on_surface_sdfs = np.zeros((on_surface_samples, 1))

  # random coords in the volume
  off_surface_coords = np.random.uniform(-1, 1, (off_surface_samples, 3))
  off_surface_normals = np.ones((off_surface_samples, 3)) * -1
  off_surface_sdfs = np.ones((off_surface_samples, 1)) * -1

  # outputs
  coords = np.concatenate((on_surface_coords, off_surface_coords), axis=0)
  sdf = np.concatenate((on_surface_sdfs, off_surface_sdfs), axis=0)
  normals = np.concatenate((on_surface_normals, off_surface_normals), axis=0)
  return coords.astype(np.float32), sdf.astype(np.float32), normals.astype(np.float32)


# Reshape point cloud such that it lies in bounding box of (-1, 1) * scale
def normalize_points(coords, scale=0.9):
  coords -= np.mean(coords, axis=0, keepdims=True)
  coord_max, coord_min = np.amax(coords), np.amin(coords)
  coords = (coords - coord_min) / (coord_max - coord_min)  # (0, 1)
  coords = (coords - 0.5) * (2.0 * scale)
  return coords


def load_points(filename: str):
  if filename.endswith('.xyz'):
    points = np.loadtxt(filename)
  elif filename.endswith('.npy'):
    points = np.load(filename)
  else:
    raise NotImplementedError
  return points


class PointCloud(Dataset):
  def __init__(self, pointcloud_path, on_surface_samples, scale=0.9, 
               normalize=True, **kwargs):
    super().__init__()
    point_cloud = load_points(pointcloud_path)
    self.on_surface_samples = on_surface_samples
    assert on_surface_samples < point_cloud.shape[0]
    self.coords = point_cloud[:, :3]
    self.normals = point_cloud[:, 3:]
    if normalize:
      self.coords = normalize_points(self.coords, scale)

  def __len__(self):
    return self.coords.shape[0] // self.on_surface_samples

  def __getitem__(self, idx):
    coords, sdfs, normals = sample_points(self.coords, self.normals, self.on_surface_samples)
    return torch.from_numpy(coords), torch.from_numpy(sdfs), torch.from_numpy(normals)


class DFaustDataset(Dataset):
  def __init__(self, root_folder, filename, pc_num, in_memory=False, **kwargs):
    super().__init__()
    self.root_folder = root_folder
    self.filenames = self.get_filenames(filename)
    self.on_surface_samples = pc_num
    self.in_memory = in_memory
    self.points = [None] * len(self.filenames)
    if in_memory:
      print('Load {} files into memory.'.format(len(self.filenames)))
      for i, filename in enumerate(self.filenames):
        self.points[i] = load_points(filename)

  def get_filenames(self, filelist):
    with open(filelist, 'r') as fid:
      lines = fid.readlines()
    filenames = [os.path.join(self.root_folder, line.strip()) for line in lines]
    return filenames

  def __len__(self):
    return len(self.filenames)

  def __getitem__(self, idx):
    if self.in_memory:
      points = self.points[idx]
    else:
      filename = self.filenames[idx]
      points = load_points(filename)
    coords, sdfs, normals = sample_points(points[:, :3], points[:, 3:], self.on_surface_samples)
    return torch.from_numpy(coords), torch.from_numpy(sdfs), torch.from_numpy(normals), idx


class SingleImage(Dataset):
  def __init__(self, filename):
    img = np.asarray(imageio.imread(filename)).astype(np.float32)
    img = img / 255.0 # [0, 1]
    # img = img * (2.0 / 255.0) - 1.0 # [-1, 1]
    assert img.shape[0] == img.shape[1]
    self.img = img.reshape(1, img.shape[0]*img.shape[1], -1)
    self.channel = img.shape[-1]
    self.resolution = img.shape[0]
    coords = get_mgrid(self.resolution, dim=2, offset=0.5, r=-1) # [-1, 1]
    self.coords = np.expand_dims(coords, axis=0)

  def __len__(self):
    return 1

  def __getitem__(self, idx):
    return torch.from_numpy(self.coords), torch.from_numpy(self.img)


class SDFVolume(Dataset):
  def __init__(self, filename, pc_num):
    self.pc_num = pc_num

    sdf = np.load(filename)
    assert sdf.shape[0] == sdf.shape[1] and sdf.shape[1] == sdf.shape[2]
    self.resolution = sdf.shape[0]
    self.total_num = sdf.shape[0] ** 3
    self.sdf = sdf.reshape(-1, 1)
    
    self.coords = get_mgrid(self.resolution, dim=3, offset=0, r=1)

  def __len__(self):
    return self.total_num // self.pc_num

  def __getitem__(self, idx):
    rnd = np.random.choice(self.total_num, self.pc_num)
    return torch.from_numpy(self.coords[rnd, :]), torch.from_numpy(self.sdf[rnd, :])
