import os
import argparse
import trimesh
import trimesh.sample
import numpy as np
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--root_folder', required=True, type=str)
parser.add_argument('--filelist', type=str, default='data/dfaust/all_shapes.txt')
parser.add_argument('--output_path', type=str, default='data/dfaust/points')
parser.add_argument('--samples', type=int, default=200000)
parser.add_argument('--scale', type=float, default=0.8)
args = parser.parse_args()


with open(args.filelist, 'r') as fid:
  lines = fid.readlines()
filenames = [line.strip() for line in lines]

for filename in tqdm(filenames, ncols=80):
  filename_ply = os.path.join(args.root_folder, filename + '.ply')
  filename_pts = os.path.join(args.output_path, filename + '.npy')
  filename_center = filename_pts[:-3] + 'center.npy'

  folder_pts = os.path.dirname(filename_pts)
  if not os.path.exists(folder_pts): 
    os.makedirs(folder_pts)

  mesh = trimesh.load(filename_ply)
  points, idx = trimesh.sample.sample_surface(mesh, args.samples)
  normals = mesh.face_normals[idx]

  center = np.mean(points, axis=0, keepdims=True)
  points = (points - center) * args.scale
  point_set = np.concatenate((points, normals), axis=-1).astype(np.float32)

  np.save(filename_pts, point_set)
  np.save(filename_center, center)
