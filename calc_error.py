import os
import argparse
from utils import calc_chamfer, calc_sdf_err

parser = argparse.ArgumentParser()
parser.add_argument('--filename_gt', type=str, required=True)
parser.add_argument('--filename_pred', type=str, required=True)
parser.add_argument('--metric', type=str, required=True)
parser.add_argument('--point_num', type=int, default=320000)
parser.add_argument('--filename_out', type=str, required=True)
args = parser.parse_args()


filename_gt = args.filename_gt
filename_pred = args.filename_pred
metric = args.metric
filename_out = args.filename_out
point_num = args.point_num

output_path = os.path.dirname(filename_out)
if not os.path.exists(output_path): os.makedirs(output_path)

with open(filename_out, 'a') as fid:
  if metric == 'chamfer':
    chamfer_a, chamfer_b = calc_chamfer(filename_gt, filename_pred, point_num)
    result = '{}, {}, {}, {:.4f}, {:.4f}, {:.4f}\n'.format(
        os.path.dirname(filename_gt),
        os.path.basename(filename_pred),
        point_num, chamfer_a, chamfer_b, chamfer_a + chamfer_b)
  elif metric == 'sdf':
    sdf_err = calc_sdf_err(filename_gt, filename_pred)
    result = '{}, {}, {:.4f}'.format(
        os.path.dirname(filename_gt),
        os.path.basename(filename_pred),
        sdf_err)
  else:
    raise NotImplementedError

  fid.write(result)
  print(result)
