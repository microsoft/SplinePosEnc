import os
import sys
import shutil
import argparse
from yacs.config import CfgNode as CN

_C = CN()

# SOLVER related parameters
_C.SOLVER = CN()
_C.SOLVER.gpu              = (0,)      # The gpu ids
_C.SOLVER.ckpt             = ''        # Restore weights from checkpoint file
_C.SOLVER.alias            = 'exp'     # The alias of the experiment
_C.SOLVER.logdir           = 'logs'    # Directory where to write event logs
_C.SOLVER.run              = 'train'   # Choose from train or test
_C.SOLVER.type             = 'adam'    # Choose from sgd or adam
_C.SOLVER.num_epochs       = 10000     # Maximum training iterations
_C.SOLVER.test_every_epoch = 200       # Test model every n training epochs
_C.SOLVER.lr_type          = 'constant'# Learning rate type: step or cos
_C.SOLVER.learning_rate    = 1.0e-4    # Initial learning rate
_C.SOLVER.gamma            = 0.1       # Learning rate step-wise decay
_C.SOLVER.step_size        = (100,)    # Learning rate step size.
_C.SOLVER.upsample_size    = -1        # Used to upsample the hidden code
_C.SOLVER.resolution       = 256       # For marching_cube
_C.SOLVER.level_set        = 0         # For marching_cube
_C.SOLVER.sphere_init      = ''        # Init from sphere
_C.SOLVER.start_epoch      = 0         # The initial epoch number
_C.SOLVER.optim_ckpt       = ''
_C.SOLVER.save_sdf         = True


# DATA related parameters
_C.DATA = CN()
_C.DATA.train = CN()
_C.DATA.train.batch_size = 1          # The batch size
_C.DATA.train.root_folder= ''         # The root folder containing point clouds
_C.DATA.train.in_memory  = False      # Whether to load the dataset in memory
_C.DATA.train.filename   = '.'        # The path of the point cloud
_C.DATA.train.pc_num     = 10000      # The point number used for training
_C.DATA.train.scale      = 0.9        # The scale factor used to normalize the points
_C.DATA.train.normalize  = True       # Whether to normalize the points

_C.DATA.test = _C.DATA.train.clone()


# MODEL related parameters
_C.MODEL = CN()
_C.MODEL.name              = ''         # The name of the model
_C.MODEL.in_features       = 3          # The input feature channel
_C.MODEL.projs             = -1         # The number of projection directions
_C.MODEL.out_features      = 1          # The output feature channel
_C.MODEL.num_hidden_layers = 3          # The number of hidden layers
_C.MODEL.hidden_features   = 256        # The number of hidden neurons
_C.MODEL.activation        = 'relu'     # The activation function
_C.MODEL.code_num          = 64         # The dim of hidden codes for OptPosEnc
_C.MODEL.shape_num         = 1          # The number of shapes
_C.MODEL.code_channel      = 64         # The channel of hidden codes for OptPosEnc
_C.MODEL.num_frequencies   = 8          # The number of frequencies for the Fourier Feature
_C.MODEL.scale             = 10         # The scale for the random Fourier Feature
_C.MODEL.fpe               = False     


# loss related parameters
_C.LOSS = CN()
_C.LOSS.normal_weight      = 1.0        # The weight decay on model weights
_C.LOSS.grad_weight        = 0.1        # The weight factors for different losses
_C.LOSS.sphere_loss        = False        


# backup the commands
_C.SYS = CN()
_C.SYS.cmds                = ''          # Used to backup the commands


FLAGS = _C


def _update_config(FLAGS, args):
  FLAGS.defrost()
  if args.config:
    FLAGS.merge_from_file(args.config)
  if args.opts:
    FLAGS.merge_from_list(args.opts)
  FLAGS.SYS.cmds = ' '.join(sys.argv)
  FLAGS.freeze()


def _backup_config(FLAGS, args):
  logdir = FLAGS.SOLVER.logdir
  if not os.path.exists(logdir):
    os.makedirs(logdir)
  else:
    print('Warning: The logdir already exists: %s' % logdir)
  # copy the file to logdir
  if args.config:
    shutil.copy2(args.config, logdir)
  # dump all configs
  filename = os.path.join(logdir, 'all_configs.yaml')
  with open(filename, 'w') as fid:
    fid.write(FLAGS.dump())


def _set_env_var(FLAGS):
  gpus = ','.join([str(a) for a in FLAGS.SOLVER.gpu])
  os.environ['CUDA_VISIBLE_DEVICES'] = gpus


def parse_args(backup=True):
  parser = argparse.ArgumentParser(description='The configs')
  parser.add_argument('--config', type=str,
                      help='experiment configure file name')
  parser.add_argument('opts', nargs=argparse.REMAINDER,
                      help="Modify config options using the command-line")

  args = parser.parse_args()
  _update_config(FLAGS, args)
  if backup:
    _backup_config(FLAGS, args)
  _set_env_var(FLAGS)
  return FLAGS


if __name__ == '__main__':
  flags = parse_args(backup=False)
  print(flags)
