import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--alias', type=str, default='0408')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--dry-run', action='store_true',)
args = parser.parse_args()

alias = args.alias
gpu = args.gpu

models = ['optpos', 'randfourier', 'siren', 'mlp']
images = sorted(os.listdir('data/images'))
images = [img[:-4] for img in images if img.endswith('.bmp')]

for img in images:
  for model in models:
    cmds = [
        'python regress_img.py',
        'MODEL.name {}'.format(model),
        'MODEL.activation relu',
        'MODEL.in_features 2',
        'MODEL.out_features 3',
        'MODEL.code_num 256',
        'MODEL.code_channel 64',
        'MODEL.projs 32',
        'MODEL.scale 12',
        'MODEL.num_frequencies 256',
        'DATA.train.filename data/images/{}.bmp'.format(img),
        'SOLVER.gpu {},'.format(gpu),
        'SOLVER.logdir logs/img/{}/{}/{}'.format(alias, img, model),
        'SOLVER.num_epochs 2000',
        'SOLVER.alias {}'.format(img),
        'SOLVER.test_every_epoch 100']

    cmd = ' '.join(cmds)
    print(cmd + '\n')
    if not args.dry_run:
      os.system(cmd)
