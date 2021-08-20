import torch
import numpy as np
from torch import nn


class ABS(torch.autograd.Function):
  '''The derivative of torch.abs on `0` is `0`, and in this implementation, we
  modified it to `1`
  '''
  @staticmethod
  def forward(ctx, input):
    ctx.save_for_backward(input)
    return input.abs()

  @staticmethod
  def backward(ctx, grad_in):
    input, = ctx.saved_tensors
    sign = input < 0
    grad_out = grad_in * (-2.0 * sign.to(input.dtype) + 1.0)
    return grad_out

Abs = ABS.apply


def spline_basis(x):
  t = torch.where(x < -0.5, 0.5 * (x + 1.5)** 2, 0.75 - x ** 2)
  weight = torch.where(x < 0.5, t, 0.5 * (x - 1.5)** 2)
  return weight


class Sine(nn.Module):
  '''  See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for the
  discussion of factor 30
  '''

  def __init__(self, w=30.0):
    super().__init__()
    self.w = w

  def forward(self, input):
    return torch.sin(self.w * input)

  def extra_repr(self):
    return 'w={}'.format(self.w)


class FCLayer(nn.Module):
  def __init__(self, in_features, out_features, act=nn.ReLU(inplace=True)):
    super().__init__()
    self.linear = nn.Linear(in_features, out_features)
    self.act = act

  def forward(self, input):
    output = self.linear(input)
    output = self.act(output)
    return output


class MLP(nn.Module):
  def __init__(self, in_features=2, out_features=1, num_hidden_layers=3,
               hidden_features=256, act=nn.Softplus(beta=100), **kwargs):
    super().__init__()
    net = [FCLayer(in_features, hidden_features, act)]
    for i in range(num_hidden_layers):
      net.append(FCLayer(hidden_features, hidden_features, act))
    net.append(FCLayer(hidden_features, out_features, nn.Identity()))
    self.net = nn.Sequential(*net)

  def forward(self, input):
    output = self.net(input)
    return output


class Siren(MLP):
  def __init__(self, in_features=2, out_features=1, num_hidden_layers=3,
               hidden_features=256, w=30, **kwargs):
    super().__init__(in_features, out_features, num_hidden_layers,
                     hidden_features, act=Sine(w))
    self.reset_parameters(w)

  def reset_parameters(self, w=30.0):
    is_first_layer = True
    for k, v in self.net.named_parameters():
      if 'weight' in k:
        num_input = v.size(-1)
        if is_first_layer:
          is_first_layer = False
          bnd = 1.0 / num_input
        else:
          bnd = np.sqrt(6.0 / num_input) / w
        nn.init.uniform_(v, -bnd, bnd)


class RandFourier(nn.Module):
  def __init__(self, in_features, num_frequencies=64, scale=10, **kwargs):
    super().__init__()
    self.in_features = in_features
    self.scale = scale
    self.num_frequencies = num_frequencies
    self.out_features = 2 * self.num_frequencies
    self.register_buffer('proj', torch.Tensor(in_features, num_frequencies))
    self.reset_parameters()

  def reset_parameters(self):
    with torch.no_grad():
      # Multiply by np.pi instead of (2 * np.pi) since the input is in [-1, 1].
      # In the original paper, the input is in [0, 1]
      self.proj.copy_(torch.randn_like(self.proj) * (self.scale * np.pi))

  def forward(self, coords):
    pos_enc = torch.mm(coords.flatten(start_dim=0, end_dim=1), self.proj)
    pos_enc = torch.cat([torch.sin(pos_enc), torch.cos(pos_enc)], axis=-1)
    output = pos_enc.view(coords.shape[0], coords.shape[1], self.out_features)
    return output

  def extra_repr(self) -> str:
    return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)


class FourierFeat(nn.Module):
  def __init__(self, in_features, num_frequencies=4, **kwargs):
    super().__init__()
    self.in_features = in_features
    self.num_frequencies = num_frequencies
    self.out_features = 2 * in_features * self.num_frequencies

  def forward(self, coords):
    # mul = [np.pi * 2 ** i for i in range(self.num_frequencies)]
    mul = [2 ** i for i in range(self.num_frequencies)]
    pos_enc = torch.unsqueeze(coords, -1) * torch.Tensor(mul).to(coords.device)
    pos_enc = torch.cat([torch.sin(pos_enc), torch.cos(pos_enc)], axis=-1)
    output = pos_enc.flatten(start_dim=-2, end_dim=-1)
    return output

  def extra_repr(self) -> str:
    return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)


class OptPosEnc(nn.Module):
  def __init__(self, in_features, code_num=64, code_channel=64, **kwargs):
    super().__init__()
    self.in_features = in_features
    self.out_features = code_channel
    self.code_num = code_num

    code_size = [code_channel, in_features * code_num]
    self.shape_code = nn.Parameter(torch.Tensor(*code_size))
    # self.register_buffer('shape_code', torch.Tensor(*code_size))
    self.reset_parameters()

  def reset_parameters(self,):
    nn.init.xavier_uniform_(self.shape_code)
    # bnd = np.sqrt(1.0 / self.in_features)
    # nn.init.uniform_(self.shape_code, -bnd, bnd)

  def forward(self, coords):
    return self._forward(coords, self.shape_code)

  def _forward(self, coords, shape_code):
    pt_num, in_features = coords.size(1), coords.size(2)
    assert in_features == self.in_features
    code_num = shape_code.size(1) // in_features
    mul = [[[[code_num * i] for i in range(in_features)]]] # [1, 1, D, 1]
    mul = torch.tensor(mul, dtype=torch.int64, device=coords.device)
    mask = torch.tensor([[[[0, 1]]]], dtype=torch.float32, device=coords.device)

    coords = (coords + 1.0) * ((code_num - 1) / 2.0) # [-1, 1] -> [0, code_num-1]
    corners = torch.floor(coords).detach()    # [1, N, D]
    corners = corners.unsqueeze(-1) + mask    # [1, N, D, 2]
    index = corners.to(torch.int64) + mul     # [1, N, D, 2]
    coordsf = coords.unsqueeze(-1) - corners  # [1, N, D, 2], local coords [-1, 1]
    weights = 1.0 - torch.abs(coordsf)        # (1, N, D, 2)

    coords_code = torch.index_select(shape_code, 1, index.view(-1))
    coords_code = coords_code.view(-1, pt_num, in_features, 2) # (C, N, D, 2)
    output = torch.sum(coords_code * weights, dim=(-2, -1), keepdim=True)
    output = output.squeeze(-1).permute(2, 1, 0)
    return output

  def upsample(self, size=64):
    code = self.shape_code.view(self.out_features, self.in_features, self.code_num)
    code = code.permute(1, 0, 2)
    output = torch.nn.functional.upsample(code, size=size, mode='linear',
                                          align_corners=True)
    output = output.permute(1, 0, 2)
    output = output.reshape(self.out_features, -1)
    return output

  def extra_repr(self) -> str:
    return 'in_features={}, code_channel={}, code_num={}'.format(
        self.in_features, self.out_features, self.code_num)


class PosEncMLP(nn.Module):
  def __init__(self, in_features=3, out_features=1, num_hidden_layers=3,
               hidden_features=256, act=nn.Softplus(beta=100),
               pos_enc=FourierFeat, projs=-1, mlp=MLP, **kwargs):
    super().__init__()
    if projs > 0:
      assert projs >= in_features
      self.proj = PosProj(in_features, projs)
      in_features = projs
    else:
      self.proj = nn.Identity()

    self.pos_enc = pos_enc(in_features, **kwargs)
    self.net = mlp(self.pos_enc.out_features, out_features, num_hidden_layers,
                   hidden_features, act=act)

  def forward(self, input):
    coords = self.proj(input)
    enc = self.pos_enc(coords)
    output = self.net(enc)
    return output


class PosEncSiren(PosEncMLP):
  def __init__(self, in_features=3, out_features=1, num_hidden_layers=3,
               hidden_features=256, projs=-1, **kwargs):
    super().__init__(in_features, out_features, num_hidden_layers,
                     hidden_features, projs=projs, mlp=Siren, pos_enc=OptPosEnc,
                     **kwargs)
    self.reset_parameters()

  def reset_parameters(self):
    with torch.no_grad():
      shape_code = torch.zeros_like(self.pos_enc.shape_code)
      channel, num = self.pos_enc.shape_code.shape
      code_num, in_features = self.pos_enc.code_num, self.pos_enc.in_features

      delta =  2./ (code_num-1)
      ch = channel // in_features
      t = torch.arange(-1, 1 + 0.1 * delta, step=delta)
      t = t.unsqueeze(0).repeat(ch, 1)
      for i in range(in_features):
        n = (torch.rand_like(t) - 0.5) * 1.0e-2
        shape_code[i*ch:(i+1)*ch, i*code_num:(i+1)*code_num] = t + n
      self.pos_enc.shape_code.copy_(shape_code)


class GlobalPosEnc(nn.Module):
  def __init__(self, in_features=3, shape_num=1, code_channel=64, fpe=True, **kwargs):
    super().__init__()
    self.in_features = in_features
    self.code_channel = code_channel
    self.out_features = code_channel + in_features
    self.shape_num = shape_num
    self.fpe = fpe
    if self.fpe:
      self.pos_enc = RandFourier(in_features, num_frequencies=64, scale=4)
      self.out_features = code_channel + self.pos_enc.out_features

    self.shape_code = nn.Parameter(torch.Tensor(shape_num, code_channel))
    nn.init.zeros_(self.shape_code)

  def forward(self, coords, idx):
    points_num = coords.shape[1]
    if type(idx) is int: idx = torch.tensor([idx]).to(coords.device)
    code = torch.index_select(self.shape_code, 0, idx) # (B, C)
    code = code.unsqueeze(1).repeat(1, points_num, 1)  # (B, N, C)
    if self.fpe:
      coords = self.pos_enc(coords)
    output = torch.cat([coords, code], dim=-1)
    return output

  def get_shape_code(self, idx):
    return torch.index_select(self.shape_code, 0, idx)

  def get_mean_code(self):
    return self.shape_code.mean(0, keepdim=True)

  def extra_repr(self) -> str:
    return 'shape=({}, {})'.format(self.shape_num, self.code_channel)


class OptPosEncBatch(nn.Module):
  def __init__(self, in_features, code_num=64, code_channel=64, shape_num=1, **kwargs):
    super().__init__()
    self.in_features = in_features
    self.out_features = code_channel
    self.code_num = code_num
    self.shape_num = shape_num

    code_size = [code_channel, shape_num * in_features * code_num]
    self.shape_code = nn.Parameter(torch.Tensor(*code_size))
    nn.init.xavier_uniform_(self.shape_code)

  def forward(self, coords, idx):
    if type(idx) is int: idx = torch.tensor(idx).to(coords.device, dtype=torch.int64)
    batch_size, pt_num = coords.size(0), coords.size(1)
    mask = torch.tensor([[[[0, 1]]]], dtype=torch.float32, device=coords.device)
    mul = [[[[self.code_num * i] for i in range(self.in_features)]]]
    mul = torch.tensor(mul, dtype=torch.int64, device=coords.device)
    mul1 = (idx * (self.code_num * self.in_features)).view(-1, 1, 1, 1)

    coords = (coords + 1.0) * ((self.code_num - 1) / 2.0) # [-1, 1] -> [0, code_num-1]
    corners = torch.floor(coords).detach()       # [B, N, D]
    corners = corners.unsqueeze(-1) + mask       # [B, N, D, 2]
    coordsf = coords.unsqueeze(-1) - corners     # [B, N, D, 2], local coords [-1, 1]
    index = corners.to(torch.int64) + mul        # [B, N, D, 2]
    index = index + mul1                         # [B, N, D, 2]
    weights = 1.0 - torch.abs(coordsf)           # [B, N, D, 2], TODO: use ABS
    weights = weights.unsqueeze(0)               # [1, B, N, D, 2]

    coords_code = torch.index_select(self.shape_code, 1, index.view(-1))
    coords_code = coords_code.view(self.out_features, batch_size,
                                   pt_num, self.in_features, 2) # [C, B, N, D, 2]
    output = torch.sum(coords_code * weights, dim=(-2, -1))     # [C, B, N]
    output = output.permute(1, 2, 0)                            # [B, N, C]
    return output

  def get_shape_code(self, idx):
    shape_code = self.shape_code.view(self.out_features, self.shape_num, -1)
    output = torch.index_select(shape_code, 1, idx).view(self.out_features, -1)
    return output

  def get_mean_code(self):
    return self.shape_code.view(self.out_features, self.shape_num, -1).mean(1)

  def upsample(self, size=64):
    code = self.shape_code.view(self.out_features, -1, self.code_num)
    code = code.permute(1, 0, 2)
    output = torch.nn.functional.upsample(code, size=size, mode='linear',
                                          align_corners=self._align_corners())
    output = output.permute(1, 0, 2)
    output = output.reshape(self.out_features, -1)
    return output

  def _align_corners(self):
    return True

  def extra_repr(self) -> str:
    return 'in_features={}, code_channel={}, code_num={}, shape_num={}'.format(
        self.in_features, self.out_features, self.code_num, self.shape_num)


class OptPosEncBatch2(OptPosEncBatch):
  def forward(self, coords, idx=0):
    if type(idx) is int: idx = torch.tensor(idx).to(coords.device, dtype=torch.int64)
    batch_size, pt_num = coords.size(0), coords.size(1)
    mask = torch.tensor([[[[-1, 0, 1]]]], dtype=torch.float32, device=coords.device)
    mul = [[[[self.code_num * i] for i in range(self.in_features)]]]
    mul = torch.tensor(mul, dtype=torch.int64, device=coords.device)
    mul1 = (idx * (self.code_num * self.in_features)).view(-1, 1, 1, 1)

    coords = (coords * 0.8 + 1.0) * (self.code_num / 2.0) # [-1, 1] -> [1, code_num]
    coords = coords - 0.5           # Suppose the code is defined on the grid center
    corners = torch.round(coords).detach()       # [B, N, D]
    corners = corners.unsqueeze(-1) + mask       # [B, N, D, 3]
    coordsf = coords.unsqueeze(-1) - corners     # [B, N, D, 3], local coords [-1.5, 1.5]
    index = corners.to(torch.int64) + mul        # [B, N, D, 3]
    index = torch.clamp(index, 0, self.code_num * self.in_features - 1)
    index = index + mul1                         # [B, N, D, 3]
    weights = spline_basis(coordsf)              # [B, N, D, 3]
    weights = weights.unsqueeze(0)               # [1, B, N, D, 3]

    coords_code = torch.index_select(self.shape_code, 1, index.view(-1))
    coords_code = coords_code.view(self.out_features, batch_size,
                                   pt_num, self.in_features, 3) # [C, B, N, D, 3]
    output = torch.sum(coords_code * weights, dim=(-2, -1))     # [C, B, N]
    output = output.permute(1, 2, 0)                            # [B, N, C]
    return output

  def _align_corners(self):
    return False


class OptPosEncVol(nn.Module):
  def __init__(self, in_features, code_num=64, code_channel=64, shape_num=1, **kwargs):
    super().__init__()
    self.in_features = in_features
    self.out_features = code_channel
    self.code_num = code_num
    self.shape_num = shape_num

    code_size = [code_channel, shape_num * code_num ** in_features]
    self.shape_code = nn.Parameter(torch.Tensor(*code_size))
    nn.init.xavier_uniform_(self.shape_code)

  def forward(self, coords, idx=0):
    if type(idx) is int:
      idx = torch.tensor(idx).to(coords.device, dtype=torch.int64)
    batch_size, pt_num = coords.size(0), coords.size(1)

    mask = torch.tensor([0, 1], dtype=torch.float32, device=coords.device)
    mask = torch.meshgrid([mask]*self.in_features)
    mask = torch.stack(mask, -1).view(1, 1, -1, self.in_features)
    mul = [self.code_num ** i for i in range(self.in_features)]
    mul = torch.tensor(mul, dtype=torch.int64, device=coords.device)
    mul1 = (idx * (self.code_num ** self.in_features)).view(-1, 1, 1, 1)

    coords = (coords + 1.0) * ((self.code_num - 1) / 2.0) # [-1, 1] -> [0, code_num-1]
    corners = torch.floor(coords).detach()    # [B, N, D]
    corners = corners.unsqueeze(-2) + mask    # [B, N, 8/4, D]
    coordsf = coords.unsqueeze(-2) - corners  # [B, N, 8/4, D], local coords [-1, 1]

    index = corners.to(torch.int64) * mul     # [B, N, 8/4, D]
    index = torch.sum(index, -1)              # [B, N, 8/4]
    index = index + mul1                      # [B, N, 8/4]
    weights = 1.0 - torch.abs(coordsf)        # [B, N, 8/4, D]
    weights = torch.prod(weights, axis=-1)    # [B, N, 8/4]

    coords_code = torch.index_select(self.shape_code, 1, index.view(-1))
    coords_code = coords_code.view(self.out_features, batch_size, pt_num, -1)
    output = torch.sum(coords_code * weights, dim=-1)      # [C, B, N]
    output = output.permute(1, 2, 0)                       # [B, N, C]
    return output

  def get_shape_code(self, idx):
    shape_code = self.shape_code.view(self.out_features, self.shape_num, -1)
    output = torch.index_select(shape_code, 1, idx).view(self.out_features, -1)
    return output

  def get_mean_code(self):
    return self.shape_code.view(self.out_features, self.shape_num, -1).mean(1)

  def _align_corners(self):
    return True

  def upsample(self, size=64):
    code_size = [self.out_features, self.shape_num] + [self.code_num] * self.in_features
    shape_code = self.shape_code.view(*code_size)
    shape_code = torch.transpose(shape_code, 0, 1)
    output = torch.nn.functional.upsample(shape_code, size=size, mode='trilinear', 
                                          align_corners=self._align_corners())
    output = torch.transpose(output, 0, 1)
    output = output.reshape(self.out_features, -1)
    return output

  def extra_repr(self) -> str:
    return 'in_features={}, code_channel={}, code_num={}, shape_num={}'.format(
        self.in_features, self.out_features, self.code_num, self.shape_num)


class OptPosEncVol2(OptPosEncVol):
  def forward(self, coords, idx=0):
    if type(idx) is int:
      idx = torch.tensor(idx).to(coords.device, dtype=torch.int64)
    batch_size, pt_num = coords.size(0), coords.size(1)

    mask = torch.tensor([-1, 0, 1], dtype=torch.float32, device=coords.device)
    mask = torch.meshgrid([mask]*self.in_features)
    mask = torch.stack(mask, -1).view(1, 1, -1, self.in_features)
    mul = [self.code_num ** i for i in range(self.in_features)]
    mul = torch.tensor(mul, dtype=torch.int64, device=coords.device)
    mul1 = (idx * (self.code_num ** self.in_features)).view(-1, 1, 1, 1)
   
    coords = (coords * 0.8 + 1.0) * (self.code_num / 2.0) # [-1, 1] -> [0, code_num]
    coords = coords - 0.5           # Suppose the code is defined on the grid center
    corners = torch.round(coords).detach()    # [B, N, D]
    corners = corners.unsqueeze(-2) + mask    # [B, N, K, D], K = 9 or 27
    coordsf = coords.unsqueeze(-2) - corners  # [B, N, K, D], local coords [-1.5, 1.5]

    index = corners.to(torch.int64) * mul     # [B, N, K, D]
    index = torch.sum(index, -1)              # [B, N, K]
    index = torch.clamp(index, 0, self.code_num ** self.in_features - 1)
    index = index + mul1                      # [B, N, K]
    weights = spline_basis(coordsf)           # [B, N, K, D]
    weights = torch.prod(weights, axis=-1)    # [B, N, K]

    coords_code = torch.index_select(self.shape_code, 1, index.view(-1))
    coords_code = coords_code.view(self.out_features, batch_size, pt_num, -1)
    output = torch.sum(coords_code * weights, dim=-1)      # [C, B, N]
    output = output.permute(1, 2, 0)                       # [B, N, C]
    return output

  def _align_corners(self):
    return False


class MLPSpace(nn.Module):
  def __init__(self, in_features=3, out_features=1, num_hidden_layers=3,
              hidden_features=256, name='mlp', **kwargs):
    super().__init__()
    Net = Siren if name == 'siren' else MLP
    enc_type = OptPosEncBatch if name == 'optpos' else GlobalPosEnc
    self.pos_enc = enc_type(in_features, **kwargs)
    self.net = Net(self.pos_enc.out_features, out_features, num_hidden_layers,
                   hidden_features)

  def forward(self, input, idx=0):
    enc = self.pos_enc(input, idx)
    output = self.net(enc)
    return output


class PosProj(nn.Module):
  def __init__(self, in_dim, out_dim):
    super().__init__()
    assert(out_dim > in_dim)
    self.in_dim = in_dim
    self.out_dim = out_dim
    self.channel = out_dim - in_dim

    # self.proj = nn.Parameter(torch.Tensor(in_dim, self.channel))
    self.register_buffer('proj', torch.Tensor(in_dim, self.channel))
    self.reset_parameters()

  def reset_parameters(self):
    with torch.no_grad():
      proj = torch.randn_like(self.proj)
      scale = self.in_dim ** 0.5 # TODO: use small scale
      scale = torch.norm(proj, dim=0, keepdim=True) * scale
      proj = proj / (1.0e-6 + scale)
      self.proj.copy_(proj)

  def forward(self, coords):
    proj = torch.flatten(coords, end_dim=1).mm(self.proj)
    proj = proj.view(list(coords.size())[:-1] + [self.channel])
    output = torch.cat([coords, proj], dim=-1)
    return output

  def extra_repr(self) -> str:
    return "in_dim={}, out_dim={}".format(self.in_dim, self.out_dim)


def make_mlp_model(flags):
  # make the activation function
  if flags.activation.lower() == 'relu':
    act = nn.ReLU(inplace=True)
  elif flags.activation.lower() == 'softplus':
    act = nn.Softplus(beta=100)
  else:
    raise NotImplementedError

  # make the mlp
  if flags.name == 'siren':
    return Siren(**flags)
  elif flags.name == 'mlp':
    return MLP(**flags, act=act)
  elif flags.name == 'nerf':
    return PosEncMLP(**flags, pos_enc=FourierFeat, act=act)
  elif flags.name == 'randfourier':
    return PosEncMLP(**flags, pos_enc=RandFourier, act=act)
  elif flags.name == 'optpos':
    return PosEncMLP(**flags, pos_enc=OptPosEnc, act=act)
  elif flags.name == 'optpos2':
    return PosEncMLP(**flags, pos_enc=OptPosEncBatch2, act=act)
  elif flags.name == 'optposvol':
    return PosEncMLP(**flags, pos_enc=OptPosEncVol, act=act)
  elif flags.name == 'optposvol2':
    return PosEncMLP(**flags, pos_enc=OptPosEncVol2, act=act)
  elif flags.name == 'pesiren':
    return PosEncSiren(**flags)
  else:
    raise NotImplementedError


if __name__ == '__main__':
  model = Siren()
  print(model)
