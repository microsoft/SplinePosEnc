import torch
import numpy as np

def calc_gradient(y, x, grad_outputs=None):
  if grad_outputs is None:
    grad_outputs = torch.ones_like(y)
  grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
  return grad


def img_mse(output, gt): 
  return 0.5 * ((output - gt) ** 2).mean()


def img_psnr(mse): 
  return -10.0 * np.log10(2.0 * mse)


def sdf_loss(sdf_pred, coords, sdf_gt, normal_gt, normal_weight=1.0, grad_weight=0.1):
  gradient = calc_gradient(sdf_pred, coords)

  mask = sdf_gt != -1  # (B, N, 1)
  sdf_loss = sdf_pred[mask].abs().mean()
  inter_loss = torch.exp(-40 * torch.abs(sdf_pred[mask.logical_not()])).mean()
  mask = mask.squeeze(-1)
  normal_loss = (gradient[mask, :] - normal_gt[mask, :]).norm(2, dim=-1).mean()
  grad_loss = (gradient[mask.logical_not(), :].norm(2, dim=-1) - 1).abs().mean()

  losses = [sdf_loss * 10.0, inter_loss * 0.1, normal_loss * normal_weight, grad_loss * grad_weight]
  total_loss = torch.stack(losses).sum()
  names = ['sdf', 'inter', 'normal_constraint', 'grad_constraint', 'total_train_loss']
  loss_dict = dict(zip(names, losses + [total_loss]))
  return loss_dict


def poisson_loss(sdf_pred, coords, sdf_gt, normal_gt, normal_weight=1.0, grad_weight=0.1):
  gradient = calc_gradient(sdf_pred, coords)

  mask = sdf_gt != -1  # (B, N, 1)
  sdf_loss = sdf_pred[mask].abs().mean()
  inter_loss = torch.exp(-40 * torch.abs(sdf_pred[mask.logical_not()])).mean()
  mask = mask.squeeze(-1)
  normal_loss = (gradient[mask, :] - normal_gt[mask, :]).norm(2, dim=-1).mean()
  grad_loss = (gradient[mask.logical_not(), :]**2).mean() # the only difference with sdf_loss

  losses = [sdf_loss * 10.0, inter_loss * 0.1, normal_loss * normal_weight, grad_loss * grad_weight]
  total_loss = torch.stack(losses).sum()
  names = ['sdf', 'inter', 'normal_constraint', 'grad_constraint', 'total_train_loss']
  loss_dict = dict(zip(names, losses + [total_loss]))
  return loss_dict


def sdf_sphere_loss(sdf_pred, coords, sdf_gt, normal_gt, normal_weight=1.0, grad_weight=0.1):
  gradient = calc_gradient(sdf_pred, coords)

  with torch.no_grad():
    radius = 0.9
    length = torch.sqrt(torch.sum(coords ** 2, dim=-1, keepdim=True))
    gradient_gt = coords / (length + 1.0e-10)
    sdf = length - radius

  mask = sdf_gt != -1
  sdf_loss = sdf_pred[mask].abs().mean()
  inter_loss = (sdf_pred - sdf).abs().mean()
  mask = mask.squeeze(-1)
  normal_loss = (gradient[mask, :] - normal_gt[mask, :]).norm(2, dim=-1).mean()
  grad_loss = (gradient - gradient_gt).norm(2, dim=-1).mean()

  losses = [sdf_loss * 10.0, inter_loss * 0.1, normal_loss * normal_weight, grad_loss * grad_weight]
  total_loss = torch.stack(losses).sum()
  names = ['sdf', 'inter', 'normal_constraint', 'grad_constraint', 'total_train_loss']
  loss_dict = dict(zip(names, losses + [total_loss]))
  return loss_dict


class DiffClamp(torch.autograd.Function):
  """
  In the forward pass this operation behaves like torch.clamp. But in the
  backward pass its gradient is 1 everywhere, as if instead of clamp one had
  used the identity function.
  """
  @staticmethod
  def forward(ctx, input, min, max):
    return input.clamp(min, max)

  @staticmethod
  def backward(ctx, grad_output):
    return grad_output.clone(), None, None


def diff_clamp(input, min, max):
  return DiffClamp.apply(input, min, max)

def sdf_mae(sdf_pred, sdf_gt, min=-1, max=1):
  csdf_pred = diff_clamp(sdf_pred, min, max)
  csdf_gt = diff_clamp(sdf_gt, min, max)
  loss = torch.abs(csdf_pred - csdf_gt).mean()
  return {'total_train_loss' : loss}
