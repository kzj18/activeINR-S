#!/usr/bin/env python
from copy import deepcopy
from typing import List, Iterator
from math import exp, log, floor

# package imports
import torch
import torch.nn.functional as F
from torch.autograd import grad, Variable
import numpy as np

def mse2psnr(x):
    '''
    MSE to PSNR
    '''
    return -10. * torch.log(x) / torch.log(torch.Tensor([10.])).to(x)

def coordinates(voxel_dim, device: torch.device):
    '''
    Params: voxel_dim: int or tuple of int
    Return: coordinates of the voxel grid
    '''
    if type(voxel_dim) is int:
        nx = ny = nz = voxel_dim
    else:
        nx, ny, nz = voxel_dim[0], voxel_dim[1], voxel_dim[2]
    x = torch.arange(0, nx, dtype=torch.long, device=device)
    y = torch.arange(0, ny, dtype=torch.long, device=device)
    z = torch.arange(0, nz, dtype=torch.long, device=device)
    x, y, z = torch.meshgrid(x, y, z, indexing="ij")

    return torch.stack((x.flatten(), y.flatten(), z.flatten()))

def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    '''
    Params:
        bins: torch.Tensor, (Bs, N_samples)
        weights: torch.Tensor, (Bs, N_samples)
        N_importance: int
    Return:
        samples: torch.Tensor, (Bs, N_importance)
    '''
    # device = weights.get_device()
    device = weights.device
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True) # Bs, N_samples-2
    cdf = torch.cumsum(pdf, -1) 
    cdf = torch.cat([torch.zeros_like(cdf[..., :1], device=device), cdf], -1) # Bs, N_samples-1
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / N_importance, 1. - 0.5 / N_importance, steps=N_importance, device=device)
        u = u.expand(list(cdf.shape[:-1]) + [N_importance])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_importance], device=device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom, device=device), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples

def batchify(fn, chunk=1024*64):
        """Constructs a version of 'fn' that applies to smaller batches.
        """
        if chunk is None:
            return fn
        def ret(inputs, inputs_dir=None):
            if inputs_dir is not None:
                return torch.cat([fn(inputs[i:i+chunk], inputs_dir[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
            return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
        return ret

def get_masks(z_vals, target_d, truncation):
    '''
    Params:
        z_vals: torch.Tensor, (Bs, N_samples)
        target_d: torch.Tensor, (Bs,)
        truncation: float
    Return:
        front_mask: torch.Tensor, (Bs, N_samples)
        sdf_mask: torch.Tensor, (Bs, N_samples)
        fs_weight: float
        sdf_weight: float
    '''

    # before truncation
    front_mask = torch.where(z_vals < (target_d - truncation), torch.ones_like(z_vals), torch.zeros_like(z_vals))
    # after truncation
    back_mask = torch.where(z_vals > (target_d + truncation), torch.ones_like(z_vals), torch.zeros_like(z_vals))
    # valid mask
    depth_mask = torch.where(target_d > 0.0, torch.ones_like(target_d), torch.zeros_like(target_d))
    # Valid sdf regionn
    sdf_mask = (1.0 - front_mask) * (1.0 - back_mask) * depth_mask

    num_fs_samples = torch.count_nonzero(front_mask)
    num_sdf_samples = torch.count_nonzero(sdf_mask)
    num_samples = num_sdf_samples + num_fs_samples
    fs_weight = 1.0 - num_fs_samples / num_samples
    sdf_weight = 1.0 - num_sdf_samples / num_samples

    return front_mask, sdf_mask, fs_weight, sdf_weight

def compute_loss(prediction, target, loss_type='l2'):
    '''
    Params: 
        prediction: torch.Tensor, (Bs, N_samples)
        target: torch.Tensor, (Bs, N_samples)
        loss_type: str
    Return:
        loss: torch.Tensor, (1,)
    '''

    if loss_type == 'l2':
        return F.mse_loss(prediction, target)
    elif loss_type == 'l1':
        return F.l1_loss(prediction, target)

    raise Exception('Unsupported loss type')
    
def get_sdf_loss(z_vals, target_d, predicted_sdf, truncation, loss_type=None, grad=None):
    '''
    Params:
        z_vals: torch.Tensor, (Bs, N_samples)
        target_d: torch.Tensor, (Bs,)
        predicted_sdf: torch.Tensor, (Bs, N_samples)
        truncation: float
    Return:
        fs_loss: torch.Tensor, (1,)
        sdf_loss: torch.Tensor, (1,)
        eikonal_loss: torch.Tensor, (1,)
    '''
    front_mask, sdf_mask, fs_weight, sdf_weight = get_masks(z_vals, target_d, truncation)

    fs_loss = compute_loss(predicted_sdf * front_mask, torch.ones_like(predicted_sdf) * front_mask, loss_type) * fs_weight
    sdf_loss = compute_loss((z_vals + predicted_sdf * truncation) * sdf_mask, target_d * sdf_mask, loss_type) * sdf_weight

    if grad is not None:
        eikonal_loss = (((grad.norm(2, dim=-1) - 1) ** 2) * sdf_mask / sdf_mask.sum()).sum()
        return fs_loss, sdf_loss, eikonal_loss

    return fs_loss, sdf_loss

def gradient(inputs:torch.Tensor, outputs:torch.Tensor):
    d_points = torch.ones_like(
        outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0].detach()
    return points_grad

def subtract_params(param1:List[torch.Tensor], param2:List[torch.Tensor], inplace:bool=False):
    if not inplace:
        params = deepcopy(param1)
    else:
        params = param1
    for p, p2 in zip(params, param2):
        p.data.sub_(p2.data)
    return params

def set_net_params(model, params:List[torch.Tensor], inplace:bool=False):
    from scripts.mapper.model.scene_rep import JointEncoding
    if inplace:
        model_cp = model
    else:
        model_cp = deepcopy(model)
    model_cp:JointEncoding
    for p, new_p in zip(model_cp.decoder.sdf_net.parameters(), params):
        p.data.copy_(new_p.data)
    return model_cp

def param_norm(params:List[torch.Tensor]):
    return np.sqrt(sum([p.data.pow(2).sum().item() for p in params]))

def unit_params(like:Iterator[torch.Tensor]):
    new_params = [Variable(p.data.new(*p.size()).normal_(), requires_grad=True) for p in like]
    norm = param_norm(new_params)
    for p in new_params:
        p.data.div_(norm)
    return new_params

def scale_params(params:List[torch.Tensor], scale:float, inplace:bool=False):
    if not inplace:
        params = deepcopy(params)
    if scale != 1.0:
        for p in params:
            p.data.mul_(scale)
    return params

def scale_params_(params:List[torch.Tensor], scale:float):
    return scale_params(params, scale, True)

def sum_params(param1:List[torch.Tensor], param2:List[torch.Tensor], inplace:bool=False):
    if not inplace:
        params = deepcopy(param1)
    else:
        params = param1
    for p, p2 in zip(params, param2):
        p.data.add_(p2.data)
    return params