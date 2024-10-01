import random
from typing import Tuple, Union, Callable, Dict

import numpy as np
import torch

import rospy

from scripts import start_timing, end_timing

from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix, rotation_6d_to_matrix, quaternion_to_axis_angle
from pytorch3d.ops.marching_cubes import marching_cubes

# NOTE: Functions for 3D
def coordinates(voxel_dim, device:torch.device, flatten:bool=True):
    if type(voxel_dim) is int:
        nx = ny = nz = voxel_dim
    else:
        nx, ny, nz = voxel_dim[0], voxel_dim[1], voxel_dim[2]
    x = torch.arange(0, nx, dtype=torch.long, device=device)
    y = torch.arange(0, ny, dtype=torch.long, device=device)
    z = torch.arange(0, nz, dtype=torch.long, device=device)
    x, y, z = torch.meshgrid(x, y, z, indexing="ij")

    if not flatten:
        return torch.stack([x, y, z], dim=-1)

    return torch.stack((x.flatten(), y.flatten(), z.flatten()))

def getVoxels(x_min, x_max, y_min, y_max, z_min, z_max, voxel_size=None, resolution=None):

    if not isinstance(x_max, float):
        x_min = float(x_min)
        x_max = float(x_max)
        y_min = float(y_min)
        y_max = float(y_max)
        z_min = float(z_min)
        z_max = float(z_max)
    
    if voxel_size is not None and resolution is None:
        Nx = round((x_max - x_min) / voxel_size + 0.0005)
        Ny = round((y_max - y_min) / voxel_size + 0.0005)
        Nz = round((z_max - z_min) / voxel_size + 0.0005)

        tx = torch.linspace(x_min, x_max, Nx + 1)
        ty = torch.linspace(y_min, y_max, Ny + 1)
        tz = torch.linspace(z_min, z_max, Nz + 1)
    elif voxel_size is None and resolution is not None:
        tx = torch.linspace(x_min, x_max, resolution)
        ty = torch.linspace(y_min, y_max, resolution)
        tz = torch.linspace(z_min, z_max, resolution)
    else:
        raise ValueError(f'voxel_size {voxel_size} and resolution {resolution} cannot be both None or both not None')

    return tx, ty, tz

def get_batch_query_fn(
    query_fn:Union[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
    num_args:int=1,
    device:torch.device=None) -> Union[Callable[[torch.Tensor, int, int], torch.Tensor], Callable[[torch.Tensor, torch.Tensor, int, int], torch.Tensor]]:

    if num_args == 1:
        fn:Callable[[torch.Tensor, int, int], torch.Tensor] = lambda f, i0, i1: query_fn(f[i0:i1, None, :].to(device))
    else:
        fn:Callable[[torch.Tensor, torch.Tensor, int, int], torch.Tensor] = lambda f, f1, i0, i1: query_fn(f[i0:i1, None, :].to(device), f1[i0:i1, :].to(device))

    return fn

def sdf_to_vertices_faces(sdf_torch:torch.Tensor, sdf_config:Dict[str, torch.Tensor], sc_factor:float, mesh_translation:torch.Tensor, device:torch.device, isolevel:float=0.0) -> Tuple[torch.Tensor, np.ndarray]:
    sdf_torch = sdf_torch.to(device).unsqueeze(0)
    for k, v in sdf_config.items():
        if isinstance(v, torch.Tensor):
            sdf_config[k] = v.to(device)
    timing_marching_cubes = start_timing()
    vertices, faces = marching_cubes(sdf_torch, isolevel, False)
    rospy.logdebug(f'Marching Cubes Torch took {end_timing(*timing_marching_cubes):.2f} ms')
    vertices = vertices[0]
    faces = faces[0]
    
    vertices[:, [0, 2]] = vertices[:, [2, 0]]
    vertices[:, :3] /= (sdf_config['dim_num'] - 1)
    vertices[:, :3] = sdf_config['scale'][None, :] * vertices[:, :3] + sdf_config['offset']
    vertices[:, :3] = vertices[:, :3] / sc_factor - mesh_translation
    return vertices, faces.detach().cpu().numpy()

# NOTE: Functions for SAMPLE
def select_samples(H:int, W:int, samples:int):
    '''
    randomly select samples from the image
    '''
    indice = random.sample(range(H * W), int(samples))
    indice = torch.tensor(indice)
    return indice

# NOTE: Functions for POSE
def get_pose_param_optim(
    pose_optimizer_config:dict,
    poses:torch.Tensor,
    matrix_to_tensor:Callable) -> Tuple[torch.nn.parameter.Parameter, torch.nn.parameter.Parameter, torch.optim.Optimizer]:
    cur_trans = torch.nn.parameter.Parameter(poses[:, :3, 3])
    cur_rot = torch.nn.parameter.Parameter(matrix_to_tensor(poses[:, :3, :3]))
    if pose_optimizer_config['enable']:
        pose_optimizer = torch.optim.Adam([
            {"params": cur_rot, "lr": pose_optimizer_config['lr_rot']},
            {"params": cur_trans, "lr": pose_optimizer_config['lr_trans']}
        ])
    else:
        pose_optimizer = None
    
    return cur_rot, cur_trans, pose_optimizer

def get_pose_representation(rot_rep:str) -> Tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]:
    '''
    Get the pose representation axis-angle or quaternion
    '''
    if rot_rep == 'axis_angle':
        return matrix_to_axis_angle, at_to_transform_matrix
    elif rot_rep == "quat":
        return matrix_to_quaternion, qt_to_transform_matrix
    else:
        raise NotImplementedError(f'Unsupported rotation representation: {rot_rep}')

def axis_angle_to_matrix(data:torch.Tensor) -> torch.Tensor:
    batch_dims = data.shape[:-1]

    theta = torch.norm(data, dim=-1, keepdim=True)
    omega = data / theta

    omega1 = omega[...,0:1]
    omega2 = omega[...,1:2]
    omega3 = omega[...,2:3]
    zeros = torch.zeros_like(omega1)

    K = torch.concat([torch.concat([zeros, -omega3, omega2], dim=-1)[...,None,:],
                      torch.concat([omega3, zeros, -omega1], dim=-1)[...,None,:],
                      torch.concat([-omega2, omega1, zeros], dim=-1)[...,None,:]], dim=-2)
    I = torch.eye(3).expand(*batch_dims,3,3).to(data)

    return I + torch.sin(theta).unsqueeze(-1) * K + (1. - torch.cos(theta).unsqueeze(-1)) * (K @ K)

def matrix_to_axis_angle(rot:torch.Tensor) -> torch.Tensor:
    """
    :param rot: [N, 3, 3]
    :return:
    """
    return quaternion_to_axis_angle(matrix_to_quaternion(rot))

def at_to_transform_matrix(rot:torch.Tensor, trans:torch.Tensor) -> torch.Tensor:
    """
    :param rot: axis-angle [bs, 3]
    :param trans: translation vector[bs, 3]
    :return: transformation matrix [b, 4, 4]
    """
    bs = rot.shape[0]
    T = torch.eye(4).to(rot)[None, ...].repeat(bs, 1, 1)
    R = axis_angle_to_matrix(rot)
    T[:, :3, :3] = R
    T[:, :3, 3] = trans
    return T

def qt_to_transform_matrix(rot:torch.Tensor, trans:torch.Tensor) -> torch.Tensor:
    """
    :param rot: axis-angle [bs, 3]
    :param trans: translation vector[bs, 3]
    :return: transformation matrix [b, 4, 4]
    """
    bs = rot.shape[0]
    T = torch.eye(4).to(rot)[None, ...].repeat(bs, 1, 1)
    R = quaternion_to_matrix(rot)
    T[:, :3, :3] = R
    T[:, :3, 3] = trans
    return T

def six_t_to_transform_matrix(rot:torch.Tensor, trans:torch.Tensor) -> torch.Tensor:
    """
    :param rot: 6d rotation [bs, 6]
    :param trans: translation vector[bs, 3]
    :return: transformation matrix [b, 4, 4]
    """
    bs = rot.shape[0]
    T = torch.eye(4).to(rot)[None, ...].repeat(bs, 1, 1)
    R = rotation_6d_to_matrix(rot)
    T[:, :3, :3] = R
    T[:, :3, 3] = trans
    return 