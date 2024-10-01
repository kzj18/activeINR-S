#!/usr/bin/env python
import random
from typing import Dict, List, Union, Tuple
from enum import Enum

import numpy as np
import torch
import torch.optim as optim
from imgviz import depth2rgb
import cv2
import open3d as o3d

import rospy

from scripts import start_timing, end_timing
from scripts.dataloader import RGBDSensor
from scripts.mapper import select_samples, coordinates, get_pose_representation, get_pose_param_optim, getVoxels, get_batch_query_fn, sdf_to_vertices_faces
from scripts.mapper.model import gradient, subtract_params, set_net_params, param_norm, unit_params, scale_params_, sum_params
from scripts.mapper.model.keyframe import KeyFrameDatabaseFlexible
from scripts.mapper.model.scene_rep import JointEncoding

class MapperState(Enum):
    BOOTSTRAP = 0
    KEYFRAME = 1
    MAPPING = 2
    KEYFRAME_MAPPING = 3
    IDLE = 4
    POST_PROCESSING = 5
    
class MeshColorType(Enum):
    NONE = 'NONE'
    RENDER_SURFACE_COLOR = 'RENDER_SURFACE_COLOR'
    QUERY_VERTICES_COLOR = 'QUERY_VERTICES_COLOR'
    
class MapperInfo():
    
    def __init__(self) -> None:
        self.__mapper_loss_array:List[List[torch.Tensor]] = []
        self.__mapper_time_array:List[List[float]] = []
        
    def update(self, idx:int, loss:List[torch.Tensor], time:List[float]) -> None:
        if idx >= len(self.__mapper_loss_array):
            self.__mapper_loss_array.extend([[] for _ in range(idx - len(self.__mapper_loss_array) + 1)])
            self.__mapper_time_array.extend([[] for _ in range(idx - len(self.__mapper_time_array) + 1)])
        self.__mapper_loss_array[idx].extend(loss)
        self.__mapper_time_array[idx].extend(time)
        
    @staticmethod
    def get_loss_from_ret(ret:Dict[str, torch.Tensor], loss:torch.Tensor) -> dict:
        return {
            'rgb_loss': ret['rgb_loss'].detach().cpu().numpy(),
            'depth_loss': ret['depth_loss'].detach().cpu().numpy(),
            'sdf_loss': ret['sdf_loss'].detach().cpu().numpy(),
            'fs_loss': ret['fs_loss'].detach().cpu().numpy(),
            'total_loss': loss.detach().cpu().numpy()
        }

class Mapper():
    def __init__(
        self,
        config:dict,
        bounding_box:torch.Tensor,
        topdown_voxel_grid:torch.Tensor,
        precise_voxel_grid:torch.Tensor,
        rgbd_sensor:RGBDSensor,
        device:torch.device) -> None:
        self.__device = device
        self.__device_o3c = o3d.core.Device(self.__device.type, self.__device.index)
        self.__bbox = bounding_box
        self.__topdown_voxel_grid = topdown_voxel_grid
        self.__precise_voxel_grid = precise_voxel_grid
        self.__rgbd_sensor = rgbd_sensor
        self.__directions = rgbd_sensor.directions[None, ...].to(self.__device)
        
        self.__downsample_render = config['painter']['render_rgbd_downsample']
        if self.__downsample_render > 1:
            self.__rgbd_sensor_render = RGBDSensor(
                height=rgbd_sensor.height,
                width=rgbd_sensor.width,
                fx=rgbd_sensor.fx,
                fy=rgbd_sensor.fy,
                cx=rgbd_sensor.cx,
                cy=rgbd_sensor.cy,
                depth_min=rgbd_sensor.depth_min,
                depth_max=rgbd_sensor.depth_max,
                depth_scale=rgbd_sensor.depth_scale,
                position=rgbd_sensor.position,
                downsample_factor=self.__downsample_render)
        else:
            self.__rgbd_sensor_render = rgbd_sensor
        self.__rays_d_cam_render = self.__rgbd_sensor_render.directions.to(self.__device).reshape(-1, 3)
        
        self.__sc_factor = config['dataset']['sc_factor']
        self.__mesh_translation = torch.Tensor(config['dataset']['translation']).to(self.__device)
        
        self.__kf_every = config['mapper']['keyframe_every']
        self.__map_every = config['mapper']['mapping_every']
        kf_num_max = config['dataset']['step_num'] // self.__kf_every
        
        self.__filter_depth = config['mapper']['filter_depth']
        self.__pose_accum_step = config['mapper']['pose_accum_step']
        self.__map_wait_step = config['mapper']['map_wait_step']
        self.__map_accum_step = config['mapper']['map_accum_step']
        self.__min_pixels_cur = config['mapper']['min_pixels_cur']
        self.__loss_rgb_weight = config['mapper']['loss']['rgb_weight']
        self.__loss_depth_weight = config['mapper']['loss']['depth_weight']
        self.__loss_sdf_weight = config['mapper']['loss']['sdf_weight']
        self.__loss_fs_weight = config['mapper']['loss']['fs_weight']
        self.__loss_smooth_weight = config['mapper']['loss']['smooth_weight']
        self.__loss_smooth_pts = config['mapper']['loss']['smooth_pts']
        self.__loss_smooth_vox = config['mapper']['loss']['smooth_vox']
        self.__loss_smooth_margin = config['mapper']['loss']['smooth_margin']
        self.__one_grid = config['mapper']['scene_rep']['grid']['oneGrid']
        self.__lr_decoder = config['mapper']['optimizer']['lr_decoder']
        self.__lr_embed = config['mapper']['optimizer']['lr_embed']
        self.__lr_embed_color = config['mapper']['optimizer']['lr_embed_color']
        self.__num_perturb = config['mapper']['uncertainty']['num_perturb']
            
        self.__optimizer_config = {
            'enable': config['mapper']['optimizer']['pose']['enable'],
            'lr_rot': config['mapper']['optimizer']['pose']['lr_rot'],
            'lr_trans': config['mapper']['optimizer']['pose']['lr_trans']
        }
        
        self.__mapper_info = MapperInfo()
        
        self.__model = JointEncoding(
            config,
            self.__bbox,
            self.__rgbd_sensor.depth_max).to(self.__device)
        self.__query_sdf_batch = get_batch_query_fn(self.__model.query_sdf, device=self.__device)
        self.__query_color_batch = get_batch_query_fn(self.__model.query_color, 1, device=self.__device)
        self.__render_surface_color_batch = get_batch_query_fn(self.__model.render_surface_color, 2, device=self.__device)
        self.__query_chunk_size = 1024 * 64
        
        self.__kf_database = KeyFrameDatabaseFlexible(
            self.__rgbd_sensor,
            config['mapper']['keyframe_rays_store_ratio'],
            self.__device)
        
        self.__create_optimizer()
        self.__matrix_to_tensor, self.__matrix_from_tensor = get_pose_representation(config['mapper']['pose']['rot_rep'])
        self.__optimize_cur = config['mapper']['pose']['optim_cur']
        
        self.__mapping_idx = None
        self.__mesh_voxels = None
        self.__last_net_params = None
        self.__tracking_idx = 0
        
        self.__mapping_iters = config['mapper']['mapping_iters']
        self.__mapping_first_iters = config['mapper']['mapping_first_iters']
        self.__mapping_iters_counter = 0
        self.__mapping_sample = config['mapper']['mapping_sample']

        self.__est_c2w_data = None
        
        # XXX: The bounding box for training has to rotate 180 degree around x-axis, then when we get the mesh, we rotate it back, i do not know the reason
        self.__construct_mesh_transform_matrix = np.eye(4)
        self.__construct_mesh_transform_matrix[1, 1] = -1
        self.__construct_mesh_transform_matrix[2, 2] = -1
        self.__construct_mesh_transform_matrix_o3c = o3d.core.Tensor(self.__construct_mesh_transform_matrix, device=self.__device_o3c)
        
    def __create_optimizer(self):
        '''
        Create optimizer for mapping
        '''
        trainable_parameters = [
            {'params': self.__model.decoder.parameters(), 'weight_decay': 1e-6, 'lr': self.__lr_decoder},
            {'params': self.__model.embed_fn.parameters(), 'eps': 1e-15, 'lr': self.__lr_embed}
        ]
    
        if not self.__one_grid:
            trainable_parameters.append({'params': self.__model.embed_fn_color.parameters(), 'eps': 1e-15, 'lr': self.__lr_embed_color})
        
        self.__map_optimizer = optim.Adam(trainable_parameters, betas=(0.9, 0.99))
        
    def __get_loss_from_ret(self, ret, rgb=True, sdf=True, depth=True, fs=True, smooth=False):
        '''
        Get the training loss
        '''
        loss = 0
        if rgb:
            loss += self.__loss_rgb_weight * ret['rgb_loss']
        if depth:
            loss += self.__loss_depth_weight * ret['depth_loss']
        if sdf:
            loss += self.__loss_sdf_weight * ret['sdf_loss']
        if fs:
            loss +=  self.__loss_fs_weight * ret['fs_loss']
        
        if smooth:
            loss += self.__loss_smooth_weight * self.__smoothness(
                self.__loss_smooth_pts,
                self.__loss_smooth_vox,
                margin=self.__loss_smooth_margin)
        
        return loss            

    def __smoothness(self, sample_points=256, voxel_size=0.1, margin=0.05):
        '''
        Smoothness loss of feature grid
        '''
        volume = self.__bbox[:, 1] - self.__bbox[:, 0]

        grid_size = (sample_points-1) * voxel_size
        offset_max = self.__bbox[:, 1]-self.__bbox[:, 0] - grid_size - 2 * margin

        offset = torch.rand(3).to(offset_max) * offset_max + margin
        coords = coordinates(sample_points - 1, 'cpu', flatten=False).float().to(volume)
        pts = (coords + torch.rand((1,1,1,3)).to(volume)) * voxel_size + self.__bbox[:, 0] + offset

        pts_tcnn = self.__model.tcnn_encoding(pts)

        sdf = self.__model.query_sdf(pts_tcnn, embed=True)
        tv_x = torch.pow(sdf[1:,...]-sdf[:-1,...], 2).sum()
        tv_y = torch.pow(sdf[:,1:,...]-sdf[:,:-1,...], 2).sum()
        tv_z = torch.pow(sdf[:,:,1:,...]-sdf[:,:,:-1,...], 2).sum()

        loss = (tv_x + tv_y + tv_z)/ (sample_points**3)

        return loss
    
    def __first_frame_mapping(self, batch:Dict[str, torch.Tensor]):
        '''
        First frame mapping
        Params:
            batch['c2w']: [1, 4, 4]
            batch['rgb']: [1, H, W, 3]
            batch['depth']: [1, H, W, 1]
        Returns:
            status: List[Tuple[dict, float, float]]
                ret: dict
                loss: float
                step_time: float
        '''
        timing_first_frame = start_timing()
        rospy.logdebug('First frame mapping...')
        assert batch['frame_id'] == self.__mapping_idx == 0, f'Everything must be 0, but got batch frame id {batch["frame_id"]} and mapping idx {self.__mapping_idx}'

        directions = self.__directions.reshape(batch['rgb'].shape)
        
        self.__model.train()

        # Training
        status_loss = []
        status_time = []
        for i in range(self.__mapping_first_iters):
            timing_iter = start_timing()
            self.__map_optimizer.zero_grad()
            indice = select_samples(self.__rgbd_sensor.height, self.__rgbd_sensor.width, self.__mapping_sample)
            indice_h, indice_w = indice % (self.__rgbd_sensor.height), indice // (self.__rgbd_sensor.height)
            rays_d_cam = directions[indice_h, indice_w, :]
            target_s = batch['rgb'][indice_h, indice_w, :]
            target_d = batch['depth'][indice_h, indice_w].unsqueeze(-1)

            rays_o = batch['c2w'][None, :3, -1].repeat(self.__mapping_sample, 1)
            rays_d = torch.sum(rays_d_cam[..., None, :] * batch['c2w'][:3, :3], -1)

            # Forward
            ret = self.__model.forward(rays_o, rays_d, target_s, target_d)
            ret:Dict[str, torch.Tensor]
            loss = self.__get_loss_from_ret(ret)
            loss:torch.Tensor
            loss.backward()
            self.__map_optimizer.step()
            status_time.append(end_timing(*timing_iter))
            status_loss.append(self.__mapper_info.get_loss_from_ret(ret, loss))
        
        self.__mapping_iters_counter += self.__mapping_first_iters
        
        self.__mapper_info.update(self.__mapping_idx, status_loss, status_time)
        rospy.logdebug(f'First frame mapping done, used {end_timing(*timing_first_frame):.2f} ms')

    def __global_BA(self, batch:Union[Dict[str, torch.Tensor], None], cur_frame_id:int=None):
        '''
        Global bundle adjustment that includes all the keyframes and the current frame
        Params:
            batch['c2w']: ground truth camera pose [1, 4, 4]
            batch['rgb']: rgb image [1, H, W, 3]
            batch['depth']: depth image [1, H, W, 1]
            cur_frame_id: current frame id
        '''
        self.__last_net_params = list(self.__model.sdf_net.parameters())
        timing_global_BA = start_timing()
        if cur_frame_id is None:
            # NOTE: cannot use self.__mapping_idx instead of cur_frame_id in this function, or it wiil be less flexible
            cur_frame_id = self.__mapping_idx
        pose_optimizer = None
        
        if batch is not None:
            current_rays = torch.cat([self.__directions, batch['rgb'], batch['depth'][..., None]], dim=-1)
            current_rays = current_rays.reshape(-1, current_rays.shape[-1])
            optimize_cur = self.__optimize_cur
        else:
            current_rays = None
            optimize_cur = False

        # all the KF poses: 0, 5, 10, ...
        poses = self.__est_c2w_data[self.__kf_database.get_frame_ids()]
        
        kf_num = self.__kf_database.get_kf_num()

        if kf_num < 2:
            poses_fixed = torch.nn.parameter.Parameter(poses).to(self.__device)
            current_pose = self.__est_c2w_data[cur_frame_id][None,...]
            poses_all = torch.cat([poses_fixed, current_pose], dim=0)
        
        else:
            poses_fixed = torch.nn.parameter.Parameter(poses[:1]).to(self.__device)
            current_pose = self.__est_c2w_data[cur_frame_id][None,...]

            if optimize_cur:
                cur_rot, cur_trans, pose_optimizer = get_pose_param_optim(
                    self.__optimizer_config,
                    torch.cat([poses[1:], current_pose]),
                    self.__matrix_to_tensor)
                pose_optim = self.__matrix_from_tensor(cur_rot, cur_trans).to(self.__device)
                poses_all = torch.cat([poses_fixed, pose_optim], dim=0)

            else:
                cur_rot, cur_trans, pose_optimizer = get_pose_param_optim(
                    self.__optimizer_config,
                    poses[1:],
                    self.__matrix_to_tensor)
                pose_optim = self.__matrix_from_tensor(cur_rot, cur_trans).to(self.__device)
                poses_all = torch.cat([poses_fixed, pose_optim, current_pose], dim=0)
        
        # Set up optimizer
        self.__map_optimizer.zero_grad()
        if pose_optimizer is not None:
            pose_optimizer.zero_grad()

        # Training
        status_loss = []
        status_time = []
        for i in range(self.__mapping_iters):
            timing_iter = start_timing()

            # Sample rays with real frame ids
            # rays [bs, 7]
            # frame_ids [bs]
            rays, ids_all = self.__kf_database.sample_global_rays(self.__mapping_sample)

            #TODO: Checkpoint...
            if batch is not None:
                idx_cur = random.sample(range(0, self.__rgbd_sensor.height * self.__rgbd_sensor.width),max(self.__mapping_sample // kf_num, self.__min_pixels_cur))
                current_rays_batch = current_rays[idx_cur, :]

                rays = torch.cat([rays, current_rays_batch], dim=0) # N, 7
                ids_all = torch.cat([ids_all, -torch.ones((len(idx_cur)))]).type(torch.int64)

            rays_d_cam = rays[..., :3]
            target_s = rays[..., 3:6]
            target_d = rays[..., 6:7]

            # [N, Bs, 1, 3] * [N, 1, 3, 3] = (N, Bs, 3)
            rays_d = torch.sum(rays_d_cam[..., None, None, :] * poses_all[ids_all, None, :3, :3], -1)
            rays_o = poses_all[ids_all, None, :3, -1].repeat(1, rays_d.shape[1], 1).reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)


            ret = self.__model.forward(rays_o, rays_d, target_s, target_d)
            ret:Dict[str, torch.Tensor]

            loss = self.__get_loss_from_ret(ret, smooth=True)
            loss:torch.Tensor
            
            loss.backward(retain_graph=True)
            
            if (i + 1) % self.__map_accum_step == 0:
               
                if (i + 1) > self.__map_wait_step:
                    self.__map_optimizer.step()
                else:
                    rospy.logdebug('Wait update')
                self.__map_optimizer.zero_grad()

            if pose_optimizer is not None and (i + 1) % self.__pose_accum_step == 0:
                pose_optimizer.step()
                # get SE3 poses to do forward pass
                pose_optim = self.__matrix_from_tensor(cur_rot, cur_trans)
                pose_optim = pose_optim.to(self.__device)
                # So current pose is always unchanged
                if optimize_cur:
                    poses_all = torch.cat([poses_fixed, pose_optim], dim=0)
                
                else:
                    current_pose = self.__est_c2w_data[cur_frame_id][None,...]
                    # SE3 poses

                    poses_all = torch.cat([poses_fixed, pose_optim, current_pose], dim=0)


                # zero_grad here
                pose_optimizer.zero_grad()
                
            status_time.append(end_timing(*timing_iter))
            status_loss.append(self.__mapper_info.get_loss_from_ret(ret, loss))
                
        self.__mapping_iters_counter += self.__mapping_iters
        
        self.__mapper_info.update(cur_frame_id, status_loss, status_time)
        
        if pose_optimizer is not None and kf_num > 1:
            for i in range(kf_num - 1):
                self.__est_c2w_data[int(self.__kf_database.get_frame_ids()[i+1].item())] = self.__matrix_from_tensor(cur_rot[i:i+1], cur_trans[i:i+1]).detach().clone()[0]
        
            if optimize_cur:
                rospy.logdebug('Update current pose')
                self.__est_c2w_data[cur_frame_id] = self.__matrix_from_tensor(cur_rot[-1:], cur_trans[-1:]).detach().clone()[0]
                
        rospy.logdebug(f'Global BA done, used {end_timing(*timing_global_BA):.2f} ms')
        
    def __attach_est_c2w_data(self, c2w:torch.Tensor) -> None:
        if self.__est_c2w_data is None:
            self.__est_c2w_data = c2w.unsqueeze(0)
        else:
            self.__est_c2w_data = torch.cat([self.__est_c2w_data, c2w.unsqueeze(0)], dim=0)
        
    def run(
        self,
        batch:Union[Dict[str, torch.Tensor], None],
        kf_trigger_flag:bool=False) -> MapperState:
        if self.__mapping_idx is not None:
            mapping_idx_cur = self.__mapping_idx + self.__map_every
        
        if batch is None:
            batch_copy = None
            mapper_state = MapperState.POST_PROCESSING
            kf_update_flag = False
        else:
            assert batch['frame_id'] == self.__tracking_idx, f'Frame id must be the same, but got {batch["frame_id"]} and {self.__tracking_idx}'
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.__device)
            # TODO: self.__est_c2w_data store all c2w, check if it is correct, this may cost extra memory
            self.__attach_est_c2w_data(batch['c2w'])
            assert len(self.__est_c2w_data) == batch['frame_id'] + 1, f'Est c2w data must be the same, but got {len(self.__est_c2w_data)} and {batch["frame_id"] + 1}'
            batch_copy = batch.copy()
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch_copy[k] = v[None, ...]
                else:
                    batch_copy[k] = torch.tensor([v])
            kf_update_flag = True if self.__tracking_idx % self.__kf_every == 0 else kf_trigger_flag
            if kf_trigger_flag:
                mapping_iters_cur = int(self.__mapping_iters)
                mapping_first_iters_cur = int(self.__mapping_first_iters)
                self.__mapping_iters = 100
                self.__mapping_first_iters = mapping_first_iters_cur
            self.__tracking_idx += 1
            if self.__mapping_idx is None:
                mapper_state = MapperState.BOOTSTRAP
                self.__mapping_idx = 0
            elif self.__tracking_idx > mapping_idx_cur:
                self.__mapping_idx = mapping_idx_cur
                if kf_update_flag:
                    mapper_state = MapperState.KEYFRAME_MAPPING
                else:
                    mapper_state = MapperState.MAPPING
            elif kf_update_flag:
                mapper_state = MapperState.KEYFRAME
            else:
                mapper_state = MapperState.IDLE
            
        if mapper_state == MapperState.BOOTSTRAP:
            self.__first_frame_mapping(batch)
            # First frame will always be a keyframe
            self.__kf_database.add_keyframe(batch_copy, filter_depth=self.__filter_depth)
        elif mapper_state == MapperState.MAPPING:
            self.__global_BA(batch_copy, self.__mapping_idx)
        elif mapper_state == MapperState.KEYFRAME_MAPPING:
            if self.__optimize_cur:
                self.__global_BA(batch_copy, self.__mapping_idx)
                self.__kf_database.add_keyframe(batch_copy)
            else:
                self.__kf_database.add_keyframe(batch_copy)
                self.__global_BA(batch_copy, self.__mapping_idx)
        elif mapper_state == MapperState.KEYFRAME:
            self.__kf_database.add_keyframe(batch_copy)
        elif mapper_state == MapperState.IDLE:
            pass
        elif mapper_state == MapperState.POST_PROCESSING:
            self.__global_BA(batch_copy)
        else:
            raise NotImplementedError(f'Unsupported mapper state: {mapper_state}')
        
        if kf_trigger_flag and kf_update_flag:
            self.__mapping_iters = mapping_iters_cur
            self.__mapping_first_iters = mapping_first_iters_cur
        
        return mapper_state
    
    def get_mapper_info(self) -> MapperInfo:
        return self.__mapper_info
    
    def get_kf_every(self) -> int:
        return self.__kf_every
    
    def set_kf_every(self, kf_every:int) -> None:
        self.__kf_every = int(kf_every)
    
    def get_map_every(self) -> int:
        return self.__map_every
    
    def set_map_every(self, map_every:int) -> None:
        self.__map_every = int(map_every)
    
    def get_mapping_iters(self) -> int:
        return self.__mapping_iters
    
    def set_mapping_iters(self, mapping_iters:int) -> None:
        self.__mapping_iters = int(mapping_iters)
        
    def get_mapping_iters_counter(self) -> int:
        return self.__mapping_iters_counter
    
    def set_mesh_voxels(self, voxel_size:float) -> None:
        tx, ty, tz = getVoxels(
            *self.__bbox.detach().flatten().cpu().numpy(),
            voxel_size=voxel_size)
        
        self.__mesh_voxels:torch.Tensor = torch.stack(torch.meshgrid(tx, ty, tz, indexing='ij'), dim=-1).type(torch.float32)
        
        # Rescale and translate
        tx = tx.cpu().data.numpy()
        ty = ty.cpu().data.numpy()
        tz = tz.cpu().data.numpy()
        
        self.__sdf_config = {
            'scale': torch.Tensor([tx[-1] - tx[0], ty[-1] - ty[0], tz[-1] - tz[0]]),
            'offset': torch.Tensor([tx[0], ty[0], tz[0]]),
            'dim_num': torch.Tensor([len(tx), len(ty), len(tz)])
        }
    
    def query_mesh_voxels_sdf(self) -> torch.Tensor:
        return self.query_points_sdf(self.__mesh_voxels)
    
    @torch.no_grad()
    def query_points_sdf(self, points:torch.Tensor) -> torch.Tensor:
        points_shape = points.shape
        points_flatten = points.to(self.__device).reshape(-1, 3)
        
        points_flatten = self.__model.tcnn_encoding(points_flatten)
        
        raw = [self.__query_sdf_batch(points_flatten, i, i + self.__query_chunk_size) for i in range(0, points_flatten.shape[0], self.__query_chunk_size)]
        raw = torch.cat(raw, dim=0)
        return raw.reshape(points_shape[:-1] + (raw.shape[-1],)).squeeze().detach()
    
    def get_mesh(self, mesh_color_type:MeshColorType, require_vertices_uncertainty:bool=False) -> Tuple[o3d.geometry.TriangleMesh, torch.Tensor, torch.Tensor]:
        vertices_gradient = None
        vertices_uncertainty = None
        timing_query_mesh = start_timing()
        mesh_sdf = self.query_mesh_voxels_sdf()
        rospy.logdebug(f'Query mesh voxels took {end_timing(*timing_query_mesh):.2f} ms')
        timing_sdf_to_mesh = start_timing()
        vertices, faces = sdf_to_vertices_faces(
            mesh_sdf,
            self.__sdf_config,
            self.__sc_factor,
            self.__mesh_translation,
            self.__device)
        rospy.logdebug(f'SDF to mesh took {end_timing(*timing_sdf_to_mesh):.2f} ms')
        timing_colorize_mesh = start_timing()
        mesh = self.__vertices_faces_to_mesh(
            vertices,
            faces,
            mesh_color_type)
        rospy.logdebug(f'Colorize mesh took {end_timing(*timing_colorize_mesh):.2f} ms')
        if require_vertices_uncertainty:
            assert self.__last_net_params is not None, 'Last net params must be not None'
            timing_calculate_uncertainty = start_timing()
            # TODO: Uncertainty Calculation
            vertices.requires_grad_(True)
            vertices = self.__model.tcnn_encoding(vertices)
            vertices_sdf = self.__model.query_sdf(vertices)
            vertices_gradient = gradient(vertices, vertices_sdf)
            vertices_gradient = vertices_gradient / vertices_gradient.norm(dim=-1, keepdim=True)
            with torch.no_grad():
                current_net_params = list(self.__model.sdf_net.parameters())
                param_diff = subtract_params(self.__last_net_params, current_net_params, inplace=False)
                model_diff = set_net_params(self.__model, param_diff, inplace=False)
                vertices_uncertainty = []
                for _ in range(self.__num_perturb):
                    rand_unit_params = unit_params(model_diff.decoder.sdf_net.parameters())
                    scaled_params = scale_params_(rand_unit_params, 1)
                    perturbed_params = sum_params(current_net_params, scaled_params, inplace=False)
                    new_model = set_net_params(self.__model, perturbed_params, inplace=False)
                    vertices_sdf_perturbed = new_model.query_sdf(vertices)
                    vertices_uncertainty.append(torch.abs(vertices_sdf_perturbed - vertices_sdf))
                vertices_uncertainty = torch.stack(vertices_uncertainty)
                vertices_uncertainty = vertices_uncertainty.mean(dim=0)
            rospy.logdebug(f'Calculate uncertainty took {end_timing(*timing_calculate_uncertainty):.2f} ms')
        return mesh, vertices_gradient, vertices_uncertainty
        
    @torch.no_grad()
    def __vertices_faces_to_mesh(self, vertices:torch.Tensor, faces:np.ndarray, mesh_color_type:MeshColorType) -> o3d.geometry.TriangleMesh:
        mesh = o3d.t.geometry.TriangleMesh(
            o3d.core.Tensor(vertices.detach().cpu().numpy(), device=self.__device_o3c),
            o3d.core.Tensor(faces, device=self.__device_o3c))
        
        if mesh_color_type == MeshColorType.NONE:
            pass
        
        elif mesh_color_type == MeshColorType.QUERY_VERTICES_COLOR:
            vertices_shape = vertices.shape
            vertices_flatten = self.__model.tcnn_encoding(vertices)
            raw = [self.__query_color_batch(vertices_flatten, i, i + self.__query_chunk_size) for i in range(0, vertices_flatten.shape[0], self.__query_chunk_size)]
            raw = torch.cat(raw, dim=0)
            raw = raw.reshape(vertices_shape[:-1] + (-1,))
            # XXX: Open3d 0.18.0 has different API
            mesh.vertex['colors'] = o3d.core.Tensor(raw.detach().cpu().numpy(), device=self.__device_o3c)
        
        elif mesh_color_type == MeshColorType.RENDER_SURFACE_COLOR:
            mesh_o3d = mesh.to_legacy()
            mesh_o3d.compute_vertex_normals()
            vertices_normals = torch.from_numpy(np.array(mesh_o3d.vertex_normals)).to(self.__device)
            vertices_normals_shape = vertices_normals.shape
            raw = [self.__render_surface_color_batch(vertices_normals, vertices_normals, i, i + self.__query_chunk_size) for i in range(0, vertices_normals.shape[0], self.__query_chunk_size)]
            raw = torch.cat(raw, dim=0)
            raw = raw.reshape(vertices_normals_shape[:-1] + (-1,))
            mesh.vertex['colors'] = o3d.core.Tensor(raw.detach().cpu().numpy(), device=self.__device_o3c)
        
        else:
            raise NotImplementedError(f'Unsupported mesh color type: {mesh_color_type}')
        
        # XXX: The bounding box for training has to rotate 180 degree around x-axis, then when we get the mesh, we rotate it back, i do not know the reason
        mesh.transform(self.__construct_mesh_transform_matrix_o3c)
        return mesh.to_legacy()
    
    @torch.no_grad()
    def render_rgbd(self, batch:Dict[str, torch.Tensor]) -> np.ndarray:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.__device)
        target_d = batch['depth'][::self.__downsample_render, ::self.__downsample_render].reshape(-1, 1)
        rays_o = batch['c2w'][None, :3, -1].repeat(self.__rays_d_cam_render.shape[0], 1)
        rays_d = torch.sum(self.__rays_d_cam_render[..., None, :] * batch['c2w'][:3, :3], -1)
        
        rend_dict = self.__model.render_rays(rays_o, rays_d, target_d)
        
        color_numpy:np.ndarray = rend_dict['rgb'].detach().cpu().numpy()
        depth_numpy:np.ndarray = rend_dict['depth'].detach().cpu().numpy()
        
        color_numpy = np.uint8(color_numpy * 255)
        
        color_numpy = color_numpy.reshape(self.__rgbd_sensor_render.height, self.__rgbd_sensor_render.width, 3)
        depth_numpy = depth_numpy.reshape(self.__rgbd_sensor_render.height, self.__rgbd_sensor_render.width)
        
        color_numpy = cv2.resize(color_numpy, (self.__rgbd_sensor.width, self.__rgbd_sensor.height))
        depth_numpy = cv2.resize(depth_numpy, (self.__rgbd_sensor.width, self.__rgbd_sensor.height))
        
        return np.hstack([color_numpy, depth2rgb(depth_numpy, max_value=self.__rgbd_sensor.depth_max, min_value=self.__rgbd_sensor.depth_min)])
    
    def get_topdown_voxel_grid_sdf(self) -> torch.Tensor:
        return self.query_points_sdf(self.__topdown_voxel_grid)
    
    def get_precise_voxel_grid_sdf(self, center:np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        precise_voxel_grid = self.__precise_voxel_grid + torch.from_numpy(center).to(self.__device)
        return precise_voxel_grid, self.query_points_sdf(precise_voxel_grid)
    
    def is_ready_for_uncertainty(self) -> bool:
        return self.__last_net_params is not None
    
    def get_construct_mesh_transform_matrix(self) -> np.ndarray:
        return self.__construct_mesh_transform_matrix