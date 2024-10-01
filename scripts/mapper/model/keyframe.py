import random
from typing import Dict

import torch

import rospy

from scripts.dataloader import RGBDSensor

class KeyFrameDatabaseFixed(object):
    def __init__(
        self,
        rgbd_sensor:RGBDSensor,
        keyframe_rays_store_ratio:float,
        keyframe_num_max:int,
        device:torch.device):
        self.__rgbd_sensor = rgbd_sensor
        self.__directions = rgbd_sensor.directions[None, ...].to(device)

        self.__total_pixels = self.__rgbd_sensor.height * self.__rgbd_sensor.width
        self.__num_rays_to_save = int(self.__total_pixels * keyframe_rays_store_ratio)
        rospy.logdebug(f'#Pixels to save: {self.__num_rays_to_save}')
        self.__kf_index = 0
        self.__kf_num_max = keyframe_num_max
        self.__rays = None
        self.__frame_ids = None
    
    def __sample_single_keyframe_rays(self, rays, option='random'):
        '''
        Sampling strategy for current keyframe rays
        '''
        if option == 'random':
            idxs = random.sample(range(0, self.__rgbd_sensor.height*self.__rgbd_sensor.width), self.__num_rays_to_save)
        elif option == 'filter_depth':
            valid_depth_mask = (rays[..., -1] > 0.0) & (rays[..., -1] <= self.__rgbd_sensor.depth_max)
            rays_valid = rays[valid_depth_mask, :]  # [n_valid, 7]
            num_valid = len(rays_valid)
            idxs = random.sample(range(0, num_valid), self.__num_rays_to_save)

        else:
            raise NotImplementedError()
        rays = rays[:, idxs]
        return rays
    
    def __attach_ids(self, frame_ids:torch.Tensor):
        '''
        Attach the frame ids to list
        '''
        if self.__frame_ids is None:
            self.__frame_ids = torch.zeros(self.__kf_num_max, dtype=frame_ids.dtype).to(frame_ids.device)
        self.__frame_ids[self.__kf_index % self.__kf_num_max] = frame_ids
            
    def __attach_rays(self, rays:torch.Tensor):
        '''
        Attach the rays to the list
        '''
        if self.__rays is None:
            self.__rays = torch.zeros(self.__kf_num_max, *rays.shape, dtype=rays.dtype).to(rays.device)
        self.__rays[self.__kf_index % self.__kf_num_max] = rays
    
    def add_keyframe(self, batch:Dict[str, torch.Tensor], filter_depth:bool=False):
        '''
        Add keyframe rays to the keyframe database
        '''
        # batch direction (Bs=1, H*W, 3)
        rays = torch.cat([self.__directions.reshape(batch['rgb'].shape), batch['rgb'], batch['depth'][..., None]], dim=-1)
        rays = rays.reshape(1, -1, rays.shape[-1])
        if filter_depth:
            rays = self.__sample_single_keyframe_rays(rays, 'filter_depth')
        else:
            rays = self.__sample_single_keyframe_rays(rays)
        
        if not isinstance(batch['frame_id'], torch.Tensor):
            batch['frame_id'] = torch.tensor([batch['frame_id']])

        self.__attach_ids(batch['frame_id'])

        # Store the rays
        self.__attach_rays(rays)
        if self.__kf_index > self.__kf_num_max:
            rospy.logwarn('Keyframe database is full. Start to overwrite the old keyframes.')
        self.__kf_index += 1
    
    def sample_global_rays(self, bs):
        '''
        Sample rays from self.rays as well as frame_ids
        '''
        kf_num = self.get_kf_num()
        idxs = torch.tensor(random.sample(range(kf_num * self.__num_rays_to_save), bs))
        sample_rays = self.__rays[:kf_num].reshape(-1, 7)[idxs]

        return sample_rays, idxs//self.__num_rays_to_save
    
    def get_frame_ids(self):
        return self.__frame_ids[:self.get_kf_num()]
        
    def get_kf_num(self):
        return min(self.__kf_index, self.__kf_num_max)
    
class KeyFrameDatabaseFlexible(object):
    def __init__(self, rgbd_sensor:RGBDSensor, keyframe_rays_store_ratio:float, device:torch.device):
        self.__rgbd_sensor = rgbd_sensor
        self.__directions = rgbd_sensor.directions[None, ...].to(device)
        self.__device = device

        self.__total_pixels = self.__rgbd_sensor.height * self.__rgbd_sensor.width
        self.__num_rays_to_save = int(self.__total_pixels * keyframe_rays_store_ratio)
        rospy.logdebug(f'#Pixels to save: {self.__num_rays_to_save}')
        self.__rays = None
        self.__frame_ids = None
    
    def __sample_single_keyframe_rays(self, rays, option='random'):
        '''
        Sampling strategy for current keyframe rays
        '''
        if option == 'random':
            idxs = random.sample(range(0, self.__rgbd_sensor.height*self.__rgbd_sensor.width), self.__num_rays_to_save)
        elif option == 'filter_depth':
            valid_depth_mask = (rays[..., -1] > 0.0) & (rays[..., -1] <= self.__rgbd_sensor.depth_max)
            rays_valid = rays[valid_depth_mask, :]  # [n_valid, 7]
            num_valid = len(rays_valid)
            idxs = random.sample(range(0, num_valid), self.__num_rays_to_save)

        else:
            raise NotImplementedError()
        rays = rays[:, idxs]
        return rays
    
    def __attach_ids(self, frame_ids):
        '''
        Attach the frame ids to list
        '''
        if self.__frame_ids is None:
            self.__frame_ids = frame_ids
        else:
            self.__frame_ids = torch.cat([self.__frame_ids, frame_ids], dim=0)
            
    def __attach_rays(self, rays:torch.Tensor):
        '''
        Attach the rays to the list
        '''
        if self.__rays is None:
            self.__rays = rays.unsqueeze(0).cpu()
        else:
            self.__rays = torch.cat([self.__rays, rays.unsqueeze(0).cpu()], dim=0)
    
    def add_keyframe(self, batch:Dict[str, torch.Tensor], filter_depth:bool=False):
        '''
        Add keyframe rays to the keyframe database
        '''
        # batch direction (Bs=1, H*W, 3)
        rays = torch.cat([self.__directions.reshape(batch['rgb'].shape), batch['rgb'], batch['depth'][..., None]], dim=-1)
        rays = rays.reshape(1, -1, rays.shape[-1])
        if filter_depth:
            rays = self.__sample_single_keyframe_rays(rays, 'filter_depth')
        else:
            rays = self.__sample_single_keyframe_rays(rays)
        
        if not isinstance(batch['frame_id'], torch.Tensor):
            batch['frame_id'] = torch.tensor([batch['frame_id']])

        self.__attach_ids(batch['frame_id'])

        # Store the rays
        self.__attach_rays(rays)
    
    def sample_global_rays(self, bs):
        '''
        Sample rays from self.rays as well as frame_ids
        '''
        kf_num = len(self.__frame_ids)
        idxs = torch.tensor(random.sample(range(kf_num * self.__num_rays_to_save), bs))
        sample_rays = self.__rays[:kf_num].reshape(-1, 7)[idxs]

        return sample_rays.to(self.__device), idxs//self.__num_rays_to_save
    
    def get_frame_ids(self):
        return self.__frame_ids
        
    def get_kf_num(self):
        return len(self.__frame_ids)
    
    # TODO: Use  torch split to implement the flexible keyframe database