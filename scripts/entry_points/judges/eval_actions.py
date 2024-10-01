#!/usr/bin/env python
import os
WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
import sys
sys.path.append(WORKSPACE)
import argparse
import json
import time
from typing import List, Tuple, Union, Dict
from concurrent.futures import Future, ProcessPoolExecutor

import torch
import open3d as o3d
import trimesh
from tqdm import tqdm
import numpy as np
from scipy.spatial import KDTree
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
import imgviz

from scripts import PROJECT_NAME
from scripts.visualizer import rgbd_to_pointcloud
from scripts.dataloader import RGBDSensor, load_scene_mesh, dataset_config_to_ros
from scripts.dataloader.dataloader import get_dataset, HabitatDataset, GoatBenchDataset

def eval_action(
    result_pc:np.ndarray,
    gt_mesh_pc_trimesh:trimesh.PointCloud) -> np.ndarray:
    result_pc_kd_tree = KDTree(result_pc)
    distances, _ = result_pc_kd_tree.query(gt_mesh_pc_trimesh.vertices)
    return distances

def eval_actions(
    save_path:str,
    user_config:dict,
    config:dict,
    actions:List[int],
    device:torch.device,
    rgbd_downsample_factor:int=2) -> Tuple[float, float, float, float]:
    dataset:Union[HabitatDataset, GoatBenchDataset] = get_dataset(config, user_config, 'Eval')
    dataset_config = dataset.setup()
    gt_mesh_o3d, _ = load_scene_mesh(
        dataset_config['scene_mesh_url'],
        dataset_config['scene_mesh_transform'])
    gt_mesh_trimesh = trimesh.Trimesh(
        np.asarray(gt_mesh_o3d.vertices),
        np.asarray(gt_mesh_o3d.triangles))
    gt_mesh_samples:np.ndarray = trimesh.sample.sample_surface(gt_mesh_trimesh, 200000)[0]
    gt_mesh_pc_trimesh = trimesh.PointCloud(vertices=gt_mesh_samples)
    min_distances = np.ones(gt_mesh_samples.shape[0])
    min_distances_inf = np.inf * np.ones(gt_mesh_samples.shape[0])
    dataset_config_ros = dataset_config_to_ros(dataset_config)
    rgbd_sensor = RGBDSensor(
        height=dataset_config_ros.rgbd_height,
        width=dataset_config_ros.rgbd_width,
        fx=dataset_config_ros.rgbd_fx,
        fy=dataset_config_ros.rgbd_fy,
        cx=dataset_config_ros.rgbd_cx,
        cy=dataset_config_ros.rgbd_cy,
        depth_min=dataset_config_ros.rgbd_depth_min,
        depth_max=dataset_config_ros.rgbd_depth_max,
        depth_scale=dataset_config_ros.rgbd_depth_scale,
        position=np.array([
            dataset_config_ros.rgbd_position.x,
            dataset_config_ros.rgbd_position.y,
            dataset_config_ros.rgbd_position.z]),
        downsample_factor=rgbd_downsample_factor)
    device_o3c = o3d.core.Device(device.type, device.index)
    o3d_const_camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            rgbd_sensor.width,
            rgbd_sensor.height,
            rgbd_sensor.fx,
            rgbd_sensor.fy,
            rgbd_sensor.cx,
            rgbd_sensor.cy)
    o3d_const_camera_intrinsics_o3c = o3d.core.Tensor(
        o3d_const_camera_intrinsics.intrinsic_matrix,
        device=device_o3c)
    futures:List[Future] = []
    futures_eval_tqdm = tqdm(total=len(actions) + 1, desc='Evaluating actions')
    with ProcessPoolExecutor(max_workers=os.cpu_count() - 1) as executor:
        
        def eval_action_wrapper() -> Future:
            nonlocal dataset, rgbd_sensor, o3d_const_camera_intrinsics_o3c, device_o3c, gt_mesh_pc_trimesh, futures_eval_tqdm
            frame_current = dataset.get_frame()
            rgb_vis = imgviz.resize(
                np.uint8(frame_current['rgb'] * 255),
                width=rgbd_sensor.width,
                height=rgbd_sensor.height,
                interpolation='nearest')
            depth_data = imgviz.resize(
                frame_current['depth'],
                width=rgbd_sensor.width,
                height=rgbd_sensor.height,
                interpolation='nearest')
            result_pc_o3c = rgbd_to_pointcloud(
                rgb_vis,
                depth_data,
                frame_current['c2w'],
                o3d_const_camera_intrinsics_o3c,
                1000,
                np.inf,
                device_o3c)
            result_pc = result_pc_o3c.point.positions.numpy()
            
            future = executor.submit(
                eval_action,
                result_pc,
                gt_mesh_pc_trimesh)
            future.add_done_callback(lambda p: futures_eval_tqdm.update(1))
            return future
        
        futures.append(eval_action_wrapper())
        for action in tqdm(actions, desc='Replaying actions'):
            dataset._sim.step(action)
            futures.append(eval_action_wrapper())
            
        results = []
        for future in futures:
            result_distances = future.result()
            min_distances = np.minimum(min_distances, result_distances)
            min_distances_inf = np.minimum(min_distances_inf, result_distances)
            min_comp = np.mean(min_distances)
            min_comp_ratio = np.mean(np.float64(min_distances < 0.05))
            min_comp_inf = np.mean(min_distances_inf)
            min_comp_ratio_inf = np.mean(np.float64(min_distances_inf < 0.05))
            results.append((min_comp, min_comp_ratio, min_comp_inf, min_comp_ratio_inf))
        with open(save_path, 'w') as f:
            for result in results:
                f.write(f'{result[0]} {result[1]} {result[2]} {result[3]}\n')
    return results[-1]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=f'{PROJECT_NAME} action evaluator.')
    parser.add_argument('--save_path',
                        type=str,
                        required=True,
                        help='Save url (*.txt).')
    parser.add_argument('--config',
                        type=str,
                        required=True,
                        help='Input config url (*.json).')
    parser.add_argument('--user_config',
                        type=str,
                        required=True,
                        help='User config url (*.json).')
    parser.add_argument('--gpu_id',
                        type=int,
                        required=True,
                        help='Specify gpu id.')
    parser.add_argument('--actions',
                        type=str,
                        required=True,
                        help='Specify the actions to replay.')
    parser.add_argument('--force',
                        action='store_true',
                        help='Specify whether to force evaluation.')
    args = parser.parse_args()
    
    save_path = args.save_path
    if os.path.exists(save_path):
        if args.force:
            print(f'Force evaluation.')
            save_path_name:str = os.path.basename(save_path)
            save_path_bak_name_parts = save_path_name.split('.')
            save_path_bak_name_parts.insert(-1, time.strftime("%Y-%m-%d_%H-%M-%S"))
            save_path_bak_name = '.'.join(save_path_bak_name_parts)
            os.rename(
                save_path,
                os.path.join(os.path.dirname(save_path), save_path_bak_name))
        else:
            print(f'Evaluation already exists.')
            exit(0)
    
    with open(args.config) as f:
        config = json.load(f)
    
    with open(args.user_config) as f:
        user_config = json.load(f)
                
    with open(args.actions) as f:
        actions = f.readlines()
        actions = [int(action.strip()) for action in actions]
        
    if torch.cuda.is_available():
        device = torch.device('cuda', args.gpu_id)
    else:
        print('No GPU available.')
        device = torch.device('cpu')
    
    eval_actions(
        save_path,
        user_config,
        config,
        actions,
        device)