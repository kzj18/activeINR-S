#!/usr/bin/env python
import os
WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
import sys
sys.path.append(WORKSPACE)
import argparse
import json

from scripts import PROJECT_NAME
from scripts.configer import get_active_branch, get_repo_status
from scripts.entry_points.nodes import GetDatasetConfig, GetDatasetConfigResponse, GetDatasetConfigRequest

import rospy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=f'{PROJECT_NAME} configer node.')
    parser.add_argument('--launch',
                        type=str,
                        required=True,
                        help='Launch file name (*.launch).')
    parser.add_argument('--gpu_id',
                        type=int,
                        required=True,
                        help='Specify gpu id.')
    parser.add_argument('--user_config',
                        type=str,
                        required=True,
                        help='User config url (*.json).')
    parser.add_argument('--mode',
                        type=str,
                        required=True,
                        help='Specify the mode to start with.')
    parser.add_argument('--actions',
                        type=str,
                        required=True,
                        help='Specify the actions to replay.')
    parser.add_argument('--hide_mapper_windows',
                        type=int,
                        required=True,
                        help='Specify whether to hide mapper windows.')
    parser.add_argument('--hide_planner_windows',
                        type=int,
                        required=True,
                        help='Specify whether to hide planner windows.')
    parser.add_argument('--step_num',
                        type=int,
                        required=True,
                        help='Specify the step number.')
    
    parser.add_argument('--config',
                        type=str,
                        default=None,
                        help='Input config url (*.json).')
    parser.add_argument('--scene_id',
                        type=str,
                        default=None,
                        help='Specify test scene id.')
    parser.add_argument('--ros_dataloader',
                        type=int,
                        default=None,
                        help='Specify whether to use ros dataloader.')
    parser.add_argument('--parallelized',
                        type=int,
                        default=None,
                        help='Specify whether to use parallelized.')
    
    parser.add_argument('--camera_config_file_path',
                        type=str,
                        default=None,
                        help='Specify camera config file path.')
    parser.add_argument('--camera_launch_file_path',
                        type=str,
                        default=None,
                        help='Specify camera launch file path.')
    parser.add_argument('--serial_number',
                        type=str,
                        default=None,
                        help='Specify camera serial number.')
    
    args, ros_args = parser.parse_known_args()
    
    ros_args = dict([arg.split(':=') for arg in ros_args])
    
    rospy.init_node(ros_args['__name'], anonymous=True, log_level=rospy.INFO)
    
    launch_file_name:str = args.launch
    
    launch_command = f'roslaunch {PROJECT_NAME.lower()} {launch_file_name}'
    if launch_file_name == 'habitat.launch':
        assert args.config is not None, 'Config is required for habitat launch.'
        assert args.scene_id is not None, 'Scene id is required for habitat launch.'
        assert args.ros_dataloader is not None, 'ROS dataloader is required for habitat launch.'
        assert args.parallelized is not None, 'Parallelized is required for habitat launch.'
        launch_command += f' config:={args.config} scene_id:={args.scene_id} ros_dataloader:={args.ros_dataloader} parallelized:={args.parallelized}'
    elif launch_file_name == 'realsense.launch':
        assert args.camera_config_file_path is not None, 'Camera config file path is required for realsense launch.'
        assert args.camera_launch_file_path is not None, 'Camera launch file path is required for realsense launch.'
        assert args.serial_number is not None, 'Serial number is required for realsense launch.'
        launch_command += f' camera_config_file_path:={args.camera_config_file_path} camera_launch_file_path:={args.camera_launch_file_path} serial_number:={args.serial_number}'
    else:
        raise ValueError(f'Unknown launch: {launch_file_name}.')
    launch_command += f' gpu_id:={args.gpu_id} user_config:={args.user_config} mode:={args.mode} actions:={args.actions} hide_mapper_windows:={args.hide_mapper_windows} hide_planner_windows:={args.hide_planner_windows} step_num:={args.step_num}'
    
    get_dataset_config = rospy.ServiceProxy('get_dataset_config', GetDatasetConfig)
    rospy.wait_for_service('get_dataset_config')
    response:GetDatasetConfigResponse = get_dataset_config(GetDatasetConfigRequest())
    results_dir = response.results_dir
    with open(os.path.join(results_dir, 'launch_config.json'), 'w') as f:
        commit_id, repo_dirty_flag = get_repo_status()
        json.dump({
            "command": launch_command,
            "git": {
                "branch": get_active_branch(),
                "commit": commit_id,
                "DIRTY": repo_dirty_flag}}, f, indent=4)