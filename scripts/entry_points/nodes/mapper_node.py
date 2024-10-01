#!/usr/bin/env python
import os
WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
import sys
sys.path.append(WORKSPACE)
import json
import argparse
from typing import Union
import subprocess

import faulthandler

import torch
import numpy as np
from open3d.visualization import gui
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

import rospy

from scripts import PROJECT_NAME, GlobalState
from scripts.dataloader.dataloader import get_dataset, HabitatDataset, GoatBenchDataset
from scripts.visualizer.visualizer import Visualizer

if __name__ == '__main__':
    faulthandler.enable()
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)

    parser = argparse.ArgumentParser(description=f'{PROJECT_NAME} mapper node.')
    parser.add_argument('--config',
                        type=str,
                        required=True,
                        help='Input config url (*.json).')
    parser.add_argument('--scene_id',
                        type=str,
                        required=True,
                        help='Specify test scene id.')
    parser.add_argument('--user_config',
                        type=str,
                        required=True,
                        help='User config url (*.json).')
    parser.add_argument('--gpu_id',
                        type=int,
                        required=True,
                        help='Specify gpu id.')
    parser.add_argument('--mode',
                        type=str,
                        choices=list(GlobalState.__members__)[:-1],
                        required=True,
                        help='Specify the mode to start with.')
    parser.add_argument('--actions',
                        type=str,
                        required=True,
                        help='Specify the actions to replay.')
    parser.add_argument('--ros_dataloader',
                        type=int,
                        required=True,
                        help='Tell the mapper node to use ROS dataloader.')
    parser.add_argument('--parallelized',
                        type=int,
                        required=True,
                        help='Tell the mapper node to be parallelized.')
    parser.add_argument('--hide_windows',
                        type=int,
                        required=True,
                        help='Disable windows.')
    
    args, ros_args = parser.parse_known_args()
    
    ros_args = dict([arg.split(':=') for arg in ros_args])
    
    rospy.init_node(ros_args['__name'], anonymous=True, log_level=rospy.DEBUG)
    
    if args.mode == 'REPLAY' and args.actions is None:
        parser.error('Replay mode requires actions to replay.')
    
    if torch.cuda.is_available():
        device = torch.device('cuda', args.gpu_id)
    else:
        rospy.logwarn('No GPU available.')
        device = torch.device('cpu')
        
    if not args.ros_dataloader:
        os.chdir(WORKSPACE)
        rospy.loginfo(f'Current working directory: {os.getcwd()}')
        with open(args.config) as f:
            config = json.load(f)
            if 'env' in config:
                config['env']['config'] = os.path.abspath(
                    os.path.join(WORKSPACE, config['env']['config']))
            if 'sensor' in config:
                config['sensor']['config'] = os.path.abspath(
                    os.path.join(WORKSPACE, config['sensor']['config']))
        
        with open(args.user_config) as f:
            user_config = json.load(f)
        
        dataset:Union[HabitatDataset, GoatBenchDataset] = get_dataset(config, user_config, args.scene_id)
    else:
        dataset = None

    hide_windows = bool(args.hide_windows)
    if not hide_windows:
        app = gui.Application.instance
        app.initialize()
    w = Visualizer(
        args.config,
        GlobalState(args.mode),
        1 if hide_windows else app.add_font(gui.FontDescription(gui.FontDescription.MONOSPACE)),
        device,
        args.actions,
        dataset,
        bool(args.parallelized),
        hide_windows)
    if hide_windows:
        rospy.spin()
    else:
        app.run()
    
    rospy.loginfo(f'{PROJECT_NAME} mapper node finished.')