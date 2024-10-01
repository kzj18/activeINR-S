#!/usr/bin/env python
import os
WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
import sys
sys.path.append(WORKSPACE)
from typing import Union
import argparse
import json
import threading

import faulthandler

import cv2
import torch
import numpy as np
import quaternion
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

import rospy
from cv_bridge import CvBridge
from std_msgs.msg import Int32
from geometry_msgs.msg import Twist, PoseStamped

from scripts import PROJECT_NAME
from scripts.visualizer import is_pose_changed, PoseChangeType
from scripts.dataloader import dataset_config_to_ros
from scripts.dataloader.dataloader import get_dataset, HabitatDataset, GoatBenchDataset, RealsenseDataset
from scripts.entry_points.nodes import frame,\
    GetDatasetConfig, GetDatasetConfigResponse, GetDatasetConfigRequest,\
        ResetEnv, ResetEnvResponse, ResetEnvRequest

class DataloaderNode:
    
    def __init__(self, config_url:str, user_config_url:str, scene_id:str) -> None:
        os.chdir(WORKSPACE)
        rospy.loginfo(f'Current working directory: {os.getcwd()}')
        with open(config_url) as f:
            config = json.load(f)
            if 'env' in config:
                config['env']['config'] = os.path.abspath(
                    os.path.join(os.path.dirname(config_url), os.pardir, os.pardir, config['env']['config']))
            if 'sensor' in config:
                config['sensor']['config'] = os.path.abspath(
                    os.path.join(os.path.dirname(config_url), os.pardir, os.pardir, config['sensor']['config']))
        self.__frame_update_translation_threshold = config['mapper']['pose']['update_threshold']['translation']
        self.__frame_update_rotation_threshold = config['mapper']['pose']['update_threshold']['rotation']
        
        with open(user_config_url) as f:
            user_config = json.load(f)
        
        self.__dataset:Union[HabitatDataset, GoatBenchDataset, RealsenseDataset] = get_dataset(config, user_config, scene_id)
        dataset_config = self.__dataset.setup()
        
        self.__dataset_config = dataset_config_to_ros(dataset_config)
        
        rospy.Service('get_dataset_config', GetDatasetConfig, self.__get_dataset_config)
        self.__reset_env_condition = threading.Condition()
        self.__resetting_env = False
        rospy.Service('reset_env', ResetEnv, self.__reset_env)
        
        movement_fail_times = 0
        movement_fail_times_pub = rospy.Publisher('movement_fail_times', Int32, queue_size=1)
        frame_pub = rospy.Publisher('frames', frame, queue_size=1)
        pose_pub = rospy.Publisher('orb_slam3/camera_pose', PoseStamped, queue_size=1)
        if isinstance(self.__dataset, (HabitatDataset, GoatBenchDataset)):
            self.__cmd_vel_condition = threading.Condition()
            self.__twist_current = None
            rospy.Subscriber('cmd_vel', Twist, self.__cmd_vel_callback, queue_size=1)
            
            cv_bridge = CvBridge()
            
            frame_c2w = None
            self.__frame_c2w_last = None
            while not rospy.is_shutdown():
                if self.__resetting_env:
                    with self.__reset_env_condition:
                        self.__dataset.reset()
                        self.__reset_env_condition.notify_all()
                        self.__resetting_env = False
                apply_movement_flag = self.__twist_current is not None
                apply_movement_result = False
                if apply_movement_flag:
                    with self.__cmd_vel_condition:
                        apply_movement_result = self.__dataset.apply_movement(self.__twist_current)
                        self.__cmd_vel_condition.notify_all()
                        self.__twist_current = None
                frame_current = self.__dataset.get_frame()
                frame_rgb = np.uint8(frame_current['rgb'] * 255)
                frame_depth = frame_current['depth']
                frame_c2w = frame_current['c2w']
                frame_quaternion = quaternion.from_rotation_matrix(frame_c2w[:3, :3])
                frame_quaternion = quaternion.as_float_array(frame_quaternion)
                frame_ros = frame()
                frame_ros.rgb = cv_bridge.cv2_to_imgmsg(cv2.Mat(frame_rgb), encoding='rgb8')
                frame_ros.depth = cv_bridge.cv2_to_imgmsg(cv2.Mat(frame_depth), encoding='32FC1')
                frame_ros.pose.position.x = frame_c2w[0, 3]
                frame_ros.pose.position.y = frame_c2w[1, 3]
                frame_ros.pose.position.z = frame_c2w[2, 3]
                frame_ros.pose.orientation.w = frame_quaternion[0]
                frame_ros.pose.orientation.x = frame_quaternion[1]
                frame_ros.pose.orientation.y = frame_quaternion[2]
                frame_ros.pose.orientation.z = frame_quaternion[3]
                frame_pub.publish(frame_ros)
                pose_ros = PoseStamped()
                pose_ros.header.stamp = rospy.Time.now()
                pose_ros.header.frame_id = 'world'
                pose_ros.pose = frame_ros.pose
                pose_pub.publish(pose_ros)
                pose_change_type = self.__is_pose_changed(frame_c2w)
                if pose_change_type != PoseChangeType.NONE:
                    if pose_change_type in (PoseChangeType.TRANSLATION, PoseChangeType.BOTH):
                        movement_fail_times = 0
                    self.__frame_c2w_last = frame_c2w
                    movement_fail_times_pub.publish(movement_fail_times)
                elif apply_movement_flag:
                    if apply_movement_result:
                        movement_fail_times += 1
                    movement_fail_times_pub.publish(movement_fail_times)
                rospy.sleep(0.01)
                
            self.__dataset.close()
        elif isinstance(self.__dataset, RealsenseDataset):
            rospy.spin()
        else:
            raise NotImplementedError(f'Invalid dataset type: {type(self.__dataset)}')
    
    def __cmd_vel_callback(self, twist:Twist) -> None:
        with self.__cmd_vel_condition:
            self.__twist_current = {
                'linear': np.array([twist.linear.x, twist.linear.y, twist.linear.z]),
                'angular': np.array([twist.angular.x, twist.angular.y, twist.angular.z])
            }
            self.__cmd_vel_condition.wait()
            
    def __get_dataset_config(self, req:GetDatasetConfigRequest) -> GetDatasetConfigResponse:
        return self.__dataset_config
    
    def __reset_env(self, req:ResetEnvRequest) -> ResetEnvResponse:
        if isinstance(self.__dataset, (HabitatDataset, GoatBenchDataset)):
            with self.__reset_env_condition:
                self.__resetting_env = True
                self.__reset_env_condition.wait()
        return ResetEnvResponse(True)
    
    def __is_pose_changed(self, frame_c2w:np.ndarray) -> PoseChangeType:
        if self.__frame_c2w_last is None:
            self.__frame_c2w_last = frame_c2w
            return PoseChangeType.BOTH
        else:
            return is_pose_changed(
                self.__frame_c2w_last,
                frame_c2w,
                self.__frame_update_translation_threshold,
                self.__frame_update_rotation_threshold)

if __name__ == '__main__':
    faulthandler.enable()
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)

    parser = argparse.ArgumentParser(description=f'{PROJECT_NAME} dataset loader.')
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
    
    args, ros_args = parser.parse_known_args()
    
    ros_args = dict([arg.split(':=') for arg in ros_args])
    
    rospy.init_node(ros_args['__name'], anonymous=True, log_level=rospy.INFO)
        
    DataloaderNode(args.config, args.user_config, args.scene_id)