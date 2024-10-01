#!/usr/bin/env python
import os
WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
import sys
sys.path.append(WORKSPACE)
import argparse
import threading
import json
from typing import Dict, Tuple, Union, List
from copy import deepcopy
from enum import Enum
import time
import shutil

import faulthandler

import torch
import numpy as np
from scipy.spatial.distance import cdist
import networkx as nx
import quaternion
from matplotlib import cm, colors
import cv2
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

import rospy
from std_msgs.msg import Int32, Bool
from geometry_msgs.msg import PoseStamped, Twist, Pose, Point

from scripts import PROJECT_NAME, GlobalState
from scripts.planner import get_update_precise_map_flag
from scripts.planner.planner import get_voronoi_graph, draw_voronoi_graph, get_closest_vertex_index, anchor_targets_frustums_to_voronoi_graph, get_safe_dijkstra_path, get_escape_plan, get_obstacle_map, optimize_navigation_path_using_precise_info, translations_topdown_to_precise, visualize_directions, optimize_navigation_path_using_precise_map_and_fast_forward, check_path_using_topdown_and_precise_map, splat_inaccessible_database
from scripts.visualizer import PoseChangeType, c2w_world_to_topdown, is_pose_changed, get_horizon_bound_topdown
from scripts.dataloader import PoseDataType, convert_to_c2w_opencv
from scripts.entry_points.nodes import\
    GetTopdownConfig, GetTopdownConfigResponse, GetTopdownConfigRequest,\
        GetTopdown, GetTopdownResponse, GetTopdownRequest,\
            SetPlannerState, SetPlannerStateResponse, SetPlannerStateRequest,\
                GetDatasetConfig, GetDatasetConfigResponse, GetDatasetConfigRequest,\
                    SetMapper, SetMapperResponse, SetMapperRequest,\
                        GetPrecise, GetPreciseResponse, GetPreciseRequest,\
                            SetKFTriggerPoses, SetKFTriggerPosesRequest, SetKFTriggerPosesResponse
    
class NodesFlagsType(Enum):
    UNARRIVED = 'UNARRIVED'
    IN_HORIZON = 'IN_HORIZON'
    GEOMETRY_UNCERTAINTY = 'GEOMETRY_UNCERTAINTY'
    SEMANTIC_GUIDANCE = 'SEMANTIC_GUIDANCE'
    FAIL = 'FAIL',

NODES_FLAGS_WEIGHT_INIT = {
    NodesFlagsType.UNARRIVED: 3,
    NodesFlagsType.IN_HORIZON: 2,
    NodesFlagsType.GEOMETRY_UNCERTAINTY: 1,
    NodesFlagsType.SEMANTIC_GUIDANCE: 3,
    NodesFlagsType.FAIL: -6,
}

class PlannerNode:
    
    __ENABLE_STATES = (GlobalState.AUTO_PLANNING, GlobalState.MANUAL_PLANNING, GlobalState.POINTNAV)
        
    __NODES_FLAGS_WEIGHT = NODES_FLAGS_WEIGHT_INIT.copy()
        
    class EscapeFlag(Enum):
        NONE = 'NONE'
        ESCAPE_ROTATION = 'ESCAPE_ROTATION'
        ESCAPE_TRANSLATION = 'ESCAPE_TRANSLATION'
    
    def __init__(
        self,
        config_url:str,
        hide_windows:bool) -> None:
        self.__hide_windows = hide_windows
        test_escape_dir = os.path.join(os.getcwd(), 'test', 'test_escape')
        shutil.rmtree(test_escape_dir, ignore_errors=True)
        os.makedirs(test_escape_dir, exist_ok=True)
        test_splat_dir = os.path.join(os.getcwd(), 'test', 'test_splat')
        shutil.rmtree(test_splat_dir, ignore_errors=True)
        os.makedirs(test_splat_dir, exist_ok=True)
        test_optimize_dir = os.path.join(os.getcwd(), 'test', 'test_optimize')
        shutil.rmtree(test_optimize_dir, ignore_errors=True)
        os.makedirs(test_optimize_dir, exist_ok=True)
        
        # XXX: Do not predefine the variables here, because the variables will be changed
        self.__voronoi_graph_nodes_score_max = 0
        self.__voronoi_graph_nodes_score_min = 0
        for value in self.__NODES_FLAGS_WEIGHT.values():
            if value > 0:
                self.__voronoi_graph_nodes_score_max += value
            elif value < 0:
                self.__voronoi_graph_nodes_score_min += value
        
        voronoi_graph_nodes_colormap = cm.get_cmap('Reds')
        voronoi_graph_nodes_colormap_colors = voronoi_graph_nodes_colormap(np.linspace(0.25, 1, 256))
        self.__voronoi_graph_nodes_colormap = colors.LinearSegmentedColormap.from_list('voronoi_graph_nodes_colormap', voronoi_graph_nodes_colormap_colors)
        
        os.chdir(WORKSPACE)
        rospy.loginfo(f'Current working directory: {os.getcwd()}')
        with open(config_url) as f:
            config = json.load(f)
            
        self.__pose_update_translation_threshold = config['mapper']['pose']['update_threshold']['translation']
        self.__pose_update_rotation_threshold = config['mapper']['pose']['update_threshold']['rotation']
        self.__step_num_as_visited = config['planner']['step_num_as_visited']
        self.__step_num_as_arrived = config['planner']['step_num_as_arrived']
        self.__step_num_as_too_far = 200
        
        self.__global_state = None
        self.__global_state_condition = threading.Condition()
        self.__point_nav_arrived_pub = rospy.Publisher('point_nav_arrived', Bool, queue_size=1)
        rospy.Service('set_planner_state', SetPlannerState, self.__set_planner_state)
        with self.__global_state_condition:
            self.__global_state_condition.wait()
        
        self.__get_dataset_config_service = rospy.ServiceProxy('get_dataset_config', GetDatasetConfig)
        rospy.wait_for_service('get_dataset_config')
        
        self.__get_topdown_config_service = rospy.ServiceProxy('get_topdown_config', GetTopdownConfig)
        rospy.wait_for_service('get_topdown_config')
        
        self.__get_topdown_service = rospy.ServiceProxy('get_topdown', GetTopdown)
        rospy.wait_for_service('get_topdown')
        
        self.__update_map_cv2_condition = threading.Condition()
        
        self.__setup_for_episode(init=True)
        
        set_mapper = rospy.ServiceProxy('set_mapper', SetMapper)
        rospy.wait_for_service('set_mapper')
        
        set_kf_trigger_poses = rospy.ServiceProxy('set_kf_trigger_poses', SetKFTriggerPoses)
        rospy.wait_for_service('set_kf_trigger_poses')
        
        self.__get_precise_service = rospy.ServiceProxy('get_precise', GetPrecise)
        rospy.wait_for_service('get_precise')
        
        self.__update_precise_map_condition = threading.Condition()
        self.__update_precise_map_condition_topdown = threading.Condition()
        
        threading.Thread(
            name='update_precise_map',
            target=self.__update_precise_map,
            daemon=True).start()
        
        rospy.Subscriber('orb_slam3/camera_pose', PoseStamped, self.__camera_pose_callback)
        rospy.wait_for_message('orb_slam3/camera_pose', PoseStamped)
        
        rospy.Subscriber('subgoal_center', Point, self.__subgoal_center_callback)
        
        rospy.Subscriber('movement_fail_times', Int32, self.__movement_fail_times_callback)
        
        self.__cv2_windows_with_callback_opened = {
            'topdown_free_map': False}
        
        threading.Thread(
            name='update_map_cv2',
            target=self.__update_map_cv2,
            daemon=True).start()
        
        self.__cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        subgoal_translation_old = None
        
        while not rospy.is_shutdown() and self.__global_state != GlobalState.QUIT:
            if self.__global_state not in self.__ENABLE_STATES:
                if self.__global_state == GlobalState.REPLAY:
                    self.__setup_for_episode()
                with self.__global_state_condition:
                    self.__global_state_condition.wait()
                    self.__arrived_flag = True
                continue
            else:
                if self.__bootstrap_flag:
                    set_mapper_request:SetMapperRequest = SetMapperRequest()
                    set_mapper_request.kf_every = 1
                    try:
                        set_mapper_response:SetMapperResponse = set_mapper(set_mapper_request)
                    except rospy.ServiceException as e:
                        rospy.logerr(f'Set mapper service call failed: {e}')
                        self.__global_state = GlobalState.QUIT
                        continue
                    kf_every_old = set_mapper_response.kf_every_old
                    twist_bootstrap = Twist()
                    twist_bootstrap.angular.z = 1
                    twist_bootstrap_up_down = Twist()
                    self.__arrived_flag = False
                    for booststrap_turn_index in range(int(np.ceil(360 / self.__dataset_config.agent_turn_angle))):
                        
                        pose_c2w_world = self.__pose_last['c2w_world'].copy()
                        self.__publish_cmd_vel(twist_bootstrap)
                        self.__get_topdown()
                        with self.__update_map_cv2_condition:
                            self.__update_map_cv2_condition.notify_all()
                        while is_pose_changed(
                                pose_c2w_world,
                                self.__pose_last['c2w_world'],
                                self.__pose_update_translation_threshold,
                                self.__pose_update_rotation_threshold) == PoseChangeType.NONE and self.__global_state != GlobalState.QUIT:
                            self.__publish_cmd_vel(twist_bootstrap)
                            self.__get_topdown()
                            with self.__update_map_cv2_condition:
                                self.__update_map_cv2_condition.notify_all()
                                
                        pose_c2w_world = self.__pose_last['c2w_world'].copy()
                        twist_bootstrap_up_down.angular.y = 1 - booststrap_turn_index % 2 * 2
                        self.__publish_cmd_vel(twist_bootstrap_up_down)
                        self.__get_topdown()
                        with self.__update_map_cv2_condition:
                            self.__update_map_cv2_condition.notify_all()
                        while is_pose_changed(
                                pose_c2w_world,
                                self.__pose_last['c2w_world'],
                                self.__pose_update_translation_threshold,
                                self.__pose_update_rotation_threshold) == PoseChangeType.NONE and self.__global_state != GlobalState.QUIT:
                            self.__publish_cmd_vel(twist_bootstrap_up_down)
                            self.__get_topdown()
                            with self.__update_map_cv2_condition:
                                self.__update_map_cv2_condition.notify_all()
                    booststrap_turn_index += 1
                    if booststrap_turn_index % 2 == 1:
                        pose_c2w_world = self.__pose_last['c2w_world'].copy()
                        twist_bootstrap_up_down.angular.y = -1
                        self.__publish_cmd_vel(twist_bootstrap_up_down)
                        self.__get_topdown()
                        with self.__update_map_cv2_condition:
                            self.__update_map_cv2_condition.notify_all()
                        while is_pose_changed(
                                pose_c2w_world,
                                self.__pose_last['c2w_world'],
                                self.__pose_update_translation_threshold,
                                self.__pose_update_rotation_threshold) == PoseChangeType.NONE and self.__global_state != GlobalState.QUIT:
                            self.__publish_cmd_vel(twist_bootstrap_up_down)
                            self.__get_topdown()
                            with self.__update_map_cv2_condition:
                                self.__update_map_cv2_condition.notify_all()
                    set_mapper_request.kf_every = kf_every_old
                    try:
                        set_mapper_response:SetMapperResponse = set_mapper(set_mapper_request)
                    except rospy.ServiceException as e:
                        rospy.logerr(f'Set mapper service call failed: {e}')
                        self.__global_state = GlobalState.QUIT
                        continue
                    self.__bootstrap_flag = False
                    self.__arrived_flag = True
                elif self.__arrived_flag:
                    self.__publish_cmd_vel(Twist())
                    self.__get_topdown()
                    if not (self.__global_state in self.__ENABLE_STATES):
                        continue
                    if self.__global_state == GlobalState.AUTO_PLANNING:
                        pose_last = self.__pose_last['topdown_translation'].copy()
                        vertex_closest_index = get_closest_vertex_index(
                            self.__voronoi_graph['vertices'],
                            self.__voronoi_graph['obstacle_map'],
                            pose_last,
                            self.__agent_radius_pixel)
                        self.__navigation_path = None
                        self.__destination_orientations = None
                        target_too_far_but_prioritize = {
                            "node_index": None,
                            "navigation_path": None,
                            "navigation_path_length": None}
                        
                        nodes_unarrived_scores = self.__voronoi_graph['nodes_score'][self.__voronoi_graph['unarrived_nodes']]
                        nodes_unarrived_index = self.__voronoi_graph['nodes_index'][self.__voronoi_graph['unarrived_nodes']]
                        
                        obstacle_map_splat = splat_inaccessible_database(
                            agent_position=pose_last,
                            global_obstacle_map=self.__voronoi_graph['obstacle_map'],
                            inaccessible_database=self.__inaccessible_database,
                            splat_size_pixel=max(self.__agent_step_size_pixel, self.__agent_radius_pixel))
                                
                        for map_used, nodes_scores_used, nodes_index_used, bootstrap_used, max_step_used, agent_radius_pixel_used in zip(
                            [self.__voronoi_graph['obstacle_map'], ],
                            [self.__voronoi_graph['nodes_score'], ],
                            [self.__voronoi_graph['nodes_index'], ],
                            [True, ],
                            [self.__step_num_as_too_far, ],
                            [self.__agent_radius_pixel, ]):
                                
                            if self.__navigation_path is not None:
                                break
                            self.__destination_orientations = None
                            target_too_far_but_prioritize = {
                                "node_index": None,
                                "navigation_path": None,
                                "navigation_path_length": None}
                            for voronoi_graph_score in range(self.__voronoi_graph_nodes_score_max, self.__voronoi_graph_nodes_score_min - 1, -1):
                                nodes_condition = nodes_scores_used == voronoi_graph_score
                                nodes_index = nodes_index_used[nodes_condition]
                                nodes_path = []
                                nodes_path_length = []
                                for node_index in nodes_index:
                                    node_vertice = self.__voronoi_graph['vertices'][node_index]
                                    if np.linalg.norm(pose_last - node_vertice) < self.__pixel_as_arrived:
                                        nodes_path_length.append(np.nan)
                                        nodes_path.append(None)
                                        continue
                                    navigation_path_index, navigation_path, graph_search_success = get_safe_dijkstra_path(
                                        self.__voronoi_graph['graph'],
                                        vertex_closest_index,
                                        node_index,
                                        self.__voronoi_graph['vertices'],
                                        map_used,
                                        pose_last,
                                        agent_radius_pixel_used)
                                    # if not graph_search_success:
                                    #     self.__fail_vertices_nodes = np.vstack([self.__fail_vertices_nodes, self.__voronoi_graph['vertices'][node_index]])
                                    if navigation_path_index is None or navigation_path is None:
                                        nodes_path_length.append(np.nan)
                                    else:
                                        whole_path = np.vstack([pose_last, navigation_path])
                                        whole_path_length = np.sum(np.linalg.norm(whole_path[1:] - whole_path[:-1], axis=1))
                                        nodes_path_length.append(whole_path_length)
                                    nodes_path.append(navigation_path)
                                nodes_path_length = np.array(nodes_path_length)
                                if np.all(np.isnan(nodes_path_length)):
                                    continue
                                else:
                                    if self.__NODES_FLAGS_WEIGHT == NODES_FLAGS_WEIGHT_INIT:
                                        if (target_too_far_but_prioritize["node_index"] is not None) and\
                                            (target_too_far_but_prioritize["navigation_path"] is not None) and\
                                                (target_too_far_but_prioritize["navigation_path_length"] is not None):
                                            nodes_path_condition = nodes_path_length < max_step_used * self.__agent_step_size_pixel
                                            if np.any(nodes_path_condition):
                                                nodes_index = nodes_index[nodes_path_condition]
                                                nodes_path = [nodes_path[i] for i in np.where(nodes_path_condition)[0]]
                                                nodes_path_length = nodes_path_length[nodes_path_condition]
                                                nodes_to_target_node = []
                                                nodes_to_target_node_length = []
                                                for node_index in nodes_index:
                                                    node_navigation_path_index, node_navigation_path, node_graph_search_success = get_safe_dijkstra_path(
                                                        self.__voronoi_graph['graph'],
                                                        node_index,
                                                        target_too_far_but_prioritize["node_index"],
                                                        self.__voronoi_graph['vertices'],
                                                        map_used,
                                                        pose_last,
                                                        agent_radius_pixel_used)
                                                    if node_navigation_path_index is None or node_navigation_path is None:
                                                        nodes_to_target_node.append(None)
                                                        nodes_to_target_node_length.append(np.nan)
                                                    else:
                                                        node_navigation_path_length = np.sum(np.linalg.norm(node_navigation_path[1:] - node_navigation_path[:-1], axis=1))
                                                        nodes_to_target_node.append(node_navigation_path)
                                                        nodes_to_target_node_length.append(node_navigation_path_length)
                                                nodes_to_target_node_length = np.array(nodes_to_target_node_length)
                                                nodes_to_target_node_length_condition = nodes_to_target_node_length < target_too_far_but_prioritize["navigation_path_length"]
                                                if np.any(nodes_to_target_node_length_condition):
                                                    node_count = np.nanargmin(nodes_to_target_node_length)
                                                    node_index = nodes_index[node_count]
                                                    navigation_path = nodes_path[node_count]
                                                    navigation_path_length = nodes_path_length[node_count]
                                                else:
                                                    continue
                                            else:
                                                continue
                                        else:
                                            node_count = np.nanargmin(nodes_path_length)
                                            node_index = nodes_index[node_count]
                                            navigation_path = nodes_path[node_count]
                                            navigation_path_length = nodes_path_length[node_count]
                                            if navigation_path_length > max_step_used * self.__agent_step_size_pixel:
                                                if bootstrap_used:  
                                                    target_too_far_but_prioritize["node_index"] = node_index
                                                    target_too_far_but_prioritize["navigation_path"] = navigation_path
                                                    target_too_far_but_prioritize["navigation_path_length"] = navigation_path_length
                                                continue
                                    else:
                                        non_nan_node_counts = np.where(np.invert(np.isnan(nodes_path_length)))[0]
                                        node_count = np.random.choice(non_nan_node_counts)
                                        node_index = nodes_index[node_count]
                                        navigation_path = nodes_path[node_count]
                                    if self.__precise_map_last['map_min_coord'] is not None and\
                                        self.__precise_map_last['map_max_coord'] is not None and\
                                            self.__precise_map_last['voronoi']['graph'] is not None and\
                                                self.__precise_map_last['voronoi']['vertices'] is not None and\
                                                    self.__precise_map_last['voronoi']['obstacle_map'] is not None:
                                        navigation_path = optimize_navigation_path_using_precise_info(
                                            agent_position_topdown=pose_last,
                                            navigation_path=navigation_path,
                                            agent_radius_pixel_topdown=self.__agent_radius_pixel,
                                            topdown_to_precise_ratio=self.__precise_config['topdown_to_precise'],
                                            precise_min_coord=self.__precise_map_last['map_min_coord'],
                                            precise_max_coord=self.__precise_map_last['map_max_coord'],
                                            precise_voronoi_graph=self.__precise_map_last['voronoi']['graph'],
                                            precise_voronoi_graph_vertices=self.__precise_map_last['voronoi']['vertices'],
                                            precise_strict_voronoi_graph=self.__precise_map_last['voronoi_strict']['graph'],
                                            precise_strict_voronoi_graph_vertices=self.__precise_map_last['voronoi_strict']['vertices'],
                                            precise_map=self.__precise_map_last['map'])
                                    self.__navigation_path = navigation_path
                                    break
                            if self.__navigation_path is None:
                                if (target_too_far_but_prioritize["node_index"] is not None) and\
                                    (target_too_far_but_prioritize["navigation_path"] is not None) and\
                                        (target_too_far_but_prioritize["navigation_path_length"] is not None):
                                    self.__navigation_path = target_too_far_but_prioritize["navigation_path"]
                                elif bootstrap_used:
                                    rospy.logwarn('No node is reachable.')
                                    self.__bootstrap_flag = True
                            else:
                                targets_frustums_index:List[int] = self.__voronoi_graph['nodes_index_anchor_targets_frustums_index'][node_index]
                                if len(targets_frustums_index) > 0:
                                    targets_frustums_index = np.array(targets_frustums_index)
                                    targets_frustums:np.ndarray = self.__cluster_info['targets_frustums_translation'][targets_frustums_index]
                                    targets_frustums = targets_frustums.reshape(-1, 2)
                                    destination_orientations = targets_frustums - self.__navigation_path[-1]
                                    # XXX: Remove after debug
                                    use_pruned_chain_count = 0
                                    for pruned_chain in self.__voronoi_graph['pruned_chains']:
                                        if np.linalg.norm(pruned_chain[-1] - self.__navigation_path[-1]) < self.__agent_radius_pixel:
                                            destination_orientations = np.vstack([
                                                destination_orientations,
                                                np.array(pruned_chain[0]).reshape(-1, 2) - self.__navigation_path[-1]])
                                            use_pruned_chain_count += 1
                                    rospy.logdebug(f'Use {use_pruned_chain_count} pruned chains.')
                                    self.__destination_orientations = np.arctan2(
                                        destination_orientations[:, 1],
                                        destination_orientations[:, 0])
                    elif self.__global_state == GlobalState.POINTNAV:
                        self.__navigation_path = None
                        self.__destination_orientations = None
                        if np.linalg.norm(self.__pose_last['topdown_translation'] - self.__subgoal_translation) < self.__task_finish_pixel:
                            self.__point_nav_arrived_pub.publish(Bool(True))
                            rospy.loginfo('Arrived.')
                        elif self.__subgoal_translation is None:
                            rospy.logwarn('No subgoal center.')
                        elif self.__escape_flag != self.EscapeFlag.NONE:
                            if subgoal_translation_old is None:
                                rospy.logwarn('Try my best to arrive, but crash')
                                subgoal_translation_old = self.__subgoal_translation.copy()
                                self.__point_nav_arrived_pub.publish(Bool(True))
                            elif np.linalg.norm(subgoal_translation_old - self.__subgoal_translation) < self.__agent_step_size_pixel:
                                rospy.logwarn('Try my best to arrive, but crash')
                                subgoal_translation_old = self.__subgoal_translation.copy()
                                self.__point_nav_arrived_pub.publish(Bool(True))
                            else:
                                pass
                        else:
                            pose_last = self.__pose_last['topdown_translation'].copy()
                            vertex_closest_index = get_closest_vertex_index(
                                self.__voronoi_graph['vertices'],
                                self.__voronoi_graph['obstacle_map'],
                                pose_last,
                                self.__agent_radius_pixel)
                            vertex_destination_index = get_closest_vertex_index(
                                self.__voronoi_graph['vertices'],
                                self.__voronoi_graph['obstacle_map'],
                                self.__subgoal_translation,
                                self.__agent_radius_pixel)
                            try:
                                navigation_path_index = nx.dijkstra_path(
                                    self.__voronoi_graph['graph'],
                                    vertex_closest_index,
                                    vertex_destination_index)
                            except nx.NetworkXNoPath:
                                rospy.logwarn('No path found.')
                                # self.__fail_vertices_nodes = np.vstack([self.__fail_vertices_nodes, self.__voronoi_graph['vertices'][vertex_destination_index]])
                                navigation_path_index = None
                            if navigation_path_index is not None:
                                navigation_path = self.__voronoi_graph['vertices'][navigation_path_index]
                                navigation_path = np.vstack([navigation_path, self.__subgoal_translation])
                                if self.__precise_map_last['map_min_coord'] is not None and\
                                    self.__precise_map_last['map_max_coord'] is not None and\
                                        self.__precise_map_last['voronoi']['graph'] is not None and\
                                            self.__precise_map_last['voronoi']['vertices'] is not None and\
                                                self.__precise_map_last['voronoi']['obstacle_map'] is not None:
                                    navigation_path = optimize_navigation_path_using_precise_info(
                                        agent_position_topdown=pose_last,
                                        navigation_path=navigation_path,
                                        agent_radius_pixel_topdown=self.__agent_radius_pixel,
                                        topdown_to_precise_ratio=self.__precise_config['topdown_to_precise'],
                                        precise_min_coord=self.__precise_map_last['map_min_coord'],
                                        precise_max_coord=self.__precise_map_last['map_max_coord'],
                                        precise_voronoi_graph=self.__precise_map_last['voronoi']['graph'],
                                        precise_voronoi_graph_vertices=self.__precise_map_last['voronoi']['vertices'],
                                        precise_strict_voronoi_graph=self.__precise_map_last['voronoi_strict']['graph'],
                                        precise_strict_voronoi_graph_vertices=self.__precise_map_last['voronoi_strict']['vertices'],
                                        precise_map=self.__precise_map_last['map'])
                                self.__navigation_path = navigation_path
                    with self.__update_map_cv2_condition:
                        self.__update_map_cv2_condition.notify_all()
                        if self.__global_state == GlobalState.MANUAL_PLANNING:
                            self.__navigation_path = None
                            self.__destination_orientations = None
                            # NOTE: Wait for the manual click for the destination.
                            self.__update_map_cv2_condition.wait()
                    if self.__navigation_path is not None:
                        self.__arrived_flag = False
                else:
                    self.__get_topdown()
                    pixel_success = self.__task_finish_pixel if self.__global_state == GlobalState.POINTNAV else self.__pixel_as_arrived
                    if np.linalg.norm(self.__pose_last['topdown_translation'] - self.__navigation_path[-1]) < pixel_success:
                        if self.__destination_orientations is not None:
                            if len(self.__destination_orientations) > 0:
                                start_orientation = np.arctan2(self.__pose_last['topdown_rotation_vector'][1], self.__pose_last['topdown_rotation_vector'][0])
                                diff_orientations = (np.degrees(self.__destination_orientations - start_orientation) + 180) % 360 - 180
                                
                                agent_turn_angle_threshold = self.__dataset_config.agent_turn_angle / 2
                                
                                diff_orientations_condition = np.abs(diff_orientations) > agent_turn_angle_threshold
                                self.__destination_orientations = self.__destination_orientations[diff_orientations_condition]
                                kf_trigger_poses_response:SetKFTriggerPosesResponse = set_kf_trigger_poses(
                                    SetKFTriggerPosesRequest(
                                        x=self.__pose_last['topdown_translation'][0],
                                        y=self.__pose_last['topdown_translation'][1],
                                        theta=self.__destination_orientations.tolist()))
                                diff_orientations = diff_orientations[diff_orientations_condition]
                                if len(diff_orientations) > 0:
                                    diff_orientation = diff_orientations[np.argmin(np.abs(diff_orientations))]
                                    cmd_vel_msg = Twist()
                                    if diff_orientation > agent_turn_angle_threshold:
                                        cmd_vel_msg.angular.z = -1
                                    elif diff_orientation < -agent_turn_angle_threshold:
                                        cmd_vel_msg.angular.z = 1
                                    else:
                                        raise ValueError('Unknown condition.')
                                    self.__publish_cmd_vel(cmd_vel_msg)
                                    with self.__update_map_cv2_condition:
                                        self.__update_map_cv2_condition.notify_all()
                                    continue
                        rospy.loginfo('Arrived.')
                        self.__arrived_flag = True
                        if self.__escape_flag != self.EscapeFlag.NONE:
                            rospy.logwarn('Cancel the escape plan because arrived.')
                            self.__escape_flag = self.EscapeFlag.NONE
                        if self.__global_state == GlobalState.POINTNAV:
                            self.__point_nav_arrived_pub.publish(Bool(True))
                        continue
                    navigation_point_index_start = 0
                    for navigation_point_index, navigation_point in enumerate(self.__navigation_path):
                        if np.linalg.norm(self.__pose_last['topdown_translation'] - navigation_point) <= self.__agent_step_size_pixel:
                            navigation_point_index_start = navigation_point_index + 1
                    self.__navigation_path = self.__navigation_path[navigation_point_index_start:]
                    if self.__precise_map_last['map'] is not None and\
                        self.__precise_map_last['map_min_coord'] is not None and\
                            self.__precise_map_last['map_max_coord'] is not None:
                        self.__navigation_path = optimize_navigation_path_using_precise_map_and_fast_forward(
                            navigation_path_topdown=self.__navigation_path,
                            precise_map=self.__precise_map_last['map'],
                            agent_position_topdown=self.__pose_last['topdown_translation'],
                            agent_radius_pixel_topdown=self.__agent_radius_pixel,
                            topdown_to_precise_ratio=self.__precise_config['topdown_to_precise'],
                            precise_min_coord=self.__precise_map_last['map_min_coord'],
                            precise_max_coord=self.__precise_map_last['map_max_coord'])
                    if self.__global_state == GlobalState.POINTNAV:
                        navigation_path = self.__navigation_path[:-1]
                    else:
                        navigation_path = self.__navigation_path.copy()
                    pose_last = self.__pose_last['topdown_translation'].copy()
                    whole_path = np.vstack([pose_last, navigation_path]).reshape(-1, 2)
                    if len(whole_path) >= 2:
                        whole_path_length = np.linalg.norm(np.diff(whole_path, axis=0), axis=1)
                        whole_path_accumulated_length = np.cumsum(whole_path_length)
                        whole_path_accumulated_length_condition = whole_path_accumulated_length <= self.__pixel_as_visited
                        if not np.any(whole_path_accumulated_length_condition):
                            whole_path = whole_path[:2]
                        elif np.all(whole_path_accumulated_length_condition):
                            pass
                        else:
                            whole_path = whole_path[:np.argmin(whole_path_accumulated_length_condition)]
                        
                        check_result = check_path_using_topdown_and_precise_map(
                            whole_path_topdown=whole_path,
                            topdown_map=self.__topdown_free_map,
                            agent_radius_pixel_topdown=self.__agent_radius_pixel,
                            precise_map=self.__precise_map_last['map'],
                            precise_min_coord=self.__precise_map_last['map_min_coord'],
                            precise_max_coord=self.__precise_map_last['map_max_coord'],
                            topdown_to_precise_ratio=self.__precise_config['topdown_to_precise'])
                        
                        if not check_result:
                            rospy.logwarn('Line test failed, crash if follow the routine.')
                            self.__arrived_flag = True
                            if self.__escape_flag != self.EscapeFlag.NONE:
                                rospy.logwarn('Cancel the escape plan because line test.')
                                self.__escape_flag = self.EscapeFlag.NONE
                            continue
                    if self.__escape_flag == self.EscapeFlag.NONE:
                        # TODO: Use inaccessibility database to avoid the crash.
                        diff_vector = self.__navigation_path[0] - self.__pose_last['topdown_translation']
                        start_orientation = np.arctan2(self.__pose_last['topdown_rotation_vector'][1], self.__pose_last['topdown_rotation_vector'][0])
                        end_orientation = np.arctan2(diff_vector[1], diff_vector[0])
                        diff_orientation = (np.degrees(end_orientation - start_orientation) + 180) % 360 - 180
                        diff_translation = np.linalg.norm(diff_vector)
                        cmd_vel_msg = Twist()
                        if diff_orientation > self.__dataset_config.agent_turn_angle:
                            cmd_vel_msg.angular.z = -1
                        elif diff_orientation < -self.__dataset_config.agent_turn_angle:
                            cmd_vel_msg.angular.z = 1
                        elif diff_translation > self.__agent_step_size_pixel:
                            cmd_vel_msg.linear.x = 1
                        else:
                            raise ValueError('Unknown condition.')
                        self.__publish_cmd_vel(cmd_vel_msg)
                        with self.__update_map_cv2_condition:
                            self.__update_map_cv2_condition.notify_all()
                    elif self.__escape_flag == self.EscapeFlag.ESCAPE_ROTATION:
                        topdown_translation_np = self.__pose_last['topdown_translation'].copy()
                        if len(self.__inaccessible_database) == 0:
                            topdown_translation = tuple(topdown_translation_np.tolist())
                            self.__inaccessible_database.setdefault(topdown_translation, np.array([]).reshape(-1, 2))
                        else:
                            inaccessible_database_topdown_translation_array =\
                                np.array(list(self.__inaccessible_database.keys())).reshape(-1, 2)
                            assert np.issubdtype(inaccessible_database_topdown_translation_array.dtype, np.floating) or np.issubdtype(inaccessible_database_topdown_translation_array.dtype, np.integer), f"Invalid dtype: {inaccessible_database_topdown_translation_array.dtype}"
                            topdown_translation_to_inaccessible_database = np.linalg.norm(
                                topdown_translation_np - inaccessible_database_topdown_translation_array,
                                axis=1)
                            inaccessible_database_topdown_translation_array_condition =\
                                topdown_translation_to_inaccessible_database < self.__agent_step_size_pixel * 0.01
                            if np.any(inaccessible_database_topdown_translation_array_condition):
                                topdown_translation_np:np.ndarray = inaccessible_database_topdown_translation_array[
                                    np.argmin(topdown_translation_to_inaccessible_database)]
                                topdown_translation = tuple(topdown_translation_np.tolist())
                            else:
                                topdown_translation = tuple(topdown_translation_np.tolist())
                                self.__inaccessible_database.setdefault(topdown_translation, np.array([]).reshape(-1, 2))
                        assert topdown_translation in self.__inaccessible_database, f"Invalid topdown_translation: {topdown_translation}"
                        if self.__global_state == GlobalState.POINTNAV:
                            self.__arrived_flag = True
                            continue
                        rotation_direction, translation_test_condition = get_escape_plan(
                            self.__topdown_free_map,
                            topdown_translation_np,
                            self.__pose_last['topdown_rotation_vector'],
                            self.__dataset_config.agent_turn_angle,
                            self.__agent_step_size_pixel,
                            self.__inaccessible_database[topdown_translation])
                        twist_rotation = Twist()
                        twist_rotation.angular.z = -rotation_direction
                        twist_translation = Twist()
                        twist_translation.linear.x = 1
                        translation_rotation_vectors = []
                        for translation_success in translation_test_condition:
                            rospy.logwarn('Start escape rotation.')
                            pose_c2w_world = self.__pose_last['c2w_world'].copy()
                            self.__publish_cmd_vel(twist_rotation)
                            self.__get_topdown()
                            with self.__update_map_cv2_condition:
                                self.__update_map_cv2_condition.notify_all()
                            while (not is_pose_changed(
                                    pose_c2w_world,
                                    self.__pose_last['c2w_world'],
                                    self.__pose_update_translation_threshold,
                                    self.__pose_update_rotation_threshold) in [PoseChangeType.ROTATION, PoseChangeType.BOTH]) and self.__global_state != GlobalState.QUIT:
                                pose_c2w_world = self.__pose_last['c2w_world'].copy()
                                self.__publish_cmd_vel(twist_rotation)
                                self.__get_topdown()
                                with self.__update_map_cv2_condition:
                                    self.__update_map_cv2_condition.notify_all()
                            if not (self.__global_state in self.__ENABLE_STATES):
                                break
                            if translation_success:
                                translation_rotation_vectors.append(self.__pose_last['topdown_rotation_vector'])
                                rospy.logwarn('Start escape translation.')
                                self.__escape_flag = self.EscapeFlag.ESCAPE_TRANSLATION
                                while self.__escape_flag == self.EscapeFlag.ESCAPE_TRANSLATION and self.__global_state != GlobalState.QUIT:
                                    self.__publish_cmd_vel(twist_translation)
                                    self.__get_topdown()
                                    with self.__update_map_cv2_condition:
                                        self.__update_map_cv2_condition.notify_all()
                                if not (self.__global_state in self.__ENABLE_STATES):
                                    break
                                if self.__escape_flag == self.EscapeFlag.NONE:
                                    rospy.logwarn('Escape finished.')
                                    break
                                elif self.__escape_flag == self.EscapeFlag.ESCAPE_ROTATION:
                                    rospy.logwarn('Cancel the escape translation plan.')
                                    if len(self.__inaccessible_database[topdown_translation]) > 0:
                                        assert np.linalg.norm(self.__inaccessible_database[topdown_translation] - self.__pose_last['topdown_translation']) >= self.__agent_step_size_pixel * 0.1, f"Invalid inaccessible_database: {self.__inaccessible_database[topdown_translation]}"
                                    self.__inaccessible_database[topdown_translation] = np.vstack([
                                        self.__inaccessible_database[topdown_translation],
                                        self.__pose_last['topdown_rotation_vector']])
                                else:
                                    raise ValueError('Uprecise_voronoi_graphnknown escape flag.')
                        translation_rotation_vectors = np.array(translation_rotation_vectors)
                        translation_rotation_vectors_vis = visualize_directions(translation_rotation_vectors)
                        cv2.imwrite(os.path.join(test_escape_dir, time.strftime('%Y-%m-%d_%H-%M-%S') + '_escape.png'), translation_rotation_vectors_vis)
                        if not (self.__global_state in self.__ENABLE_STATES):
                            continue
                        if self.__escape_flag == self.EscapeFlag.NONE:
                            rospy.logwarn('Escape finished, now replan.')
                            self.__arrived_flag = True
                        else:
                            # FIXME: Escape failed, it should not happen.
                            rospy.logerr('Escape failed, it should not happen.')
                    elif self.__escape_flag == self.EscapeFlag.ESCAPE_TRANSLATION:
                        # FIXME: Escape failed, it should not happen.
                        rospy.logerr('Invalid escape flag, it should not happen.')
                        self.__escape_flag = self.EscapeFlag.NONE
        self.__save_results()
                        
    def __setup_for_episode(self, init:bool=False) -> None:
        if not init:
            self.__save_results()
        self.__NODES_FLAGS_WEIGHT = NODES_FLAGS_WEIGHT_INIT.copy()
        self.__visited_map = None
        self.__topdown_free_map_imshow = None

        self.__dataset_config:GetDatasetConfigResponse = self.__get_dataset_config_service(GetDatasetConfigRequest())
        self.__results_dir = self.__dataset_config.results_dir
        os.makedirs(self.__results_dir, exist_ok=True)
        self.__pose_data_type = PoseDataType(self.__dataset_config.pose_data_type)
        self.__height_direction = (self.__dataset_config.height_direction // 2, (self.__dataset_config.height_direction % 2) * 2 - 1)
        
        topdown_config_response:GetTopdownConfigResponse = self.__get_topdown_config_service(GetTopdownConfigRequest())
        self.__topdown_config = {
            'world_dim_index': (
                topdown_config_response.topdown_x_world_dim_index,
                topdown_config_response.topdown_y_world_dim_index),
            'world_2d_bbox': (
                (topdown_config_response.topdown_x_world_lower_bound, topdown_config_response.topdown_x_world_upper_bound),
                (topdown_config_response.topdown_y_world_lower_bound, topdown_config_response.topdown_y_world_upper_bound)),
            'meter_per_pixel': topdown_config_response.topdown_meter_per_pixel,
            'grid_map_shape': (
                topdown_config_response.topdown_x_length,
                topdown_config_response.topdown_y_length)}
        self.__precise_config:Dict[str, float] = {
            'meter_per_pixel': topdown_config_response.precise_meter_per_pixel,
            'size_topdown': topdown_config_response.precise_size,
            'core_size_topdown': topdown_config_response.precise_core_size,
            'topdown_to_precise': topdown_config_response.topdown_meter_per_pixel / topdown_config_response.precise_meter_per_pixel,
            'precise_to_topdown': topdown_config_response.precise_meter_per_pixel / topdown_config_response.topdown_meter_per_pixel}
        assert self.__precise_config['meter_per_pixel'] <= self.__topdown_config['meter_per_pixel'], f"Invalid meter per pixel: {self.__precise_config['meter_per_pixel']} > {self.__topdown_config['meter_per_pixel']}"
        self.__topdown_image_shape = (topdown_config_response.topdown_y_length, topdown_config_response.topdown_x_length)
        self.__agent_radius_pixel = self.__dataset_config.agent_radius / self.__topdown_config['meter_per_pixel']
        self.__agent_step_size_pixel = self.__dataset_config.agent_forward_step_size / self.__topdown_config['meter_per_pixel']
        self.__task_finish_pixel = 1 / self.__topdown_config['meter_per_pixel']
        self.__pixel_as_visited = self.__agent_step_size_pixel * self.__step_num_as_visited
        self.__pixel_as_arrived = self.__agent_step_size_pixel * self.__step_num_as_arrived
        self.__pixel_as_voronoi_nodes = 5
    
        self.__inaccessible_database:Dict[Tuple[float, float], np.ndarray] = dict()
        
        self.__pose_last:Dict[str, np.ndarray] = {
            'c2w_world': None,
            'topdown_rotation_vector': None,
            'topdown_translation': None}
        self.__precise_map_last:Dict[str, Union[np.ndarray, Dict[str, Union[nx.Graph, np.ndarray, cv2.Mat, List[List[np.ndarray]]]]]] = {
            'map_min_coord': None,
            'map_max_coord': None,
            'map_x': None,
            'map_y': None,
            'map': None,
            'map_cv2': None,
            'map_approx_contour': None,
            'map_children_approx_contours': None,
            'voronoi': {
                'graph': None,
                'vertices': None,
                'obstacle_map': None,
                'pruned_chains': None,
                'nodes_index': None},
            'voronoi_strict': {
                'graph': None,
                'vertices': None,
                'obstacle_map': None,
                'pruned_chains': None,
                'nodes_index': None},
            'center_topdown': None}
        self.__topdown_translation_array = np.array([]).reshape(-1, 2)
        self.__fail_vertices_nodes = np.array([]).reshape(-1, 2)
        
        self.__subgoal_translation = None
        
        self.__movement_fail_times = 0
        self.__arrived_flag = True
        self.__escape_flag = self.EscapeFlag.NONE
        
        self.__navigation_path:np.ndarray = None
        self.__destination_orientations:np.ndarray = None
        self.__voronoi_graph:Dict[str, Union[nx.Graph, np.ndarray, cv2.Mat, List[List[np.ndarray]], Dict[int, List[int]]]] = None
        self.__topdown_free_map_raw:np.ndarray = None
        self.__topdown_free_map:np.ndarray = None
        self.__topdown_visible_map:np.ndarray = None
        self.__voronoi_graph_cv2:cv2.Mat = None
        self.__topdown_free_map_cv2:cv2.Mat = None
        self.__topdown_visible_map_cv2:cv2.Mat = None
        self.__horizon_bbox:np.ndarray = None
        self.__horizon_bbox_last_translation:np.ndarray = None
        self.__cluster_info:Dict[str, np.ndarray]  = {
            'targets_frustums_translation': None,
            'clusters_area': None,
            'clusters_n_triangles': None,
            'clusters_uncertainty_max': None,
            'clusters_uncertainty_mean': None}
        
        self.__last_twist = Twist()
        
        self.__bootstrap_flag = self.__global_state in self.__ENABLE_STATES if init else True
        with self.__update_map_cv2_condition:
            self.__update_map_cv2_condition.notify_all()
                        
    def __get_topdown(self) -> None:
        try:
            get_topdown_response:GetTopdownResponse = self.__get_topdown_service(GetTopdownRequest(self.__arrived_flag))
        except rospy.ServiceException as e:
            rospy.logerr(f'Set mapper service call failed: {e}')
            self.__global_state = GlobalState.QUIT
            with self.__global_state_condition:
                self.__global_state_condition.notify_all()
            return
        pose_last = self.__pose_last['topdown_translation'].copy()
        self.__topdown_free_map_raw = np.array(get_topdown_response.free_map).reshape(self.__topdown_image_shape).astype(np.uint8) * 255
        self.__topdown_free_map, local_obstacle_map_approx_contour, obstacle_map_children_approx_contours = get_obstacle_map(
            self.__topdown_free_map_raw,
            pose_last,
            0.225 / self.__topdown_config['meter_per_pixel'])
        self.__topdown_visible_map = np.array(get_topdown_response.visible_map).reshape(self.__topdown_image_shape).astype(np.uint8) * 255
        self.__topdown_free_map_cv2 = cv2.cvtColor(self.__topdown_free_map, cv2.COLOR_GRAY2BGR)
        self.__topdown_visible_map_cv2 = cv2.cvtColor(self.__topdown_visible_map, cv2.COLOR_GRAY2BGR)
        
        self.__horizon_bbox = get_horizon_bound_topdown(
            np.array([
                get_topdown_response.horizon_bound_min.x,
                get_topdown_response.horizon_bound_min.y,
                get_topdown_response.horizon_bound_min.z]),
            np.array([
                get_topdown_response.horizon_bound_max.x,
                get_topdown_response.horizon_bound_max.y,
                get_topdown_response.horizon_bound_max.z]),
            self.__topdown_config,
            self.__height_direction)
        
        if self.__last_twist.linear.x > 0 and self.__last_twist.angular.z == 0:
            self.__horizon_bbox_last_translation = deepcopy(self.__horizon_bbox)
        
        if self.__arrived_flag:
            with self.__update_precise_map_condition:
                self.__update_precise_map_condition.notify_all()
            with self.__update_precise_map_condition_topdown:
                self.__update_precise_map_condition_topdown.wait()
                
        while True:
            update_precise_map_flag = get_update_precise_map_flag(
                self.__precise_map_last,
                self.__precise_config,
                self.__pose_last['topdown_translation'])
            if update_precise_map_flag:
                with self.__update_precise_map_condition:
                    self.__update_precise_map_condition.notify_all()
                with self.__update_precise_map_condition_topdown:
                    self.__update_precise_map_condition_topdown.wait()
            else:
                break
                
        if self.__arrived_flag:
            self.__cluster_info['clusters_area'] = np.array(get_topdown_response.clusters_area)
            self.__cluster_info['clusters_n_triangles'] = np.array(get_topdown_response.clusters_n_triangles)
            self.__cluster_info['clusters_uncertainty_max'] = np.array(get_topdown_response.clusters_uncertainty_max)
            self.__cluster_info['clusters_uncertainty_mean'] = np.array(get_topdown_response.clusters_uncertainty_mean)
            
            targets_frustums_translation = []
            for target_frustum in get_topdown_response.targets_frustums:
                target_frustum:Pose
                target_frustum_quaternion = np.array([
                    target_frustum.orientation.w,
                    target_frustum.orientation.x,
                    target_frustum.orientation.y,
                    target_frustum.orientation.z])
                target_frustum_c2w_world = np.eye(4)
                target_frustum_c2w_world[:3, :3] = quaternion.as_rotation_matrix(quaternion.from_float_array(target_frustum_quaternion))
                target_frustum_c2w_world[:3, 3] = np.array([
                    target_frustum.position.x,
                    target_frustum.position.y,
                    target_frustum.position.z])
                target_frustum_rotation_vector, target_frustum_translation = c2w_world_to_topdown(
                    target_frustum_c2w_world,
                    self.__topdown_config,
                    self.__height_direction,
                    np.float64)
                targets_frustums_translation.append(target_frustum_translation)
                self.__cluster_info['targets_frustums_translation'] = np.array(targets_frustums_translation).reshape(-1, 2)
                
            # inaccessible_points = np.array(list(self.__inaccessible_database.keys())).reshape(-1, 2)
            inaccessible_points = np.array([]).reshape(-1, 2)
            self.__voronoi_graph = get_voronoi_graph(
                obstacle_map=self.__topdown_free_map,
                obstacle_map_approx_contour=local_obstacle_map_approx_contour,
                obstacle_map_children_approx_contours=obstacle_map_children_approx_contours,
                edge_sample_num=5,
                agent_radius_pixel=self.__agent_radius_pixel,
                inaccessible_points=inaccessible_points)
            self.__voronoi_graph['nodes_score'] = np.ones_like(
                self.__voronoi_graph['nodes_index']) * self.__voronoi_graph_nodes_score_max
            if self.__cluster_info['targets_frustums_translation'] is not None:
                nodes_index_anchor_targets_frustums_index, self.__voronoi_graph['targets_frustums_index_to_nodes_index'] = anchor_targets_frustums_to_voronoi_graph(
                    self.__voronoi_graph['vertices'],
                    self.__voronoi_graph['nodes_index'],
                    self.__cluster_info['targets_frustums_translation'])
                self.__voronoi_graph['nodes_index_anchor_targets_frustums_index'] = nodes_index_anchor_targets_frustums_index
            elif 'nodes_index_anchor_targets_frustums_index' in self.__voronoi_graph:
                nodes_index_anchor_targets_frustums_index = self.__voronoi_graph['nodes_index_anchor_targets_frustums_index']
            else:
                nodes_index_anchor_targets_frustums_index, _ = anchor_targets_frustums_to_voronoi_graph(
                    self.__voronoi_graph['vertices'],
                    self.__voronoi_graph['nodes_index'],
                    [])
                
            nodes_vertices = self.__voronoi_graph['vertices'][self.__voronoi_graph['nodes_index']]
            nodes_flags:Dict[NodesFlagsType, np.ndarray] = dict()
            
            if len(self.__topdown_translation_array) > 0:
                nodes_to_translation_array = cdist(
                    nodes_vertices,
                    self.__topdown_translation_array)
                nodes_to_translation_array = np.min(nodes_to_translation_array, axis=1)
            else:
                nodes_to_translation_array = np.ones_like(self.__voronoi_graph['nodes_index']) * np.inf
            self.__voronoi_graph['unarrived_nodes'] = nodes_to_translation_array > self.__pixel_as_visited
            nodes_flags[NodesFlagsType.UNARRIVED] = np.int32(self.__voronoi_graph['unarrived_nodes'])
            
            if len(self.__fail_vertices_nodes) > 0:
                nodes_to_fail_vertices = cdist(
                    nodes_vertices,
                    self.__fail_vertices_nodes)
                nodes_to_fail_vertices = np.min(nodes_to_fail_vertices, axis=1)
            else:
                nodes_to_fail_vertices = np.ones_like(self.__voronoi_graph['nodes_index']) * np.inf
            nodes_flags[NodesFlagsType.FAIL] = np.int32(nodes_to_fail_vertices <= self.__pixel_as_voronoi_nodes)
            
            if np.all(np.logical_or(
                np.logical_not(nodes_flags[NodesFlagsType.UNARRIVED]),
                nodes_flags[NodesFlagsType.FAIL])):
                self.__fail_vertices_nodes = np.array([]).reshape(-1, 2)
                nodes_flags[NodesFlagsType.FAIL] = np.zeros_like(self.__voronoi_graph['nodes_index'])
            
            free_space_pixels_num = cv2.countNonZero(self.__topdown_free_map)
            agent_mask = cv2.circle(
                np.zeros_like(self.__topdown_free_map),
                np.int32(pose_last),
                int(np.ceil(self.__agent_radius_pixel)),
                255,
                -1)
            line_test_results = []
            for node_vertice in nodes_vertices:
                line_test_result = cv2.line(
                    self.__topdown_free_map.copy(),
                    np.int32(pose_last),
                    np.int32(node_vertice),
                    255,
                    1)
                line_test_result[agent_mask > 0] = self.__topdown_free_map[agent_mask > 0]
                line_test_results.append(cv2.countNonZero(line_test_result) == free_space_pixels_num)
            if self.__precise_map_last['map'] is not None and\
                self.__precise_map_last['map_min_coord'] is not None:
                precise_map_last = self.__precise_map_last.copy()
                pose_last_precise = translations_topdown_to_precise(
                    pose_last,
                    precise_map_last['map_min_coord'],
                    self.__precise_config['topdown_to_precise'])
                nodes_vertices_precise = translations_topdown_to_precise(
                    nodes_vertices,
                    precise_map_last['map_min_coord'],
                    self.__precise_config['topdown_to_precise'])
                free_space_pixels_num = cv2.countNonZero(precise_map_last['map'])
                agent_mask = cv2.circle(
                    np.zeros_like(precise_map_last['map']),
                    np.int32(pose_last_precise),
                    int(np.ceil(self.__agent_radius_pixel * self.__precise_config['topdown_to_precise'])),
                    255,
                    -1)
                line_test_results_precise = []
                for node_vertice_precise in nodes_vertices_precise:
                    line_test_result_precise = cv2.line(
                        precise_map_last['map'].copy(),
                        np.int32(pose_last_precise),
                        np.int32(node_vertice_precise),
                        255,
                        1)
                    line_test_result_precise[agent_mask > 0] = precise_map_last['map'][agent_mask > 0]
                    line_test_results_precise.append(cv2.countNonZero(line_test_result_precise) == free_space_pixels_num)
                line_test_results_new = np.logical_and(
                    np.array(line_test_results),
                    np.array(line_test_results_precise))
                if np.any(line_test_results_new):
                    line_test_results = line_test_results_new
                
            if self.__horizon_bbox_last_translation is not None:
                in_horizon_bbox_condition = np.int32(
                    np.logical_and(
                        np.logical_and(
                            nodes_vertices[:, 0] >= self.__horizon_bbox_last_translation[0, 0],
                            nodes_vertices[:, 0] <= self.__horizon_bbox_last_translation[1, 0]),
                        np.logical_and(
                            nodes_vertices[:, 1] >= self.__horizon_bbox_last_translation[0, 1],
                            nodes_vertices[:, 1] <= self.__horizon_bbox_last_translation[1, 1])))
                
                line_test_results_new = np.logical_and(
                    in_horizon_bbox_condition,
                    line_test_results)
                if np.any(line_test_results_new):
                    line_test_results = line_test_results_new
                
            nodes_flags[NodesFlagsType.IN_HORIZON] = np.int32(line_test_results)
            
            nodes_flags[NodesFlagsType.GEOMETRY_UNCERTAINTY] = np.array([len(nodes_index_anchor_targets_frustums_index[node_index]) for node_index in self.__voronoi_graph['nodes_index']]).astype(bool).astype(np.int32)
            
            self.__voronoi_graph['nodes_score'] = np.zeros_like(self.__voronoi_graph['nodes_index'])
            for key, value in nodes_flags.items():
                self.__voronoi_graph['nodes_score'] += self.__NODES_FLAGS_WEIGHT[key] * value
            
            self.__voronoi_graph_cv2 = draw_voronoi_graph(
                background=np.zeros_like(self.__topdown_free_map),
                voronoi_graph_vertices=self.__voronoi_graph['vertices'],
                voronoi_graph_ridge_matrix=nx.to_numpy_array(self.__voronoi_graph['graph']),
                voronoi_graph_nodes_index=self.__voronoi_graph['nodes_index'],
                pruned_chains=self.__voronoi_graph['pruned_chains'],
                voronoi_graph_nodes_score=self.__voronoi_graph['nodes_score'],
                voronoi_graph_nodes_score_max=self.__voronoi_graph_nodes_score_max,
                voronoi_graph_nodes_score_min=self.__voronoi_graph_nodes_score_min,
                nodes_index_anchor_targets_frustums_index=nodes_index_anchor_targets_frustums_index,
                targets_frustums_translation=self.__cluster_info['targets_frustums_translation'],
                voronoi_graph_ridge_color=[255, 0, 0],
                voronoi_graph_ridge_thickness=3,
                voronoi_graph_nodes_colormap=self.__voronoi_graph_nodes_colormap,
                voronoi_graph_nodes_radius=self.__pixel_as_voronoi_nodes,
                pruned_chains_color=[0, 255, 0],
                pruned_chains_thickness=2,
                targets_frustums_color=[0, 255, 255],
                targets_frustums_radius=4,
                anchor_lines_color=[255, 0, 255],
                anchor_lines_thickness=2)
        if self.__voronoi_graph_cv2 is not None:
            voronoi_graph_cv2_mask = cv2.cvtColor(self.__voronoi_graph_cv2, cv2.COLOR_BGR2GRAY)
            self.__topdown_free_map_cv2[voronoi_graph_cv2_mask > 0] = self.__voronoi_graph_cv2[voronoi_graph_cv2_mask > 0]
                    
    def __update_map_cv2(self) -> None:
        
        def mouse_callback(event:int, x:int, y:int, flags:int, param:int) -> None:
            if event == cv2.EVENT_LBUTTONDBLCLK:
                rospy.logdebug(f'Left button double clicked at: ({x}, {y})')
                if self.__global_state == GlobalState.MANUAL_PLANNING and\
                    self.__arrived_flag and\
                        self.__voronoi_graph is not None:
                    vertices_nodes = self.__voronoi_graph['vertices'][self.__voronoi_graph['nodes_index']]
                    vertices_to_click_distance = np.linalg.norm(
                        vertices_nodes - np.array([x, y]),
                        axis=1)
                    if np.min(vertices_to_click_distance) > 20:
                        return
                    vertex_destination_index = self.__voronoi_graph['nodes_index'][np.argmin(vertices_to_click_distance)]
                    vertex_closest_index = get_closest_vertex_index(
                        self.__voronoi_graph['vertices'],
                        self.__voronoi_graph['obstacle_map'],
                        self.__pose_last['topdown_translation'],
                        self.__agent_radius_pixel)
                    navigation_path_index, navigation_path, graph_search_success = get_safe_dijkstra_path(
                        self.__voronoi_graph['graph'],
                        vertex_closest_index,
                        vertex_destination_index,
                        self.__voronoi_graph['vertices'],
                        self.__voronoi_graph['obstacle_map'],
                        self.__pose_last['topdown_translation'],
                        self.__agent_radius_pixel)
                    # if not graph_search_success:
                    #     self.__fail_vertices_nodes = np.vstack([self.__fail_vertices_nodes, self.__voronoi_graph['vertices'][vertex_destination_index]])
                    if navigation_path_index is None or navigation_path is None:
                        rospy.logwarn('No path found.')
                        return
                    else:
                        if self.__precise_map_last['map_min_coord'] is not None and\
                            self.__precise_map_last['map_max_coord'] is not None and\
                                self.__precise_map_last['voronoi']['graph'] is not None and\
                                    self.__precise_map_last['voronoi']['vertices'] is not None and\
                                        self.__precise_map_last['voronoi']['obstacle_map'] is not None:
                            navigation_path = optimize_navigation_path_using_precise_info(
                                agent_position_topdown=self.__pose_last['topdown_translation'],
                                navigation_path=navigation_path,
                                agent_radius_pixel_topdown=self.__agent_radius_pixel,
                                topdown_to_precise_ratio=self.__precise_config['topdown_to_precise'],
                                precise_min_coord=self.__precise_map_last['map_min_coord'],
                                precise_max_coord=self.__precise_map_last['map_max_coord'],
                                precise_voronoi_graph=self.__precise_map_last['voronoi']['graph'],
                                precise_voronoi_graph_vertices=self.__precise_map_last['voronoi']['vertices'],
                                precise_strict_voronoi_graph=self.__precise_map_last['voronoi_strict']['graph'],
                                precise_strict_voronoi_graph_vertices=self.__precise_map_last['voronoi_strict']['vertices'],
                                precise_map=self.__precise_map_last['map'])
                        self.__navigation_path = navigation_path
                        return
        
        while not rospy.is_shutdown() and self.__global_state != GlobalState.QUIT:
            if self.__global_state not in self.__ENABLE_STATES:
                cv2.destroyAllWindows()
                for window_name in self.__cv2_windows_with_callback_opened.keys():
                    self.__cv2_windows_with_callback_opened[window_name] = False
                with self.__global_state_condition:
                    self.__global_state_condition.wait()
                continue
            wait_for_manual_planning_flag = self.__arrived_flag and self.__global_state == GlobalState.MANUAL_PLANNING
            if not wait_for_manual_planning_flag:
                with self.__update_map_cv2_condition:
                    self.__update_map_cv2_condition.wait()
            cv2_imshow_flag = False
            if self.__topdown_free_map_cv2 is not None:
                if (not self.__cv2_windows_with_callback_opened['topdown_free_map']) and (not self.__hide_windows):
                    cv2.namedWindow('Topdown Free Map', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('Topdown Free Map', width=self.__topdown_config['grid_map_shape'][0], height=self.__topdown_config['grid_map_shape'][1])
                    cv2.moveWindow('Topdown Free Map', 0, 800)
                    cv2.namedWindow('Topdown Free Map Visited', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('Topdown Free Map Visited', width=self.__topdown_config['grid_map_shape'][0], height=self.__topdown_config['grid_map_shape'][1])
                    cv2.moveWindow('Topdown Free Map Visited', 100, 800)
                    cv2.namedWindow('Topdown Visible Map', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('Topdown Visible Map', width=self.__topdown_config['grid_map_shape'][0], height=self.__topdown_config['grid_map_shape'][1])
                    cv2.moveWindow('Topdown Visible Map', 200, 800)
                    self.__cv2_windows_with_callback_opened['topdown_free_map'] = True
                    cv2.setMouseCallback('Topdown Free Map', mouse_callback)
                    cv2.namedWindow('Precise Map', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('Precise Map', width=self.__precise_config['size_topdown'], height=self.__precise_config['size_topdown'])
                    cv2.moveWindow('Precise Map', 300, 800)
                topdown_free_map = self.__topdown_free_map_cv2.copy()
                navigation_path = deepcopy(self.__navigation_path)
                if navigation_path is not None and len(navigation_path) > 0:
                    navigation_path_cv2 = np.zeros_like(topdown_free_map)
                    cv2.polylines(
                        navigation_path_cv2,
                        [np.int32(np.vstack([self.__pose_last['topdown_translation'], navigation_path]))],
                        False,
                        (0, 64, 255),
                        6)
                    cv2.circle(
                        navigation_path_cv2,
                        np.int32(navigation_path[-1]),
                        8,
                        (0, 64, 255) if self.__global_state == GlobalState.POINTNAV else (0, 128, 255),
                        -1)
                    navigation_path_cv2_mask = cv2.cvtColor(navigation_path_cv2, cv2.COLOR_BGR2GRAY)
                    voronoi_graph_cv2_mask = cv2.cvtColor(self.__voronoi_graph_cv2, cv2.COLOR_BGR2GRAY)
                    navigation_path_cv2_condition = np.logical_and(
                        navigation_path_cv2_mask > 0,
                        voronoi_graph_cv2_mask == 0)
                    topdown_free_map[navigation_path_cv2_condition] = navigation_path_cv2[navigation_path_cv2_condition]
                if self.__pose_last['topdown_translation'] is not None:
                    if self.__pose_last['topdown_rotation_vector'] is not None:
                        cv2.arrowedLine(
                            topdown_free_map,
                            self.__pose_last['topdown_translation'].astype(np.int32).tolist(),
                            (self.__pose_last['topdown_translation'] + self.__pose_last['topdown_rotation_vector'] * 10).astype(np.int32).tolist(),
                            (0, 255, 0),
                            2)
                    cv2.circle(
                        topdown_free_map,
                        self.__pose_last['topdown_translation'].astype(np.int32).tolist(),
                        int(np.ceil(self.__agent_radius_pixel)),
                        (128, 255, 128),
                        -1)
                self.__topdown_free_map_imshow = topdown_free_map.copy()
                if not self.__hide_windows:
                    cv2.imshow('Topdown Free Map', self.__topdown_free_map_imshow)
                    cv2_imshow_flag = True
                if len(self.__topdown_translation_array) > 1:
                    visited_map = np.ones_like(self.__topdown_free_map_cv2) * 255
                    cv2.polylines(
                        visited_map,
                        [np.int32(self.__topdown_translation_array)],
                        False,
                        (0, 255, 0),
                        int(np.ceil(2 * self.__pixel_as_visited)))
                    visited_map[self.__topdown_free_map == 0] = [0, 0, 0]
                    if self.__horizon_bbox is not None:
                        cv2.rectangle(
                            visited_map,
                            np.int32(self.__horizon_bbox[0]),
                            np.int32(self.__horizon_bbox[1]),
                            (255, 0, 0),
                            1)
                    if self.__horizon_bbox_last_translation is not None:
                        cv2.rectangle(
                            visited_map,
                            np.int32(self.__horizon_bbox_last_translation[0]),
                            np.int32(self.__horizon_bbox_last_translation[1]),
                            (255, 0, 255),
                            1)
                    for fail_vertex in self.__fail_vertices_nodes:
                        cv2.circle(
                            visited_map,
                            np.int32(fail_vertex),
                            int(np.ceil(self.__pixel_as_voronoi_nodes)),
                            (0, 0, 255),
                            -1)
                    self.__visited_map = visited_map
                    if not self.__hide_windows:
                        cv2.imshow('Topdown Free Map Visited', self.__visited_map)
            if self.__topdown_visible_map_cv2 is not None:
                topdown_visible_map = self.__topdown_visible_map_cv2.copy()
                if self.__pose_last['topdown_translation'] is not None:
                    if self.__pose_last['topdown_rotation_vector'] is not None:
                        cv2.arrowedLine(
                            topdown_visible_map,
                            self.__pose_last['topdown_translation'].astype(np.int32).tolist(),
                            (self.__pose_last['topdown_translation'] + self.__pose_last['topdown_rotation_vector'] * 10).astype(np.int32).tolist(),
                            (0, 255, 0),
                            2)
                    cv2.circle(
                        topdown_visible_map,
                        self.__pose_last['topdown_translation'].astype(np.int32).tolist(),
                        int(np.ceil(self.__agent_radius_pixel)),
                        (128, 255, 128),
                        -1)
                if not self.__hide_windows:
                    cv2.imshow('Topdown Visible Map', topdown_visible_map)
                    cv2_imshow_flag = True
            if self.__precise_map_last['map_cv2'] is not None:
                precise_map = self.__precise_map_last['map_cv2'].copy()
                navigation_path = deepcopy(self.__navigation_path)
                if self.__precise_map_last['map_min_coord'] is not None:
                    precise_translation = translations_topdown_to_precise(
                        self.__pose_last['topdown_translation'],
                        self.__precise_map_last['map_min_coord'],
                        self.__precise_config['topdown_to_precise'])
                    if self.__precise_map_last['map_approx_contour'] is not None:
                        cv2.drawContours(
                            precise_map,
                            [self.__precise_map_last['map_approx_contour']],
                            -1,
                            (255, 0, 0),
                            1)
                    if self.__precise_map_last['map_children_approx_contours'] is not None:
                        cv2.drawContours(
                            precise_map,
                            self.__precise_map_last['map_children_approx_contours'],
                            -1,
                            (0, 255, 0),
                            1)
                    if navigation_path is not None and len(navigation_path) > 0:
                        navigation_path_precise = translations_topdown_to_precise(
                            navigation_path,
                            self.__precise_map_last['map_min_coord'],
                            self.__precise_config['topdown_to_precise'])
                        cv2.polylines(
                            precise_map,
                            [np.int32(np.vstack([precise_translation, navigation_path_precise]))],
                            False,
                            (0, 64, 255),
                            6)
                        cv2.circle(
                            precise_map,
                            np.int32(navigation_path_precise[-1]),
                            8,
                            (0, 64, 255) if self.__global_state == GlobalState.POINTNAV else (0, 128, 255),
                            -1)
                    cv2.circle(
                        precise_map,
                        np.int32(precise_translation),
                        int(np.ceil(self.__agent_radius_pixel * self.__precise_config['topdown_to_precise'])),
                        (128, 255, 128),
                        -1)
                    precise_map[self.__precise_map_last['map'] == 0] = self.__precise_map_last['map_cv2'][self.__precise_map_last['map'] == 0]
                if not self.__hide_windows:
                    cv2.imshow('Precise Map', precise_map)
                    cv2_imshow_flag = True
                    
            if cv2_imshow_flag:
                key = cv2.waitKey(1)
                if wait_for_manual_planning_flag:
                    if key == 13:
                        with self.__update_map_cv2_condition:
                            self.__update_map_cv2_condition.notify_all()
                            
    def __update_precise_map(self) -> None:
        while not rospy.is_shutdown() and self.__global_state != GlobalState.QUIT:
            with self.__update_precise_map_condition_topdown:
                self.__update_precise_map_condition_topdown.notify_all()
            with self.__update_precise_map_condition:
                self.__update_precise_map_condition.notify_all()
                self.__update_precise_map_condition.wait()
            pose_last = deepcopy(self.__pose_last)
            try:
                get_precise_map_response:GetPreciseResponse = self.__get_precise_service(GetPreciseRequest(*pose_last['topdown_translation']))
            except rospy.ServiceException as e:
                rospy.logerr(f'Get precise service call failed: {e}')
                self.__global_state = GlobalState.QUIT
                with self.__global_state_condition:
                    self.__global_state_condition.notify_all()
                return
            precise_map_raw = np.uint8(np.array(get_precise_map_response.precise_map).reshape((self.__precise_config['size_topdown'], self.__precise_config['size_topdown']))) * 255
            precise_map_x = np.array(get_precise_map_response.precise_map_x).reshape((self.__precise_config['size_topdown'], self.__precise_config['size_topdown']))
            precise_map_y = np.array(get_precise_map_response.precise_map_y).reshape((self.__precise_config['size_topdown'], self.__precise_config['size_topdown']))
            
            precise_map_max_coord = np.array([np.max(precise_map_x), np.max(precise_map_y)])
            precise_map_min_coord = np.array([np.min(precise_map_x), np.min(precise_map_y)])
            
            inaccessible_database_precise = dict()
            for translation, rotation_vectors in self.__inaccessible_database.items():
                translation_precise = translations_topdown_to_precise(
                    np.array(translation),
                    precise_map_min_coord,
                    self.__precise_config['topdown_to_precise'])
                inaccessible_database_precise[tuple(translation_precise.tolist())] = rotation_vectors
            
            precise_translation = translations_topdown_to_precise(
                pose_last['topdown_translation'],
                precise_map_min_coord,
                self.__precise_config['topdown_to_precise'])
            precise_map = splat_inaccessible_database(
                agent_position=precise_translation,
                global_obstacle_map=precise_map_raw,
                inaccessible_database=inaccessible_database_precise,
                splat_size_pixel=max(self.__agent_step_size_pixel, self.__agent_radius_pixel) * self.__precise_config['topdown_to_precise'])
            precise_obstacle_map, local_obstacle_map_approx_contour, obstacle_map_children_approx_contours = get_obstacle_map(
                global_obstacle_map=precise_map,
                agent_position=precise_translation,
                approx_precision=3)
            
            precise_voronoi_graph = {
                'graph': None,
                'vertices': None,
                'obstacle_map': None,
                'pruned_chains': None,
                'nodes_index': None}
            precise_strict_voronoi_graph = {
                'graph': None,
                'vertices': None,
                'obstacle_map': None,
                'pruned_chains': None,
                'nodes_index': None}
            if self.__voronoi_graph is not None:
                inaccessible_points = np.array([]).reshape(-1, 2)
                precise_voronoi_graph = get_voronoi_graph(
                    precise_obstacle_map,
                    local_obstacle_map_approx_contour,
                    obstacle_map_children_approx_contours,
                    5,
                    self.__agent_radius_pixel,
                    inaccessible_points)
                inaccessible_points = np.array(list(self.__inaccessible_database.keys())).reshape(-1, 2)
                if len(inaccessible_points) > 0:
                    inaccessible_points = translations_topdown_to_precise(
                        inaccessible_points,
                        precise_map_min_coord,
                        self.__precise_config['topdown_to_precise'])
                    precise_strict_voronoi_graph = get_voronoi_graph(
                        precise_obstacle_map,
                        local_obstacle_map_approx_contour,
                        obstacle_map_children_approx_contours,
                        5,
                        self.__agent_radius_pixel,
                        inaccessible_points)
                else:
                    precise_strict_voronoi_graph = precise_voronoi_graph.copy()
                
                if self.__navigation_path is not None:
                    if len(self.__navigation_path) > 0:
                        self.__navigation_path = optimize_navigation_path_using_precise_info(
                            agent_position_topdown=pose_last['topdown_translation'],
                            navigation_path=self.__navigation_path,
                            agent_radius_pixel_topdown=self.__agent_radius_pixel,
                            topdown_to_precise_ratio=self.__precise_config['topdown_to_precise'],
                            precise_min_coord=precise_map_min_coord,
                            precise_max_coord=precise_map_max_coord,
                            precise_voronoi_graph=precise_voronoi_graph['graph'],
                            precise_voronoi_graph_vertices=precise_voronoi_graph['vertices'],
                            precise_strict_voronoi_graph=precise_strict_voronoi_graph['graph'],
                            precise_strict_voronoi_graph_vertices=precise_strict_voronoi_graph['vertices'],
                            precise_map=precise_map)
            
            self.__precise_map_last = {
                'map_min_coord': precise_map_min_coord,
                'map_max_coord': precise_map_max_coord,
                'map_x': precise_map_x,
                'map_y': precise_map_y,
                'map': precise_map,
                'map_cv2': cv2.cvtColor(precise_map, cv2.COLOR_GRAY2BGR),
                'map_approx_contour': local_obstacle_map_approx_contour,
                'map_children_approx_contours': obstacle_map_children_approx_contours,
                'voronoi': precise_voronoi_graph,
                'voronoi_strict': precise_strict_voronoi_graph,
                'center_topdown': pose_last['topdown_translation']}
                
    def __set_planner_state(self, request:SetPlannerStateRequest) -> SetPlannerStateResponse:
        rospy.loginfo(f'Set planner state: {request.global_state}')
        if self.__global_state is None:
            self.__global_state = GlobalState(request.global_state)
            with self.__global_state_condition:
                self.__global_state_condition.notify_all()
        else:
            global_state_old = GlobalState(self.__global_state)
            self.__global_state = GlobalState(request.global_state)
            if (self.__global_state in self.__ENABLE_STATES) and (global_state_old not in self.__ENABLE_STATES):
                with self.__global_state_condition:
                    self.__global_state_condition.notify_all()
            if self.__global_state == GlobalState.QUIT:
                with self.__global_state_condition:
                    self.__global_state_condition.notify_all()
            if self.__global_state == GlobalState.POINTNAV:
                self.__point_nav_arrived_pub.publish(Bool(False))
        return SetPlannerStateResponse()
    
    def __subgoal_center_callback(self, subgoal_center:Point) -> None:
        subgoal_translation = np.array([subgoal_center.x, subgoal_center.y, subgoal_center.z])
        subgoal_c2w_world = np.eye(4)
        subgoal_c2w_world[:3, 3] = subgoal_translation
        _, self.__subgoal_translation = c2w_world_to_topdown(
            subgoal_c2w_world,
            self.__topdown_config,
            self.__height_direction,
            np.float64)
        rospy.loginfo(f"Received subgoal centers: {self.__subgoal_translation}")
        
    def __camera_pose_callback(self, pose:PoseStamped) -> None:
        pose_translation = np.array([pose.pose.position.x, pose.pose.position.y, pose.pose.position.z])
        pose_quaternion = np.array([
            pose.pose.orientation.w,
            pose.pose.orientation.x,
            pose.pose.orientation.y,
            pose.pose.orientation.z])
        pose_quaternion = quaternion.from_float_array(pose_quaternion)
        pose_rotation = quaternion.as_rotation_matrix(pose_quaternion)
        pose_c2w_world = np.eye(4)
        pose_c2w_world[:3, :3] = pose_rotation
        pose_c2w_world[:3, 3] = pose_translation
        
        pose_c2w_world = convert_to_c2w_opencv(pose_c2w_world, self.__pose_data_type)
        
        if self.__pose_last['c2w_world'] is None:
            pass
        elif is_pose_changed(
                self.__pose_last['c2w_world'],
                pose_c2w_world,
                self.__pose_update_translation_threshold,
                self.__pose_update_rotation_threshold) == PoseChangeType.NONE:
            return
        
        pose_rotation_vector = np.degrees(quaternion.as_rotation_vector(pose_quaternion))
        rospy.loginfo(f'Agent:\n\tX: {pose_translation[0]:.2f}, Y: {pose_translation[1]:.2f}, Z: {pose_translation[2]:.2f}\n\tX_angle: {pose_rotation_vector[0]:.2f}, Y_angle: {pose_rotation_vector[1]:.2f}, Z_angle: {pose_rotation_vector[2]:.2f}')
        
        pose_topdown_rotation_vector, pose_topdown_translation = c2w_world_to_topdown(
            pose_c2w_world,
            self.__topdown_config,
            self.__height_direction,
            np.float64)
        
        pose_current = {
            'c2w_world': pose_c2w_world,
            'topdown_rotation_vector': pose_topdown_rotation_vector,
            'topdown_translation': pose_topdown_translation}
        
        self.__pose_last = pose_current.copy()
        # TODO: Use unique to remove duplicate
        self.__topdown_translation_array = np.vstack([self.__topdown_translation_array, pose_topdown_translation])
        
        update_precise_map_flag = get_update_precise_map_flag(
            self.__precise_map_last,
            self.__precise_config,
            pose_topdown_translation)
        
        if update_precise_map_flag:
            with self.__update_precise_map_condition:
                self.__update_precise_map_condition.notify_all()
        
        if self.__global_state in self.__ENABLE_STATES:
            with self.__global_state_condition:
                self.__global_state_condition.notify_all()
        return
    
    def __movement_fail_times_callback(self, movement_fail_times:Int32) -> None:
        rospy.logwarn(f'Get movement fail times: {movement_fail_times.data}')
        if movement_fail_times.data > self.__movement_fail_times and not self.__arrived_flag:
            rospy.logwarn(f'Movement fail times: {self.__movement_fail_times}')
            self.__movement_fail_times = movement_fail_times.data
            if self.__escape_flag == self.EscapeFlag.NONE:
                self.__escape_flag = self.EscapeFlag.ESCAPE_ROTATION
                rospy.logwarn('Start escaping.')
                if self.__navigation_path is not None:
                    if len(self.__navigation_path) > 0:
                        self.__fail_vertices_nodes = np.vstack([self.__fail_vertices_nodes, self.__navigation_path[-1]])
            elif self.__escape_flag == self.EscapeFlag.ESCAPE_TRANSLATION:
                self.__escape_flag = self.EscapeFlag.ESCAPE_ROTATION
                rospy.logwarn('Escape failed.')
        elif movement_fail_times.data == 0 and self.__movement_fail_times > 0:
            self.__movement_fail_times = 0
            rospy.loginfo('Movement fail times reset.')
            if self.__escape_flag == self.EscapeFlag.ESCAPE_TRANSLATION:
                self.__escape_flag = self.EscapeFlag.NONE
                rospy.loginfo('Escape success.')
        return
    
    def __publish_cmd_vel(self, twist:Twist) -> None:
        self.__last_twist = twist
        self.__cmd_vel_pub.publish(twist)
        return
    
    def __save_results(self) -> None:
        if self.__visited_map is not None:
            cv2.imwrite(os.path.join(self.__results_dir, 'visited_map.png'), self.__visited_map)
        if self.__topdown_free_map_imshow is not None:
            cv2.imwrite(os.path.join(self.__results_dir, 'topdown_free_map.png'), self.__topdown_free_map_imshow)

if __name__ == '__main__':
    faulthandler.enable()
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)

    parser = argparse.ArgumentParser(description=f'{PROJECT_NAME} planner node.')
    parser.add_argument('--config',
                        type=str,
                        required=True,
                        help='Input config url (*.json).')
    parser.add_argument('--hide_windows',
                        type=int,
                        required=True,
                        help='Disable windows.')
    
    args, ros_args = parser.parse_known_args()
    
    ros_args = dict([arg.split(':=') for arg in ros_args])
    
    rospy.init_node(ros_args['__name'], anonymous=True, log_level=rospy.DEBUG)
    
    PlannerNode(args.config, bool(args.hide_windows))
    
    rospy.loginfo(f'{PROJECT_NAME} planner node finished.')