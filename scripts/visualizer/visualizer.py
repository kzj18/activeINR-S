import os
import time
WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
import json
from queue import Queue
from typing import Dict, List, Tuple, Union
from enum import Enum
import threading

import torch
import numpy as np
import quaternion
import cv2
from imgviz import depth2rgb
from matplotlib import cm
import open3d as o3d
from open3d.visualization import rendering, gui

import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import Twist, PoseStamped, Pose

from scripts import start_timing, end_timing, OPENCV_TO_OPENGL, PROJECT_NAME, GlobalState
from scripts.dataloader import RGBDSensor, PoseDataType, load_scene_mesh, dataset_config_to_ros, convert_to_c2w_opencv
from scripts.visualizer import PoseChangeType, rgbd_to_pointcloud, pose_to_matrix, rotation_matrix_from_vectors, c2w_world_to_topdown, c2w_topdown_to_world, is_pose_changed, get_horizon_bound_topdown, config_topdown_info, adjust_topdown_maps, translations_world_to_topdown, visualize_topdown_and_precise_map, visualize_agent, translations_world_to_topdown
from scripts.mapper.mapper import Mapper, MapperState, MeshColorType
from scripts.dataloader.dataloader import HabitatDataset, GoatBenchDataset
from scripts.entry_points.nodes import frame,\
    GetDatasetConfig, GetDatasetConfigResponse, GetDatasetConfigRequest,\
        ResetEnv, ResetEnvResponse, ResetEnvRequest,\
            GetTopdown, GetTopdownRequest, GetTopdownResponse,\
                GetTopdownConfig, GetTopdownConfigRequest, GetTopdownConfigResponse,\
                    SetPlannerState, SetPlannerStateRequest, SetPlannerStateResponse,\
                        SetMapper, SetMapperRequest, SetMapperResponse,\
                            GetPrecise, GetPreciseResponse, GetPreciseRequest,\
                                SetKFTriggerPoses, SetKFTriggerPosesRequest, SetKFTriggerPosesResponse

# TODO: Dataloader give height dim index, and height direction, because realsense height direction is z+
CURRENT_FRUSTUM = {
    'color': [0.961, 0.475, 0.000],
    'scale': 0.2,
    'material': 'unlit_line_mat',
}
CURRENT_AGENT = {
    'color': [0.961, 0.475, 0.000],
    'material': 'lit_mat_transparency',
}
CURRENT_HORIZON = {
    'color': [0.0, 1.0, 0.0],
}
UNCERTAINTY_MESH = {
    'colormap': 'jet',
}
TARGETS_FRUSTUMS = {
    'colormap': 'rainbow',
    'material': 'unlit_line_mat',
    'scale': 0.2
}
    
def set_enable(widget:gui.Widget, enable:bool):
    widget.enabled = enable
    for child in widget.get_children():
        set_enable(child, enable)
    return

class Visualizer:
    
    class QueryMeshFlag(Enum):
        NONE = 0
        VISUALIZATION = 1
        PLANNING = 2
        
    class LocalDatasetState(Enum):
        INITIALIZING = 0
        INITIALIZED = 1
        RUNNING = 2
        
    class QueryTopdownFlag(Enum):
        NONE = 0
        ARRIVED = 1
        RUNNING = 2
        MANUAL = 3
        
    class ClustersSortType(Enum):
        AREA = 'AREA'
        TRIANGLES_N = 'TRIANGLES_N'
        UNCERTAINTY_MAX = 'UNCERTAINTY_MAX'
        UNCERTAINTY_MEAN = 'UNCERTAINTY_MEAN'
        
    def __init__(
        self,
        config_url:str,
        init_state:GlobalState,
        font_id:int,
        device:torch.device,
        actions_url:str,
        local_dataset:Union[HabitatDataset, GoatBenchDataset],
        parallelized:bool,
        hide_windows:bool):
        self.__device = device
        self.__hide_windows = hide_windows
        self.__global_states_selectable = [GlobalState.AUTO_PLANNING, GlobalState.MANUAL_PLANNING, GlobalState.MANUAL_CONTROL, GlobalState.PAUSE]
        self.__local_dataset = local_dataset
        self.__local_dataset_parallelized = parallelized
        
        if actions_url == 'None':
            self.__actions = None
            self.__global_state = init_state
        else:
            self.__actions = np.loadtxt(actions_url, dtype=np.int32)
            self.__global_state = GlobalState.REPLAY
            self.__global_states_selectable.append(GlobalState.REPLAY)
        if self.__local_dataset is not None:
            self.__local_dataset_state = self.LocalDatasetState.INITIALIZING
            self.__local_dataset_condition = threading.Condition()
            self.__local_dataset_pose_pub = rospy.Publisher('orb_slam3/camera_pose', PoseStamped, queue_size=1)
            self.__local_dataset_pose_ros = None
            self.__local_dataset_thread = threading.Thread(
                target=self.__update_dataset,
                name='UpdateDataset',
                daemon=True)
            self.__local_dataset_thread.start()
            self.__local_dataset_label = gui.Label('')
            self.__local_dataset_label.font_id = font_id
        
        os.chdir(WORKSPACE)
        rospy.loginfo(f'Current working directory: {os.getcwd()}')
        with open(config_url) as f:
            config = json.load(f)
        
        self.__bbox_padding = config['mapper']['bbox_padding_ratio']
        self.__voxel_size_eval = config['mapper']['meshing']['voxel_size_eval']
        self.__voxel_size_final = config['mapper']['meshing']['voxel_size_final']
        assert self.__voxel_size_eval > self.__voxel_size_final, f'Invalid voxel size: {self.__voxel_size_eval} <= {self.__voxel_size_final}'
        self.__frame_update_translation_threshold = config['mapper']['pose']['update_threshold']['translation']
        self.__frame_update_rotation_threshold = config['mapper']['pose']['update_threshold']['rotation']
        self.__targets_frustums_shift_length = config['mapper']['uncertainty']['frustum_shift_length']
        
        self.__step_num_as_arrived = config['planner']['step_num_as_arrived']
        
        self.__update_main_thread = threading.Thread(
            target=self.__update_main,
            name='UpdateMain',
            daemon=True)
        
        scene_mesh = self.__init_dataset()
            
        self.__init_o3d_elements()
        
        bbox = self.__bbox_visualize.copy()
        frame_first = self.__frames_cache.get()
        c2w = frame_first['c2w'].detach().cpu().numpy()
        agent_sensor = c2w[self.__height_direction[0], 3]
        agent_height_start = agent_sensor - self.__rgbd_sensor.position[self.__height_direction[0]]
        agent_height_end = agent_height_start + self.__dataset_config.agent_height
        
        if self.__height_direction[0] in [1, 2]:
            agent_foot = -agent_height_start
            agent_sensor = -agent_sensor
            agent_head = -agent_height_end
            agent_height_start, agent_height_end = agent_head, agent_foot
        elif self.__height_direction[0] == 0:
            agent_foot = agent_height_start
            agent_head = agent_height_end
        else:
            raise ValueError(f'Invalid height direction: {self.__height_direction}')
        
        if config['mapper']['single_floor']['enable']:
            current_pcd:o3d.geometry.PointCloud = rgbd_to_pointcloud(
                frame_first['rgb'].detach().cpu().numpy(),
                np.ones_like(frame_first['depth'].detach().cpu().numpy(), dtype=np.float32) * self.__rgbd_sensor.depth_max,
                c2w,
                self.__o3d_const_camera_intrinsics_o3c,
                1000,
                self.__rgbd_sensor.depth_max * 2,
                self.__device_o3c).to_legacy()
            current_pcd_bbox:o3d.geometry.AxisAlignedBoundingBox = current_pcd.get_axis_aligned_bounding_box()
            if self.__height_direction[0] in [1, 2]:
                single_floor_height_start = agent_height_start - config['mapper']['single_floor']['expansion']['head']
                single_floor_height_end = agent_height_end + config['mapper']['single_floor']['expansion']['foot']
            elif self.__height_direction[0] == 0:
                single_floor_height_start = agent_height_start - config['mapper']['single_floor']['expansion']['foot']
                single_floor_height_end = agent_height_end + config['mapper']['single_floor']['expansion']['head']
            else:
                raise ValueError(f'Invalid height direction: {self.__height_direction}')
            bbox[self.__height_direction[0]][0] = max(
                single_floor_height_start,
                bbox[self.__height_direction[0]][0],
                current_pcd_bbox.get_min_bound()[self.__height_direction[0]])
            bbox[self.__height_direction[0]][1] = min(
                single_floor_height_end,
                bbox[self.__height_direction[0]][1],
                current_pcd_bbox.get_max_bound()[self.__height_direction[0]])
            assert bbox[self.__height_direction[0]][0] < bbox[self.__height_direction[0]][1], 'Invalid height dimension'
        if self.__frames_cache.empty(): self.__frames_cache.put(frame_first)
        
        bbox:np.ndarray = bbox + self.__bbox_padding *\
            np.reshape(np.ptp(bbox, axis=1), (3, 1)) *\
                np.array([-1, 1])
                
        # NOTE: Get basic information of topdown view
        self.__topdown_info:Dict[str, Union[float, Tuple[float, float], Tuple[Tuple[float, float]], Tuple[np.ndarray, np.ndarray], torch.Tensor, np.ndarray, cv2.Mat]] = {
            'world_dim_index': (
                (self.__height_direction[0] + self.__height_direction[1]) % 3,  # x+ of the topdown view
                (self.__height_direction[0] - self.__height_direction[1]) % 3), # y- of the topdown view
        }
        
        topdown_world_2d_bbox = (
            (bbox[self.__topdown_info['world_dim_index'][0]][0], bbox[self.__topdown_info['world_dim_index'][0]][1]),
            (bbox[self.__topdown_info['world_dim_index'][1]][0], bbox[self.__topdown_info['world_dim_index'][1]][1]))
        
        topdown_world_shape = (
            topdown_world_2d_bbox[0][1] - topdown_world_2d_bbox[0][0],
            topdown_world_2d_bbox[1][1] - topdown_world_2d_bbox[1][0])
        
        self.__topdown_info['world_center'] = (
            (topdown_world_2d_bbox[0][0] + topdown_world_2d_bbox[0][1]) / 2,
            (topdown_world_2d_bbox[1][0] + topdown_world_2d_bbox[1][1]) / 2)
        
        self.__topdown_info['pixel_per_meter'] =\
            config['painter']['grid_map']['pixel_max'] / max(topdown_world_shape)
        self.__topdown_info['meter_per_pixel'] = 1 / self.__topdown_info['pixel_per_meter']
        
        topdown_info = config_topdown_info(
            self.__height_direction,
            self.__topdown_info['world_dim_index'],
            topdown_world_shape,
            self.__topdown_info['world_center'],
            self.__topdown_info['meter_per_pixel'],
            agent_foot,
            agent_sensor,
            agent_head,
            20,
            2)
        
        self.__topdown_info.update(topdown_info)
        
        self.__topdown_info['world_topdown_origin'] = c2w_topdown_to_world(np.zeros(2), self.__topdown_info, 0)
        
        self.__topdown_info['free_map_sdf'] = None
        self.__topdown_info['free_map_binary'] = None
        self.__topdown_info['free_map_cv2'] = None
        self.__topdown_info['free_map_binary_cv2'] = None
        self.__topdown_info['visible_map_sdf'] = None
        self.__topdown_info['visible_map_binary'] = None
        self.__topdown_info['visible_map_cv2'] = None
        self.__topdown_info['visible_map_binary_cv2'] = None
        self.__topdown_info['translation'] = None
        self.__topdown_info['translation_pixel'] = None
        self.__topdown_info['rotation_vector'] = None
        self.__topdown_info['horizon_bbox'] = None
        
        self.__precise_info:Dict[str, Union[float, Tuple[float, float], Tuple[Tuple[float, float]], Tuple[np.ndarray, np.ndarray], torch.Tensor, np.ndarray, cv2.Mat]] = {
            'size': config['painter']['precise_map']['size'],
            'core_size': config['painter']['precise_map']['core_size'],
            'meter_per_pixel': config['painter']['precise_map']['meter_per_pixel'],
            'free_map_x': None,
            'free_map_y': None,
            'free_map_sdf': None,
            'free_map_binary': None,
            'free_map_cv2': None,
            'free_map_binary_cv2': None,
            'center': None}
        
        rospy.loginfo(f'meter per pixel of topdown: {self.__topdown_info["meter_per_pixel"]}')
        rospy.loginfo(f'meter per pixel of precise: {self.__precise_info["meter_per_pixel"]}')
        if self.__precise_info['meter_per_pixel'] > self.__topdown_info['meter_per_pixel']:
            self.__precise_info['meter_per_pixel'] = self.__topdown_info['meter_per_pixel']
            rospy.logwarn(f'Precise map meter per pixel is larger than topdown map, set to {self.__precise_info["meter_per_pixel"]}')
        self.__topdown_map_resize_scale = self.__topdown_info['meter_per_pixel'] / self.__precise_info['meter_per_pixel']
        
        self.__agent_step_size_pixel = self.__dataset_config.agent_forward_step_size / self.__topdown_info['meter_per_pixel']
        self.__pixel_as_arrived = self.__agent_step_size_pixel * self.__step_num_as_arrived
        
        precise_info = config_topdown_info(
            self.__height_direction,
            self.__topdown_info['world_dim_index'],
            (self.__precise_info['size'], self.__precise_info['size']),
            (0.0, 0.0),
            self.__precise_info['meter_per_pixel'],
            agent_foot,
            agent_sensor,
            agent_head,
            int(np.ceil(4 * np.linalg.norm(agent_foot - agent_sensor) / self.__precise_info['meter_per_pixel'])),
            2)
        
        self.__precise_info.update(precise_info)
        
        self.__precise_topdown_map_cv2 = None
                
        self.__clusters_info:Dict[str, Union[np.ndarray, List[np.ndarray], List[o3d.geometry.TriangleMesh]]] = {
            'targets_frustums_transform': None,
            'clusters_meshes': None,
            'clusters_area': None,
            'clusters_n_triangles': None,
            'clusters_uncertainty_max': None,
            'clusters_uncertainty_mean': None
        }
        
        self.__current_horizon = None
        
        rospy.Service('get_topdown_config', GetTopdownConfig, self.__get_topdown_config)
            
        bbox_o3d = bbox.copy()
        # XXX: The bounding box for training has to rotate 180 degree around x-axis, then when we get the mesh, we rotate it back, i do not know the reason
        bbox[1, 0], bbox[1, 1] = -bbox[1, 1], -bbox[1, 0]
        bbox[2, 0], bbox[2, 1] = -bbox[2, 1], -bbox[2, 0]
        
        topdown_voxel_grid = self.__topdown_info['world_voxel_grid'].clone()
        topdown_voxel_grid[..., 1] = -topdown_voxel_grid[..., 1]
        topdown_voxel_grid[..., 2] = -topdown_voxel_grid[..., 2]
        precise_voxel_grid = self.__precise_info['world_voxel_grid'].clone()
        precise_voxel_grid[..., 1] = -precise_voxel_grid[..., 1]
        precise_voxel_grid[..., 2] = -precise_voxel_grid[..., 2]
        
        self.__mapper = Mapper(
            config,
            torch.from_numpy(bbox).to(self.__device),
            topdown_voxel_grid.to(self.__device),
            precise_voxel_grid.to(self.__device),
            self.__rgbd_sensor,
            self.__device)
        
        rospy.Service('set_mapper', SetMapper, self.__set_mapper)
        
        self.__kf_trigger:List[np.ndarray] = [np.array([np.nan, np.nan]), np.array([])]
        rospy.Service('set_kf_trigger_poses', SetKFTriggerPoses, self.__set_kf_trigger_poses)
        
        self.__init_window(
            config['mapper']['interval_max_ratio'],
            bbox_o3d[self.__height_direction[0]][(1 - self.__height_direction[1]) // 2],
            (agent_height_start * (1 - self.__height_direction[1]) + agent_height_end * (1 + self.__height_direction[1])) / 2,
            font_id,
            scene_mesh)
        
        self.__update_main_thread.start()
        
    # NOTE: initialization functions
    
    def __init_dataset(self) -> o3d.geometry.TriangleMesh:
        self.__frames_cache:Queue[Dict[str, Union[int, torch.Tensor]]] = Queue(maxsize=1)
        self.__frame_c2w_last = None
        
        self.__get_topdown_flag = self.QueryTopdownFlag.NONE
        self.__get_topdown_condition = threading.Condition()
        self.__get_topdown_service = rospy.Service('get_topdown', GetTopdown, self.__get_topdown)
        self.__query_precise_center = None
        self.__get_precise_condition = threading.Condition()
        self.__get_precise_service = rospy.Service('get_precise', GetPrecise, self.__get_precise)
        
        if self.__local_dataset is None:
            reset_env_service = rospy.ServiceProxy('reset_env', ResetEnv)
            rospy.wait_for_service('reset_env')
            
            reset_env_success:ResetEnvResponse = reset_env_service(ResetEnvRequest())
            
            self.__cmd_vel_publisher = rospy.Publisher('cmd_vel', Twist, queue_size=1)
            get_dataset_config_service = rospy.ServiceProxy('get_dataset_config', GetDatasetConfig)
            rospy.wait_for_service('get_dataset_config')
            
            self.__dataset_config:GetDatasetConfigResponse = get_dataset_config_service(GetDatasetConfigRequest())
        else:
            self.__local_dataset_condition.acquire()
            if self.__local_dataset_state == self.LocalDatasetState.INITIALIZING:
                self.__local_dataset_condition.wait()
            
        self.__results_dir = self.__dataset_config.results_dir
        os.makedirs(self.__results_dir, exist_ok=True)
        self.__pose_data_type = PoseDataType(self.__dataset_config.pose_data_type)
        self.__height_direction = (self.__dataset_config.height_direction // 2, (self.__dataset_config.height_direction % 2) * 2 - 1)
        
        self.__rgbd_sensor = RGBDSensor(
            height=self.__dataset_config.rgbd_height,
            width=self.__dataset_config.rgbd_width,
            fx=self.__dataset_config.rgbd_fx,
            fy=self.__dataset_config.rgbd_fy,
            cx=self.__dataset_config.rgbd_cx,
            cy=self.__dataset_config.rgbd_cy,
            depth_min=self.__dataset_config.rgbd_depth_min,
            depth_max=self.__dataset_config.rgbd_depth_max,
            depth_scale=self.__dataset_config.rgbd_depth_scale,
            position=np.array([
                self.__dataset_config.rgbd_position.x,
                self.__dataset_config.rgbd_position.y,
                self.__dataset_config.rgbd_position.z]),
            downsample_factor=self.__dataset_config.rgbd_downsample_factor)
            
        if self.__local_dataset is None:
            rospy.Subscriber('frames', frame, self.__frame_callback)
            rospy.wait_for_message('frames', frame)
        else:
            if self.__local_dataset_state == self.LocalDatasetState.INITIALIZED:
                self.__local_dataset_condition.wait()
            self.__local_dataset_condition.notify_all()
            self.__local_dataset_condition.release()
        
        if os.path.exists(self.__dataset_config.scene_mesh_url):
            self.__scene_mesh_transform = pose_to_matrix(self.__dataset_config.scene_mesh_transform)
            scene_mesh, self.__bbox_visualize = load_scene_mesh(
                self.__dataset_config.scene_mesh_url,
                self.__scene_mesh_transform)
        else:
            scene_mesh = None
            self.__bbox_visualize = np.array([
                [self.__dataset_config.scene_bound_min.x, self.__dataset_config.scene_bound_max.x],
                [self.__dataset_config.scene_bound_min.y, self.__dataset_config.scene_bound_max.y],
                [self.__dataset_config.scene_bound_min.z, self.__dataset_config.scene_bound_max.z]])
        
        self.__agent_cylinder_mesh:o3d.geometry.TriangleMesh = o3d.geometry.TriangleMesh.create_cylinder(radius=self.__dataset_config.agent_radius, height=self.__dataset_config.agent_height)
        self.__agent_cylinder_mesh.compute_vertex_normals()
        vector_end = np.zeros(3)
        vector_end[self.__height_direction[0]] = self.__height_direction[1]
        self.__agent_cylinder_mesh.rotate(
            rotation_matrix_from_vectors(np.array([0, 0, 1]), vector_end),
            np.zeros(3))
        self.__agent_cylinder_mesh.translate(
            (self.__dataset_config.agent_height / 2 - self.__rgbd_sensor.position[self.__height_direction[0]]) * vector_end)
        self.__agent_cylinder_mesh.paint_uniform_color(CURRENT_AGENT['color'])
        return scene_mesh
    
    def __init_o3d_elements(self):
        self.__device_o3c = o3d.core.Device(self.__device.type, self.__device.index)
        
        # NOTE: Independent Open3D elements
        self.__o3d_meshes:Dict[str, o3d.geometry.TriangleMesh] = {
            'scene_mesh': None,
            'construct_mesh': None,
            'uncertainty_mesh': None,
            'clusters_meshes': None
        }
        self.__o3d_pcd:Dict[str, o3d.t.geometry.PointCloud] = {
            'current_pcd': None,
        }
        
        self.__o3d_const_camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            self.__rgbd_sensor.width,
            self.__rgbd_sensor.height,
            self.__rgbd_sensor.fx,
            self.__rgbd_sensor.fy,
            self.__rgbd_sensor.cx,
            self.__rgbd_sensor.cy)
        self.__o3d_const_camera_intrinsics_o3c = o3d.core.Tensor(self.__o3d_const_camera_intrinsics.intrinsic_matrix, device=self.__device_o3c)
        
        self.__o3d_materials:Dict[str, rendering.MaterialRecord] = {
            'lit_mat': None,
            'lit_mat_transparency': None,
            'unlit_mat': None,
            'unlit_line_mat': None,
            'unlit_line_mat_slim': None,
        }
            
        if not self.__hide_windows:
            self.__o3d_materials['lit_mat'] = rendering.MaterialRecord()
            self.__o3d_materials['lit_mat'].shader = 'defaultLit'
            self.__o3d_materials['lit_mat_transparency'] = rendering.MaterialRecord()
            self.__o3d_materials['lit_mat_transparency'].shader = 'defaultLitTransparency'
            self.__o3d_materials['lit_mat_transparency'].has_alpha = True
            self.__o3d_materials['lit_mat_transparency'].base_color = [1.0, 1.0, 1.0, 0.9]
            self.__o3d_materials['unlit_mat'] = rendering.MaterialRecord()
            self.__o3d_materials['unlit_mat'].shader = 'defaultUnlit'
            self.__o3d_materials['unlit_mat'].sRGB_color = True
            self.__o3d_materials['unlit_line_mat'] = rendering.MaterialRecord()
            self.__o3d_materials['unlit_line_mat'].shader = 'unlitLine'
            self.__o3d_materials['unlit_line_mat'].line_width = 5.0
            self.__o3d_materials['unlit_line_mat_slim'] = rendering.MaterialRecord()
            self.__o3d_materials['unlit_line_mat_slim'].shader = 'unlitLine'
            self.__o3d_materials['unlit_line_mat_slim'].line_width = 2.0
        
    def __init_window(self, interval_max_ratio:float, foot_value:float, head_value:float, font_id:int, scene_mesh:o3d.geometry.TriangleMesh=None):
        kf_every = self.__mapper.get_kf_every()
        assert 0 < kf_every, f'Invalid keyframe every: {kf_every}'
        map_every = self.__mapper.get_map_every()
        assert 0 < map_every, f'Invalid map every: {map_every}'
        update_interval_max = int(interval_max_ratio * max(kf_every, map_every))
        assert interval_max_ratio >= 1.0, f'Invalid interval ratio: {interval_max_ratio}'
        mapping_iters = self.__mapper.get_mapping_iters()
        
        self.__open3d_gui_widget_last_state = dict()
            
        self.__open3d_gui_widget_last_state['height_direction_bound_slider'] = [
            (foot_value * (1 + self.__height_direction[1]) + head_value * (1 - self.__height_direction[1])) / 2,
            (foot_value * (1 - self.__height_direction[1]) + head_value * (1 + self.__height_direction[1])) / 2]
        
        self.__set_planner_state_service = rospy.ServiceProxy('set_planner_state', SetPlannerState)
        rospy.wait_for_service('set_planner_state')
        self.__mesh_color_type = MeshColorType.QUERY_VERTICES_COLOR
        self.__construct_mesh_button_flag = Visualizer.QueryMeshFlag.NONE
        self.__render_rgbd_button_flag = False
        self.__retain_high_uncertainty_ratio_default = 0.2
        self.__remove_mesh_less_triangle_ratio_default = 0.005
        self.__clusters_sort_type = self.ClustersSortType.TRIANGLES_N
        
        if self.__hide_windows:
            self.__global_state_callback(self.__global_state.value, None)
            self.__o3d_meshes['scene_mesh'] = o3d.geometry.TriangleMesh(scene_mesh)
            self.__mapper.set_mesh_voxels(self.__voxel_size_eval)
            
            return
        
        else:
            # NOTE: Initialize GUI
            self.__window:gui.Window = gui.Application.instance.create_window(PROJECT_NAME, 2560, 1440)
            self.__window.show(False)
            
            em = self.__window.theme.font_size
        
            spacing = int(np.round(0.25 * em))
            vspacing = int(np.round(0.5 * em))
            
            margins = gui.Margins(vspacing)
            self.__panel_control = gui.Vert(spacing, margins)
            self.__panel_visualize = gui.Vert(spacing, margins)
            
            self.__widget_3d = gui.SceneWidget()
            self.__widget_3d.scene = rendering.Open3DScene(self.__window.renderer)
            self.__widget_3d.scene.set_background([1.0, 1.0, 1.0, 1.0])
            self.__widget_3d.scene.scene.set_sun_light([-0.2, 1.0 ,0.2], [1.0, 1.0, 1.0], 70000)
            self.__widget_3d.scene.scene.enable_sun_light(True)
            self.__widget_3d.set_on_key(self.__widget_3d_on_key)
            
            # NOTE: Widgets for control panel
            global_state_vgrid = gui.VGrid(2, spacing, gui.Margins(0, 0, em, 0))
            
            global_state_vgrid.add_child(gui.Label('Global State'))
            self.__global_state_combobox = gui.Combobox()
            for global_state in self.__global_states_selectable:
                self.__global_state_combobox.add_item(global_state.value)
            self.__global_state_combobox.selected_text = self.__global_state.value
            self.__global_state_combobox.set_on_selection_changed(self.__global_state_callback)
            global_state_vgrid.add_child(self.__global_state_combobox)
            self.__panel_control.add_child(global_state_vgrid)
            self.__global_state_callback(self.__global_state_combobox.selected_text, None)
            
            self.__construct_mesh_button = gui.Button('Construct Mesh')
            self.__construct_mesh_button.horizontal_padding_em = 0
            self.__construct_mesh_button.vertical_padding_em = 0.1
            def construct_mesh_button_callback():
                self.__construct_mesh_button_flag = Visualizer.QueryMeshFlag.VISUALIZATION
                return gui.Button.HANDLED
            self.__construct_mesh_button.set_on_clicked(construct_mesh_button_callback)
            self.__panel_control.add_child(self.__construct_mesh_button)
            
            self.__panel_control.add_fixed(vspacing)
            
            # TODO: Add log information label
            
            self.__panel_control.add_fixed(vspacing)
            
            if self.__local_dataset is not None:
                self.__panel_control.add_child(gui.Label('Local Dataset Info'))
                self.__panel_control.add_child(self.__local_dataset_label)
            
            self.__panel_control.add_child(gui.Label('3D Visualization Settings'))
            
            panel_control_vgrid = gui.VGrid(2, spacing, gui.Margins(em, 0, em, 0))
            
            panel_control_vgrid.add_child(gui.Label('    Mapper Configurations'))
            # TODO: use checkbox to close all mapper configurations
            panel_control_vgrid.add_child(gui.Label(''))
            
            panel_control_vgrid.add_child(gui.Label('        Visualize Mesh'))
            self.__visualize_construct_mesh_box = gui.Checkbox('')
            self.__visualize_construct_mesh_box.checked = True
            self.__visualize_construct_mesh_box.set_on_checked(lambda checked: self.__widget_3d.scene.show_geometry('construct_mesh', checked))
            panel_control_vgrid.add_child(self.__visualize_construct_mesh_box)
            
            panel_control_vgrid.add_child(gui.Label('        Visualize Uncertainty'))
            self.__visualize_uncertainty_mesh_box = gui.Checkbox('')
            self.__visualize_uncertainty_mesh_box.checked = False
            self.__visualize_uncertainty_mesh_box.set_on_checked(lambda checked: self.__widget_3d.scene.show_geometry('uncertainty_mesh', checked))
            panel_control_vgrid.add_child(self.__visualize_uncertainty_mesh_box)
            
            panel_control_vgrid.add_child(gui.Label('        Retain First High Uncertainty'))
            self.__retain_high_uncertainty_ratio_slider = gui.Slider(gui.Slider.DOUBLE)
            self.__retain_high_uncertainty_ratio_slider.set_limits(0.0, 1.0)
            self.__retain_high_uncertainty_ratio_slider.double_value = self.__retain_high_uncertainty_ratio_default
            panel_control_vgrid.add_child(self.__retain_high_uncertainty_ratio_slider)

            panel_control_vgrid.add_child(gui.Label('        Remove Mesh Less Triangles'))
            self.__remove_mesh_less_triangle_ratio_slider = gui.Slider(gui.Slider.DOUBLE)
            self.__remove_mesh_less_triangle_ratio_slider.set_limits(0.0, 1.0)
            self.__remove_mesh_less_triangle_ratio_slider.double_value = self.__remove_mesh_less_triangle_ratio_default
            panel_control_vgrid.add_child(self.__remove_mesh_less_triangle_ratio_slider)
            
            panel_control_vgrid.add_child(gui.Label('        Visualize Clusters and Targets'))
            self.__visualize_clusters_box = gui.Checkbox('')
            self.__visualize_clusters_box.checked = True
            def visualize_clusters_callback(checked:bool):
                self.__widget_3d.scene.show_geometry('clusters_meshes', checked)
                cluster_count = 0
                while self.__widget_3d.scene.has_geometry(f'target_frustum_{cluster_count}'):
                    self.__widget_3d.scene.show_geometry(f'target_frustum_{cluster_count}', checked)
                    cluster_count += 1
            self.__visualize_clusters_box.set_on_checked(visualize_clusters_callback)
            panel_control_vgrid.add_child(self.__visualize_clusters_box)
            
            panel_control_vgrid.add_child(gui.Label('        Sort Clusters by'))
            self.__sort_clusters_by_combobox = gui.Combobox()
            for clusters_sort_type in self.ClustersSortType:
                self.__sort_clusters_by_combobox.add_item(clusters_sort_type.value)
            self.__sort_clusters_by_combobox.selected_text = self.__clusters_sort_type.value
            self.__sort_clusters_by_combobox.set_on_selection_changed(self.__clusters_sort_by_combobox_callback)
            panel_control_vgrid.add_child(self.__sort_clusters_by_combobox)
            
            panel_control_vgrid.add_child(gui.Label('        Meshing Voxel Size'))
            self.__voxel_size_slider = gui.Slider(gui.Slider.DOUBLE)
            bbox_dim_min = np.min(np.ptp(self.__bbox_visualize, axis=1))
            voxel_size_max = max(bbox_dim_min / 10, min(bbox_dim_min, self.__voxel_size_eval * 2), self.__voxel_size_eval)
            self.__voxel_size_slider.set_limits(self.__voxel_size_final, voxel_size_max)
            self.__voxel_size_slider.double_value = self.__voxel_size_eval
            self.__voxel_size_slider.set_on_value_changed(lambda value: self.__mapper.set_mesh_voxels(value))
            self.__mapper.set_mesh_voxels(self.__voxel_size_slider.double_value)
            panel_control_vgrid.add_child(self.__voxel_size_slider)
            
            panel_control_vgrid.add_child(gui.Label('        Coloring Type'))
            self.__mesh_color_combobox = gui.Combobox()
            for mesh_color_type in MeshColorType:
                self.__mesh_color_combobox.add_item(mesh_color_type.value)
            self.__mesh_color_combobox.selected_text = MeshColorType.QUERY_VERTICES_COLOR.value
            def mesh_color_combobox_callback(color_type_name:str, color_type_index:int):
                self.__mesh_color_type = MeshColorType(color_type_name)
                return gui.Combobox.HANDLED
            self.__mesh_color_combobox.set_on_selection_changed(mesh_color_combobox_callback)
            mesh_color_combobox_callback(self.__mesh_color_combobox.selected_text, None)
            panel_control_vgrid.add_child(self.__mesh_color_combobox)
            
            panel_control_vgrid.add_child(gui.Label('        Map Every'))
            self.__map_every_slider = gui.Slider(gui.Slider.INT)
            self.__map_every_slider.set_limits(1, update_interval_max)
            self.__map_every_slider.int_value = map_every
            self.__map_every_slider.set_on_value_changed(lambda value: self.__mapper.set_map_every(value))
            panel_control_vgrid.add_child(self.__map_every_slider)
            
            panel_control_vgrid.add_child(gui.Label('        Keyframe Every'))
            self.__kf_every_slider = gui.Slider(gui.Slider.INT)
            self.__kf_every_slider.set_limits(1, update_interval_max)
            self.__kf_every_slider.int_value = kf_every
            self.__kf_every_slider.set_on_value_changed(lambda value: self.__mapper.set_kf_every(value))
            panel_control_vgrid.add_child(self.__kf_every_slider)
            
            panel_control_vgrid.add_child(gui.Label('        Training iters per step'))
            self.__train_iters_per_step_slider = gui.Slider(gui.Slider.INT)
            self.__train_iters_per_step_slider.set_limits(1, 5 * mapping_iters)
            self.__train_iters_per_step_slider.int_value = mapping_iters
            self.__train_iters_per_step_slider.set_on_value_changed(lambda value: self.__mapper.set_mapping_iters(value))
            panel_control_vgrid.add_child(self.__train_iters_per_step_slider)
            
            panel_control_vgrid.add_child(gui.Label('        Auto Meshing'))
            self.__auto_meshing_box = gui.Checkbox('')
            self.__auto_meshing_box.checked = self.__global_state in [GlobalState.MANUAL_CONTROL, GlobalState.MANUAL_PLANNING]
            self.__auto_meshing_box.set_on_checked(lambda checked: set_enable(self.__auto_meshing_every_slider, checked))
            panel_control_vgrid.add_child(self.__auto_meshing_box)
            
            panel_control_vgrid.add_child(gui.Label('        Auto Meshing Every'))
            self.__auto_meshing_every_slider = gui.Slider(gui.Slider.INT)
            self.__auto_meshing_every_slider.set_limits(1, update_interval_max)
            self.__auto_meshing_every_slider.int_value = int(self.__kf_every_slider.int_value / 2)
            self.__auto_meshing_every_slider.enabled = self.__auto_meshing_box.checked
            panel_control_vgrid.add_child(self.__auto_meshing_every_slider)
            
            panel_control_vgrid.add_child(gui.Label('    Planner Configurations'))
            # TODO: use checkbox to close all planner configurations
            panel_control_vgrid.add_child(gui.Label(''))
            
            panel_control_vgrid.add_child(gui.Label('    Global Status'))
            # TODO: use checkbox to close all global states
            panel_control_vgrid.add_child(gui.Label(''))
            
            origin_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
            self.__widget_3d.scene.add_geometry('origin_mesh', origin_mesh, self.__o3d_materials['lit_mat'])
            panel_control_vgrid.add_child(gui.Label('        Origin Mesh'))
            origin_mesh_box = gui.Checkbox('')
            origin_mesh_box.checked = True
            origin_mesh_box.set_on_checked(lambda checked: self.__widget_3d.scene.show_geometry('origin_mesh', checked))
            panel_control_vgrid.add_child(origin_mesh_box)
            
            if scene_mesh is not None:
                self.__o3d_meshes['scene_mesh'] = o3d.geometry.TriangleMesh(scene_mesh)
                panel_control_vgrid.add_child(gui.Label('        Ground Truth Mesh'))
                self.__scene_mesh_box = gui.Checkbox('')
                self.__scene_mesh_box.checked = False
                self.__scene_mesh_box.set_on_checked(lambda checked: self.__widget_3d.scene.show_geometry('scene_mesh', checked))
                panel_control_vgrid.add_child(self.__scene_mesh_box)
            else:
                self.__scene_mesh_box = None
                
            panel_control_vgrid.add_child(gui.Label('        H Lower Bound'))
            self.__height_direction_lower_bound_slider = gui.Slider(gui.Slider.DOUBLE)
            self.__height_direction_lower_bound_slider.set_limits(
                self.__bbox_visualize[self.__height_direction[0]][0],
                self.__open3d_gui_widget_last_state['height_direction_bound_slider'][1])
            self.__height_direction_lower_bound_slider.double_value = self.__open3d_gui_widget_last_state['height_direction_bound_slider'][0]
            self.__height_direction_lower_bound_slider.set_on_value_changed(
                lambda value: self.__height_direction_bound_slider_callback(value, 0))
            panel_control_vgrid.add_child(self.__height_direction_lower_bound_slider)
            
            panel_control_vgrid.add_child(gui.Label('        H Upper Bound'))
            self.__height_direction_upper_bound_slider = gui.Slider(gui.Slider.DOUBLE)
            self.__height_direction_upper_bound_slider.set_limits(
                self.__open3d_gui_widget_last_state['height_direction_bound_slider'][0],
                self.__bbox_visualize[self.__height_direction[0]][1])
            self.__height_direction_upper_bound_slider.double_value = self.__open3d_gui_widget_last_state['height_direction_bound_slider'][1]
            self.__height_direction_upper_bound_slider.set_on_value_changed(
                lambda value: self.__height_direction_bound_slider_callback(value, 1))
            panel_control_vgrid.add_child(self.__height_direction_upper_bound_slider)
            
            if scene_mesh is not None:
                self.__update_mesh('scene_mesh', self.__scene_mesh_box.checked, self.__o3d_materials['lit_mat'])
            
            panel_control_vgrid.add_child(gui.Label('    Local Status'))
            # TODO: use checkbox to close all local states
            panel_control_vgrid.add_child(gui.Label(''))
            
            panel_control_vgrid.add_child(gui.Label('        Current Frustum'))
            self.__current_frustum_box = gui.Checkbox('')
            self.__current_frustum_box.checked = True
            self.__current_frustum_box.set_on_checked(lambda checked: self.__widget_3d.scene.show_geometry('current_frustum', checked))
            panel_control_vgrid.add_child(self.__current_frustum_box)
            
            panel_control_vgrid.add_child(gui.Label('        Current Agent'))
            self.__current_agent_box = gui.Checkbox('')
            self.__current_agent_box.checked = True
            self.__current_agent_box.set_on_checked(lambda checked: self.__widget_3d.scene.show_geometry('current_agent', checked))
            panel_control_vgrid.add_child(self.__current_agent_box)
            
            panel_control_vgrid.add_child(gui.Label('        Current Horizon'))
            self.__current_horizon_box = gui.Checkbox('')
            self.__current_horizon_box.checked = True
            self.__current_horizon_box.set_on_checked(lambda checked: self.__widget_3d.scene.show_geometry('current_horizon', checked))
            panel_control_vgrid.add_child(self.__current_horizon_box)
            
            panel_control_vgrid.add_child(gui.Label('        Current PCD'))
            self.__current_pcd_box = gui.Checkbox('')
            self.__current_pcd_box.checked = False
            self.__current_pcd_box.set_on_checked(lambda checked: self.__widget_3d.scene.show_geometry('current_pcd', checked))
            panel_control_vgrid.add_child(self.__current_pcd_box)
                
            self.__panel_control.add_child(panel_control_vgrid)
                
            # NOTE: Widgets for visualize panel
            panel_visualize_tabs = gui.TabControl()
            panel_visualize_tab_margin = gui.Margins(0, int(np.round(0.5 * em)), em, em)
            
            tab_live_view = gui.ScrollableVert(0, panel_visualize_tab_margin)
            
            tab_live_view.add_child(gui.Label('RGBD Live Image'))
            self.__rgbd_live_image = gui.ImageWidget()
            tab_live_view.add_child(self.__rgbd_live_image)
            tab_live_view.add_fixed(vspacing)
            
            render_grid = gui.VGrid(2, spacing, gui.Margins(0, 0, 0, 0))
            render_grid.add_child(gui.Label('Rendered RGBD Image'))
            self.__render_box = gui.Checkbox('')
            self.__render_box.checked = True
            def render_box_callback(checked:bool):
                self.__render_every_slider.enabled = checked
                return gui.Checkbox.HANDLED
            self.__render_box.set_on_checked(render_box_callback)
            render_grid.add_child(self.__render_box)
            
            render_grid.add_child(gui.Label('    Render Every'))
            self.__render_every_slider = gui.Slider(gui.Slider.INT)
            self.__render_every_slider.set_limits(1, update_interval_max)
            self.__render_every_slider.int_value = 1
            self.__render_every_slider.enabled = self.__render_box.checked
            render_grid.add_child(self.__render_every_slider)
            
            tab_live_view.add_child(render_grid)
            
            render_rgbd_button = gui.Button('Render RGBD')
            render_rgbd_button.horizontal_padding_em = 0
            render_rgbd_button.vertical_padding_em = 0.1
            def render_rgbd_button_callback():
                self.__render_rgbd_button_flag = True
                return gui.Button.HANDLED
            render_rgbd_button.set_on_clicked(render_rgbd_button_callback)
            tab_live_view.add_child(render_rgbd_button)
            
            self.__rgbd_render_image = gui.ImageWidget()
            tab_live_view.add_child(self.__rgbd_render_image)
            tab_live_view.add_fixed(vspacing)
            
            get_topdown_button = gui.Button('Get Topdown')
            get_topdown_button.horizontal_padding_em = 0
            get_topdown_button.vertical_padding_em = 0.1
            def get_topdown_button_callback():
                self.__get_topdown_flag = self.QueryTopdownFlag.MANUAL
                return gui.Button.HANDLED
            get_topdown_button.set_on_clicked(get_topdown_button_callback)
            tab_live_view.add_child(get_topdown_button)
            
            tab_live_view.add_child(gui.Label('Free Space Map (SDF)'))
            self.__topdown_free_map_image = gui.ImageWidget()
            tab_live_view.add_child(self.__topdown_free_map_image)
            tab_live_view.add_fixed(vspacing)
            
            tab_live_view.add_child(gui.Label('Visible Map (SDF)'))
            self.__topdown_visible_map_image = gui.ImageWidget()
            tab_live_view.add_child(self.__topdown_visible_map_image)
            tab_live_view.add_fixed(vspacing)
            
            tab_live_view.add_child(gui.Label('Free Space Map (binary)'))
            self.__topdown_free_map_image_binary = gui.ImageWidget()
            tab_live_view.add_child(self.__topdown_free_map_image_binary)
            tab_live_view.add_fixed(vspacing)
            
            tab_live_view.add_child(gui.Label('Visible Map (binary)'))
            self.__topdown_visible_map_image_binary = gui.ImageWidget()
            tab_live_view.add_child(self.__topdown_visible_map_image_binary)
            tab_live_view.add_fixed(vspacing)
            
            tab_live_view.add_child(gui.Label('Precise Map (SDF)'))
            self.__precise_free_map_image = gui.ImageWidget()
            tab_live_view.add_child(self.__precise_free_map_image)
            tab_live_view.add_fixed(vspacing)
            
            tab_live_view.add_child(gui.Label('Precise & Topdown Map (SDF)'))
            self.__precise_topdown_map_image = gui.ImageWidget()
            tab_live_view.add_child(self.__precise_topdown_map_image)
            tab_live_view.add_fixed(vspacing)
            
            panel_visualize_tabs.add_tab('Live View', tab_live_view)
            
            self.__panel_visualize.add_child(panel_visualize_tabs)
            
            # TODO: Use the timing panel
            self.__panel_timing = gui.Vert(spacing, gui.Margins(em, 0.5 * em, em, em))
            self.__label_timing = gui.Label('')
            self.__panel_timing.add_child(self.__label_timing)

            # TODO: Use the warning panel
            self.__panel_warning = gui.Vert(spacing, gui.Margins(em, em, em, em))
            self.__dialog_warning = gui.Dialog('')
            self.__dialog_warning.add_child(gui.Label(''))
            self.__panel_warning.add_child(self.__dialog_warning)
            self.__panel_warning.visible = False
            
            self.__window.add_child(self.__panel_control)
            self.__window.add_child(self.__widget_3d)
            self.__window.add_child(self.__panel_visualize)
            self.__window.add_child(self.__panel_timing)
            self.__window.add_child(self.__panel_warning)
            
            self.__window.set_on_layout(self.__window_on_layout)
            self.__window.set_on_close(self.__window_on_close)
            
            # NOTE: Setup camera
            center = np.average(self.__bbox_visualize, axis=1)
            center[self.__height_direction[0]] = 0
            bbox = o3d.geometry.AxisAlignedBoundingBox(self.__bbox_visualize[:, 0], self.__bbox_visualize[:, 1])
            self.__widget_3d.setup_camera(60.0, bbox, center)
            height_location = np.max(np.ptp(self.__bbox_visualize, axis=1))
            center_bias = np.zeros(3)
            center_bias[self.__height_direction[0]] = self.__height_direction[1] * (height_location - 1)
            center_bias[(self.__height_direction[0] + 1) % 3] = 5 * self.__height_direction[1]
            up_vector = np.zeros(3)
            up_vector[(self.__height_direction[0] + 1) % 3] = -self.__height_direction[1]
            self.__widget_3d.look_at(center, center + center_bias, up_vector)
            
            self.__window.show(True)
        
    # NOTE: main function
    
    def __update_main(self):
        # TODO: Initialize GUI
        frame_id = None
        frame_last_received = None
        meshing_last_frame_id = -np.inf
        rendering_rgbd_last_frame_id = -np.inf
        
        black = np.zeros((100, 100))
        
        while self.__global_state != GlobalState.QUIT:
            timing_logical_operations = start_timing()
            if self.__global_state in [GlobalState.REPLAY, GlobalState.AUTO_PLANNING, GlobalState.MANUAL_PLANNING, GlobalState.MANUAL_CONTROL]:
                # NOTE: Get observation
                if self.__frames_cache.empty():
                    frame_current = None
                else:
                    frame_current = self.__frames_cache.get()
                    
                    if frame_id is None:
                        frame_id = 0
                        self.__update_ui_frame(frame_current)
                        self.__get_topdown_flag = self.QueryTopdownFlag.MANUAL
                    else:
                        frame_id += 1
                    frame_current['frame_id'] = frame_id
                    frame_last_received = frame_current.copy()
            else:
                frame_current = None
                
            assert frame_id is not None, 'Initialize failed'
            
            kf_update_flag = False
            # NOTE: Trigger kf save
            kf_trigger = self.__kf_trigger.copy()
            if len(kf_trigger[1]) > 0:
                if np.linalg.norm(self.__topdown_info['translation'] - kf_trigger[0]) < self.__pixel_as_arrived:
                    current_orientation = np.arctan2(self.__topdown_info['rotation_vector'][1], self.__topdown_info['rotation_vector'][0])
                    diff_orientations = (np.degrees(kf_trigger[1] - current_orientation) + 180) % 360 - 180
                    diff_orientations_condition = np.abs(diff_orientations) <= self.__dataset_config.agent_turn_angle / 2
                    if np.any(diff_orientations_condition):
                        rospy.logdebug(f'KF triggered, {len(kf_trigger[1])} at {kf_trigger[0]}')
                        kf_trigger[1] = kf_trigger[1][np.logical_not(diff_orientations_condition)]
                        rospy.logdebug(f'KF triggered, {len(kf_trigger[1])} left')
                        self.__kf_trigger = kf_trigger
                        kf_update_flag = True
                
            if self.__global_state in [GlobalState.REPLAY, GlobalState.AUTO_PLANNING, GlobalState.MANUAL_PLANNING, GlobalState.MANUAL_CONTROL]:
                # NOTE: Train model
                mapper_state = self.__mapper.run(
                    frame_current,
                    kf_update_flag)
            elif self.__global_state == GlobalState.POST_PROCESSING:
                mapper_state = self.__mapper.run(
                    None,
                    kf_update_flag)
            else:
                mapper_state = MapperState.IDLE
            
            construct_mesh_flag = self.__construct_mesh_button_flag
            if not self.__hide_windows:
                if self.__auto_meshing_box.checked and\
                    frame_id % self.__auto_meshing_every_slider.int_value == 0 and\
                        frame_id > meshing_last_frame_id:
                    construct_mesh_flag = Visualizer.QueryMeshFlag.VISUALIZATION
            if self.__global_state in [GlobalState.REPLAY, GlobalState.AUTO_PLANNING, GlobalState.MANUAL_PLANNING, GlobalState.MANUAL_CONTROL]:
                if self.__get_topdown_flag in [self.QueryTopdownFlag.ARRIVED, self.QueryTopdownFlag.MANUAL] and\
                    self.__mapper.is_ready_for_uncertainty():
                    construct_mesh_flag = Visualizer.QueryMeshFlag.PLANNING

            # NOTE: Update mesh and uncertainty
            get_uncertainty_flag = construct_mesh_flag == Visualizer.QueryMeshFlag.PLANNING
            if construct_mesh_flag in [Visualizer.QueryMeshFlag.PLANNING, Visualizer.QueryMeshFlag.VISUALIZATION]:
                timing_get_mesh = start_timing()
                self.__o3d_meshes['construct_mesh'], vertices_gradient, vertices_uncertainty = self.__mapper.get_mesh(
                    self.__mesh_color_type,
                    get_uncertainty_flag)
                
                if get_uncertainty_flag:
                    assert vertices_gradient is not None and vertices_uncertainty is not None, 'Invalid vertices gradient or uncertainty'
                    vertices_uncertainty_np = vertices_uncertainty.detach().cpu().numpy()
                    vertices_uncertainty_normalize = (vertices_uncertainty_np - np.min(vertices_uncertainty_np)) / np.ptp(vertices_uncertainty_np)
                    colormap = cm.get_cmap(UNCERTAINTY_MESH['colormap'])
                    vertices_uncertainty_color = colormap(vertices_uncertainty_normalize)[:, :3]
                    self.__o3d_meshes['uncertainty_mesh'] = o3d.geometry.TriangleMesh(self.__o3d_meshes['construct_mesh'])
                    self.__o3d_meshes['uncertainty_mesh'].vertex_colors = o3d.utility.Vector3dVector(vertices_uncertainty_color)
                    
                    # TODO: Planning Node get the mesh and uncertainty, calculate frontiers in it
                    vertices_gradient_np = vertices_gradient.detach().cpu().numpy()
                    uncertainty_mesh_for_frustums, vertices_height_condition = self.__cut_mesh_by_height(o3d.geometry.TriangleMesh(self.__o3d_meshes['construct_mesh']))
                    clusters_vertices_gradient:np.ndarray = vertices_gradient_np[np.invert(vertices_height_condition)]
                    clusters_vertices_uncertainty:np.ndarray = vertices_uncertainty_np[np.invert(vertices_height_condition)]
                    uncertainty_mesh_for_frustums.transform(np.linalg.inv(self.__mapper.get_construct_mesh_transform_matrix()))
                    
                    uncertainty_threshold = np.percentile(
                        clusters_vertices_uncertainty,
                        100 * (1 - (self.__retain_high_uncertainty_ratio_default if self.__hide_windows else self.__retain_high_uncertainty_ratio_slider.double_value)))
                    vertices_uncertainty_condition = clusters_vertices_uncertainty <= uncertainty_threshold
                    uncertainty_mesh_for_frustums.remove_vertices_by_mask(vertices_uncertainty_condition)
                    clusters_vertices_gradient:np.ndarray = clusters_vertices_gradient[np.invert(vertices_uncertainty_condition)]
                    clusters_vertices_uncertainty:np.ndarray = clusters_vertices_uncertainty[np.invert(vertices_uncertainty_condition)]
                    
                    triangles_number_threshold = int((self.__remove_mesh_less_triangle_ratio_default if self.__hide_windows else self.__remove_mesh_less_triangle_ratio_slider.double_value) * np.array(uncertainty_mesh_for_frustums.triangles).shape[0])
                    triangles_clusters, clusters_n_triangles, clusters_area = uncertainty_mesh_for_frustums.cluster_connected_triangles()
                    # NOTE: triangle_clusters shape is (n_triangles, ), the value represents the cluster index
                    triangles_clusters = np.array(triangles_clusters)
                    
                    # NOTE: clusters_n_triangles shape is (n_clusters, ), the value represents the number of triangles in the cluster
                    # NOTE: clusters_area shape is (n_clusters, ), the value represents the area of the cluster
                    
                    clusters_index = np.unique(triangles_clusters)
                    assert np.allclose(clusters_index, np.arange(clusters_index.shape[0])), 'Invalid clusters index'
                    clusters_n_triangles = np.array(clusters_n_triangles)
                    clusters_area = np.array(clusters_area)
                    clusters_n_triangles_condition = clusters_n_triangles >= triangles_number_threshold
                    
                    # NOTE: filt clusters by its number of triangles
                    clusters_index_remain:np.ndarray = clusters_index[clusters_n_triangles_condition]
                    clusters_area_remain:np.ndarray = clusters_area[clusters_n_triangles_condition]
                    clusters_n_triangles_remain:np.ndarray = clusters_n_triangles[clusters_n_triangles_condition]
                    
                    clusters_meshes = []
                    clusters_targets_frustums = []
                    clusters_uncertainty_max = []
                    clusters_uncertainty_mean = []
                    
                    # NOTE: split meshes and get frustums
                    for cluster_index in clusters_index_remain:
                        uncertainty_mesh_triangles_remove_mask = triangles_clusters != cluster_index
                        cluster_mesh = o3d.geometry.TriangleMesh(uncertainty_mesh_for_frustums)
                        cluster_mesh.remove_triangles_by_mask(uncertainty_mesh_triangles_remove_mask)
                        cluster_mesh_vertices_index = np.unique(np.array(cluster_mesh.triangles))
                        cluster_mesh_vertices_gradient:np.ndarray = clusters_vertices_gradient[cluster_mesh_vertices_index]
                        cluster_mesh_vertices_uncertainty:np.ndarray = clusters_vertices_uncertainty[cluster_mesh_vertices_index]
                        cluster_mesh_vertices:np.ndarray = np.array(cluster_mesh.vertices)[cluster_mesh_vertices_index]
                        
                        cluster_mesh_vertices_shifted:np.ndarray = cluster_mesh_vertices +\
                            self.__targets_frustums_shift_length * cluster_mesh_vertices_gradient
                            
                        # FIXME: The frustum direction is inverse
                        cluster_target_frustum_direction = -cluster_mesh_vertices_gradient.mean(0)
                        cluster_target_frustum_position = cluster_mesh_vertices_shifted.mean(0)
                        
                        # XXX: Maybe fail in experiment
                        roll = np.arctan2(cluster_target_frustum_direction[1], cluster_target_frustum_direction[0])
                        pitch = np.arctan2(cluster_target_frustum_direction[0], cluster_target_frustum_direction[2])
                        yaw = np.arctan2(cluster_target_frustum_direction[1], cluster_target_frustum_direction[2])
                        
                        cluster_target_frustum_transform = np.eye(4)
                        cluster_target_frustum_transform[:3, :3] = quaternion.as_rotation_matrix(
                            quaternion.from_euler_angles(roll, pitch, yaw))
                        cluster_target_frustum_transform[:3, 3] = cluster_target_frustum_position
                        
                        # XXX: The bounding box for training has to rotate 180 degree around x-axis, then when we get the mesh, we rotate it back, i do not know the reason
                        cluster_mesh.transform(self.__mapper.get_construct_mesh_transform_matrix())
                        
                        clusters_targets_frustums.append(cluster_target_frustum_transform)
                        clusters_meshes.append(cluster_mesh)
                        clusters_uncertainty_max.append(cluster_mesh_vertices_uncertainty.max())
                        clusters_uncertainty_mean.append(cluster_mesh_vertices_uncertainty.mean())
                    
                    self.__clusters_info['targets_frustums_transform'] = clusters_targets_frustums
                    self.__clusters_info['clusters_meshes'] = clusters_meshes
                    self.__clusters_info['clusters_area'] = clusters_area_remain
                    self.__clusters_info['clusters_n_triangles'] = clusters_n_triangles_remain
                    self.__clusters_info['clusters_uncertainty_max'] = np.array(clusters_uncertainty_max)
                    self.__clusters_info['clusters_uncertainty_mean'] = np.array(clusters_uncertainty_mean)
                    
                self.__construct_mesh_button_flag = Visualizer.QueryMeshFlag.NONE
                meshing_last_frame_id = frame_id
                rospy.logdebug(f'Get mesh used {end_timing(*timing_get_mesh):.2f} ms, with{"" if get_uncertainty_flag else "out"} uncertainty')
                
            # NOTE: Update topdown map
            if (self.__get_topdown_flag in [self.QueryTopdownFlag.ARRIVED, self.QueryTopdownFlag.RUNNING] or\
                construct_mesh_flag in [Visualizer.QueryMeshFlag.PLANNING, Visualizer.QueryMeshFlag.VISUALIZATION]):
                topdown_voxel_grid_sdf = self.__mapper.get_topdown_voxel_grid_sdf()
                topdown_free_map:torch.Tensor = torch.min(topdown_voxel_grid_sdf, dim=self.__height_direction[0])[0]
                topdown_visible_map:torch.Tensor = topdown_voxel_grid_sdf.select(self.__height_direction[0], self.__topdown_info['body_sample_num'])

                topdown_free_map, topdown_visible_map = adjust_topdown_maps(
                    [topdown_free_map, topdown_visible_map],
                    self.__topdown_info['world_dim_index'],
                    self.__height_direction)
                    
                self.__topdown_info['free_map_sdf'] = topdown_free_map.detach().cpu().numpy()
                self.__topdown_info['free_map_binary'] = (topdown_free_map > 0).type(torch.uint8).detach().cpu().numpy()
                self.__topdown_info['visible_map_sdf'] = topdown_visible_map.detach().cpu().numpy()
                self.__topdown_info['visible_map_binary'] = (topdown_visible_map > 0).type(torch.uint8).detach().cpu().numpy()
                
                topdown_free_map_vis = (topdown_free_map / topdown_free_map.abs().max() * 127.5 + 127.5).type(torch.uint8)
                self.__topdown_info['free_map_cv2'] = cv2.cvtColor(
                    cv2.applyColorMap(
                        topdown_free_map_vis.detach().cpu().numpy(),
                        cv2.COLORMAP_TWILIGHT_SHIFTED),
                    cv2.COLOR_BGR2RGB)
                self.__topdown_info['free_map_binary_cv2'] = cv2.cvtColor(
                    self.__topdown_info['free_map_binary'] * 255,
                    cv2.COLOR_GRAY2RGB)
                
                topdown_visible_map_vis = (topdown_visible_map / topdown_visible_map.abs().max() * 127.5 + 127.5).type(torch.uint8)
                self.__topdown_info['visible_map_cv2'] = cv2.cvtColor(
                    cv2.applyColorMap(
                        topdown_visible_map_vis.detach().cpu().numpy(),
                        cv2.COLORMAP_TWILIGHT_SHIFTED),
                    cv2.COLOR_BGR2RGB)
                self.__topdown_info['visible_map_binary_cv2'] = cv2.cvtColor(
                    self.__topdown_info['visible_map_binary'] * 255,
                    cv2.COLOR_GRAY2RGB)
                
                self.__update_ui_topdown()
                
                if (self.__get_topdown_flag == self.QueryTopdownFlag.RUNNING) or\
                    (self.__get_topdown_flag == self.QueryTopdownFlag.ARRIVED and get_uncertainty_flag):
                    self.__get_topdown_flag = self.QueryTopdownFlag.NONE
                    with self.__get_topdown_condition:
                        self.__get_topdown_condition.notify_all()
                        self.__get_topdown_condition.wait()
                elif self.__get_topdown_flag == self.QueryTopdownFlag.MANUAL:
                    self.__get_topdown_flag = self.QueryTopdownFlag.NONE
                    with self.__get_topdown_condition:
                        self.__get_topdown_condition.notify_all()
                        
            # NOTE: Update precise map
            if self.__query_precise_center is not None:
                precise_voxel_grid, precise_voxel_grid_sdf = self.__mapper.get_precise_voxel_grid_sdf(self.__query_precise_center)
                precise_voxel_grid_topdown:np.ndarray = precise_voxel_grid.select(self.__height_direction[0], 0)
                precise_map = torch.min(precise_voxel_grid_sdf, dim=self.__height_direction[0])[0]
                
                precise_maps = adjust_topdown_maps(
                    [precise_map] + [precise_voxel_grid_topdown[..., _] for _ in range(precise_voxel_grid_topdown.shape[-1])],
                    self.__topdown_info['world_dim_index'],
                    self.__height_direction)
                
                precise_voxel_grid_position_world:torch.Tensor = torch.stack(precise_maps[1:], axis=-1)
                precise_voxel_grid_position_world = precise_voxel_grid_position_world.detach().cpu().numpy()
                precise_voxel_grid_position_topdown = translations_world_to_topdown(
                    precise_voxel_grid_position_world,
                    self.__topdown_info,
                    np.float64)
                precise_voxel_grid_position_topdown = precise_voxel_grid_position_topdown.reshape(precise_voxel_grid_position_world.shape[:2] + (-1,))
                self.__precise_info['free_map_x'] = precise_voxel_grid_position_topdown[..., 0]
                self.__precise_info['free_map_y'] = precise_voxel_grid_position_topdown[..., 1]
                
                precise_map = precise_maps[0]
                
                self.__precise_info['free_map_sdf'] = precise_map.detach().cpu().numpy()
                self.__precise_info['free_map_binary'] = (precise_map > 0).type(torch.uint8).detach().cpu().numpy()
                self.__precise_info['center'] = self.__query_precise_center
                
                precise_map_vis = (precise_map / precise_map.abs().max() * 127.5 + 127.5).type(torch.uint8)
                self.__precise_info['free_map_cv2'] = cv2.cvtColor(
                    cv2.applyColorMap(
                        precise_map_vis.detach().cpu().numpy(),
                        cv2.COLORMAP_TWILIGHT_SHIFTED),
                    cv2.COLOR_BGR2RGB)
                self.__precise_info['free_map_binary_cv2'] = cv2.cvtColor(
                    self.__precise_info['free_map_binary'] * 255,
                    cv2.COLOR_GRAY2RGB)
                
                self.__update_ui_precise()
                
                self.__query_precise_center = None
                with self.__get_precise_condition:
                    self.__get_precise_condition.notify_all()
                    self.__get_precise_condition.wait()
                        
            rospy.logdebug(f'Logical operation used {end_timing(*timing_logical_operations):.2f} ms')
            
            # NOTE: Render RGBD image
            rerender_rgbd_flag = False
            if not self.__hide_windows:
                if (self.__render_box.checked and\
                    frame_id % self.__render_every_slider.int_value == 0 and\
                        frame_last_received is not None and\
                            frame_id > rendering_rgbd_last_frame_id) or\
                                self.__render_rgbd_button_flag:
                    self.__o3d_cache_render_rgbd = o3d.geometry.Image(self.__mapper.render_rgbd(frame_last_received))
                    self.__render_rgbd_button_flag = False
                    rendering_rgbd_last_frame_id = frame_id
                    rerender_rgbd_flag = True
                
            # TODO: Update GUI
            self.__update_ui_mapper(
                frame_current,
                mapper_state,
                construct_mesh_flag,
                rerender_rgbd_flag)
            
            if self.__local_dataset is not None:
                if self.__local_dataset_pose_ros is not None:
                    self.__local_dataset_pose_pub.publish(self.__local_dataset_pose_ros)
        
        self.__mapper.set_mesh_voxels(self.__voxel_size_final)
        mesh_to_save, _, _ = self.__mapper.get_mesh(MeshColorType.QUERY_VERTICES_COLOR, False)
        o3d.io.write_triangle_mesh(
            os.path.join(self.__results_dir, 'final_mesh.ply'),
            mesh_to_save)
        if os.path.exists(self.__dataset_config.scene_mesh_url):
            with open(os.path.join(self.__results_dir, 'gt_mesh.json'), 'w') as f:
                gt_mesh_config = {
                    'mesh_url': self.__dataset_config.scene_mesh_url,
                    'mesh_transform': self.__scene_mesh_transform.tolist()}
                json.dump(gt_mesh_config, f, indent=4)
        try:
            set_planner_state_response:SetPlannerStateResponse = self.__set_planner_state_service(SetPlannerStateRequest(GlobalState.QUIT.value))
        except rospy.ServiceException as e:
            rospy.logerr(f'[IGNORE] Set planner state service call failed: {e}')
        self.__get_topdown_service.shutdown()
        with self.__get_topdown_condition:
            self.__get_topdown_condition.notify_all()
        self.__get_precise_service.shutdown()
        with self.__get_precise_condition:
            self.__get_precise_condition.notify_all()
        if self.__hide_windows:
            rospy.signal_shutdown('Quit')
        else:
            gui.Application.instance.quit()
        
    def __update_ui_mapper(self,
                        frame_current:Union[None, Dict[str, Union[torch.Tensor, int]]],
                        mapper_state:MapperState,
                        construct_mesh_flag:QueryMeshFlag,
                        rerender_rgbd_flag:bool):
        # TODO: Use mapper_state to update the visualization
    
        construct_mesh = None
        uncertainty_mesh = None
        clusters_meshes = None
        targets_frustums = None
        if construct_mesh_flag in [Visualizer.QueryMeshFlag.PLANNING, Visualizer.QueryMeshFlag.VISUALIZATION]:
            construct_mesh = self.__update_mesh('construct_mesh', False if self.__hide_windows else self.__visualize_construct_mesh_box.checked, self.__o3d_materials['unlit_mat'], False)
            if construct_mesh_flag == Visualizer.QueryMeshFlag.PLANNING:
                uncertainty_mesh = self.__update_mesh('uncertainty_mesh', False if self.__hide_windows else self.__visualize_uncertainty_mesh_box.checked, self.__o3d_materials['unlit_mat'], False)
                clusters_meshes, targets_frustums = self.__update_clusters_targets_frustums(False if self.__hide_windows else self.__visualize_clusters_box.checked, False)
        
        if not self.__hide_windows:
            timing_update_render = start_timing()
            gui.Application.instance.post_to_main_thread(
                self.__window,
                lambda: self.__update_main_thread_ui_mapper(
                    construct_mesh,
                    uncertainty_mesh,
                    clusters_meshes,
                    targets_frustums,
                    rerender_rgbd_flag))
            rospy.logdebug(f'Update ui of mapper used {end_timing(*timing_update_render):.2f} ms')
        
        return
    
    def __update_main_thread_ui_mapper(self,
                        construct_mesh:o3d.geometry.TriangleMesh,
                        uncertainty_mesh:o3d.geometry.TriangleMesh,
                        clusters_meshes:o3d.geometry.TriangleMesh,
                        targets_frustums:List[o3d.geometry.LineSet],
                        rerender_rgbd_flag:bool):
            
        if construct_mesh is not None:
            self.__widget_3d.scene.remove_geometry('construct_mesh')
            self.__widget_3d.scene.add_geometry('construct_mesh', construct_mesh, self.__o3d_materials['unlit_mat'])
            self.__widget_3d.scene.show_geometry('construct_mesh', self.__visualize_construct_mesh_box.checked)
            
        if uncertainty_mesh is not None:
            self.__widget_3d.scene.remove_geometry('uncertainty_mesh')
            self.__widget_3d.scene.add_geometry('uncertainty_mesh', uncertainty_mesh, self.__o3d_materials['unlit_mat'])
            self.__widget_3d.scene.show_geometry('uncertainty_mesh', self.__visualize_uncertainty_mesh_box.checked)
            
        if clusters_meshes is not None:
            self.__widget_3d.scene.remove_geometry('clusters_meshes')
            self.__widget_3d.scene.add_geometry('clusters_meshes', clusters_meshes, self.__o3d_materials['unlit_mat'])
            self.__widget_3d.scene.show_geometry('clusters_meshes', self.__visualize_clusters_box.checked)
            
        if targets_frustums is not None:
            cluster_index = 0
            while self.__widget_3d.scene.has_geometry(f'target_frustum_{cluster_index}'):
                self.__widget_3d.scene.remove_geometry(f'target_frustum_{cluster_index}')
                cluster_index += 1
            for cluster_index, target_frustum in enumerate(targets_frustums):
                self.__widget_3d.scene.add_geometry(
                    f'target_frustum_{cluster_index}',
                    target_frustum,
                    self.__o3d_materials[TARGETS_FRUSTUMS['material']])
                self.__widget_3d.scene.show_geometry(
                    f'target_frustum_{cluster_index}',
                    self.__visualize_clusters_box.checked)
            
        if rerender_rgbd_flag:
            self.__rgbd_render_image.update_image(self.__o3d_cache_render_rgbd)
            
        return
        
    def __update_ui_frame(self,
                        frame_current:Union[None, Dict[str, Union[torch.Tensor, int]]]):
        if not self.__update_main_thread.is_alive():
            return
        
        current_frustum = None
        current_agent = None
        rgbd_image = None
        current_pcd = None
        current_horizon = None
        
        if frame_current is not None:
            # TODO: do not add element every time, if the checkbox is false, save them to cache, when the checkbox is true, add them to the scene
            rgb_data:np.ndarray = frame_current['rgb'].detach().cpu().numpy()
            depth_data:np.ndarray = frame_current['depth'].detach().cpu().numpy()
            pose_data:np.ndarray = frame_current['c2w'].detach().cpu().numpy()
            
            self.__topdown_info['rotation_vector'], self.__topdown_info['translation'] = c2w_world_to_topdown(
                pose_data,
                self.__topdown_info,
                self.__height_direction,
                np.float64)
            self.__topdown_info['translation_pixel'] = translations_world_to_topdown(
                pose_data[:3, 3],
                self.__topdown_info,
                np.int32).reshape(-1)
            self.__update_ui_topdown()
            # XXX: test the correctness of the transform
            # topdown_rotation_vector, topdown_translation = c2w_world_to_topdown(pose_data, self.__topdown_info, self.__height_direction, translation_topdown_dtype=np.float64)
            # world_translation = c2w_topdown_to_world(topdown_translation, self.__topdown_info, pose_data[self.__height_direction[0], 3])
            # assert np.allclose(world_translation, pose_data[:3, 3]), f'{world_translation} != {pose_data[:3, 3]}'
        
            pose_data_o3d = OPENCV_TO_OPENGL @ pose_data @ OPENCV_TO_OPENGL
            current_frustum = o3d.geometry.LineSet.create_camera_visualization(
                self.__o3d_const_camera_intrinsics,
                np.linalg.inv(pose_data_o3d),
                CURRENT_FRUSTUM['scale'])
            current_frustum.paint_uniform_color(CURRENT_FRUSTUM['color'])
            current_agent = o3d.geometry.TriangleMesh(self.__agent_cylinder_mesh)
            current_agent.translate(pose_data_o3d[:3, 3])
        
            rgb_vis = np.uint8(rgb_data * 255)
            depth_vis = depth2rgb(depth_data, min_value=self.__rgbd_sensor.depth_min, max_value=self.__rgbd_sensor.depth_max)
            rgbd_vis = np.hstack((rgb_vis, depth_vis))
            
            rgbd_image = o3d.geometry.Image(rgbd_vis)
            
            self.__o3d_pcd['current_pcd'] = rgbd_to_pointcloud(
                rgb_vis,
                depth_data,
                pose_data,
                self.__o3d_const_camera_intrinsics_o3c,
                1000,
                self.__rgbd_sensor.depth_max,
                self.__device_o3c)
            current_pcd:o3d.t.geometry.PointCloud = self.__update_pcd(
                'current_pcd',
                False if self.__hide_windows else self.__current_pcd_box.checked,
                self.__o3d_materials['unlit_mat'],
                False)
            current_pcd_legacy:o3d.geometry.PointCloud = current_pcd.to_legacy()
            current_horizon:o3d.geometry.AxisAlignedBoundingBox = current_pcd_legacy.get_axis_aligned_bounding_box()
            current_horizon.color = CURRENT_HORIZON['color']
            self.__topdown_info['horizon_bbox'] = (
                OPENCV_TO_OPENGL[:3, :3] @ current_horizon.get_min_bound(),
                OPENCV_TO_OPENGL[:3, :3] @ current_horizon.get_max_bound())
            self.__current_horizon = get_horizon_bound_topdown(
                self.__topdown_info['horizon_bbox'][0],
                self.__topdown_info['horizon_bbox'][1],
                self.__topdown_info,
                self.__height_direction)
        
        if not self.__hide_windows:
            timing_update_render = start_timing()
            gui.Application.instance.post_to_main_thread(
                self.__window,
                lambda: self.__update_main_thread_ui_frame(
                    current_frustum,
                    current_agent,
                    rgbd_image,
                    current_pcd,
                    current_horizon))
            rospy.logdebug(f'Update ui of frame used {end_timing(*timing_update_render):.2f} ms')
        
        return
    
    def __update_main_thread_ui_frame(self,
                        current_frustum:o3d.geometry.LineSet,
                        current_agent:o3d.geometry.TriangleMesh,
                        rgbd_image:o3d.geometry.Image,
                        current_pcd:o3d.geometry.PointCloud,
                        current_horizon:o3d.geometry.AxisAlignedBoundingBox):
        if current_frustum is not None:
            self.__widget_3d.scene.remove_geometry('current_frustum')
            self.__widget_3d.scene.add_geometry('current_frustum', current_frustum, self.__o3d_materials[CURRENT_FRUSTUM['material']])
            self.__widget_3d.scene.show_geometry('current_frustum', self.__current_frustum_box.checked)
            
        if current_agent is not None:
            self.__widget_3d.scene.remove_geometry('current_agent')
            self.__widget_3d.scene.add_geometry('current_agent', current_agent, self.__o3d_materials[CURRENT_AGENT['material']])
            self.__widget_3d.scene.show_geometry('current_agent', self.__current_agent_box.checked)
            
        if rgbd_image is not None:
            self.__rgbd_live_image.update_image(rgbd_image)
            
        if current_pcd is not None:
            self.__widget_3d.scene.remove_geometry('current_pcd')
            self.__widget_3d.scene.add_geometry(
                'current_pcd',
                current_pcd,
                self.__o3d_materials['unlit_mat'])
            self.__widget_3d.scene.show_geometry('current_pcd', self.__current_pcd_box.checked)
            
        if current_horizon is not None:
            self.__widget_3d.scene.remove_geometry('current_horizon')
            self.__widget_3d.scene.add_geometry('current_horizon', current_horizon, self.__o3d_materials['unlit_line_mat'])
            self.__widget_3d.scene.show_geometry('current_horizon', self.__current_horizon_box.checked)
            
        return
    
    def __update_dataset(self):
        with self.__local_dataset_condition:
            dataset_config = self.__local_dataset.setup()
            self.__dataset_config:GetDatasetConfigResponse = dataset_config_to_ros(dataset_config)
            rospy.Service('get_dataset_config', GetDatasetConfig, self.__get_dataset_config)
            self.__local_dataset_twist:Twist = None
            rospy.Subscriber('cmd_vel', Twist, self.__cmd_vel_callback, queue_size=1)
            movement_fail_times = 0
            movement_fail_times_pub = rospy.Publisher('movement_fail_times', Int32, queue_size=1)
            self.__local_dataset_state = self.LocalDatasetState.INITIALIZED
            self.__local_dataset_condition.notify_all()
            while self.__global_state != GlobalState.QUIT:
                if self.__local_dataset_state == self.LocalDatasetState.INITIALIZED:
                    self.__local_dataset_state = self.LocalDatasetState.RUNNING
                    self.__local_dataset_condition.notify_all()
                if self.__local_dataset.is_finished():
                    self.__global_state = GlobalState.QUIT
                else:
                    self.__local_dataset_condition.wait()
                if self.__global_state == GlobalState.QUIT:
                    break
                apply_movement_flag = self.__local_dataset_twist is not None
                apply_movement_result = False
                if apply_movement_flag:
                    apply_movement_result = self.__local_dataset.apply_movement(self.__local_dataset_twist)
                    self.__local_dataset_twist = None
                step_times, step_num = self.__local_dataset.get_step_info()
                scene_id = self.__local_dataset.get_scene_id()
                self.__local_dataset_label.text = f'Scene: {scene_id}\nStep: {step_times}/{step_num}'
                frame_numpy = self.__local_dataset.get_frame()
                frame_c2w = frame_numpy['c2w']
                pose_change_type = self.__is_pose_changed(frame_c2w)
                if pose_change_type != PoseChangeType.NONE:
                    if pose_change_type in [PoseChangeType.TRANSLATION, PoseChangeType.BOTH]:
                        movement_fail_times = 0
                    frame_quaternion = quaternion.from_rotation_matrix(frame_c2w[:3, :3])
                    frame_quaternion = quaternion.as_float_array(frame_quaternion)
                    pose_ros = PoseStamped()
                    pose_ros.header.stamp = rospy.Time.now()
                    pose_ros.header.frame_id = 'world'
                    pose_ros.pose.position.x = frame_c2w[0, 3]
                    pose_ros.pose.position.y = frame_c2w[1, 3]
                    pose_ros.pose.position.z = frame_c2w[2, 3]
                    pose_ros.pose.orientation.w = frame_quaternion[0]
                    pose_ros.pose.orientation.x = frame_quaternion[1]
                    pose_ros.pose.orientation.y = frame_quaternion[2]
                    pose_ros.pose.orientation.z = frame_quaternion[3]
                    self.__frame_c2w_last = frame_c2w
                    frame_torch = {
                        'rgb': torch.from_numpy(frame_numpy['rgb']),
                        'depth': torch.from_numpy(frame_numpy['depth']),
                        'c2w': torch.from_numpy(frame_c2w)}
                    self.__update_ui_frame(frame_torch)
                    if self.__frames_cache.empty():
                        self.__frames_cache.put(frame_torch)
                    self.__local_dataset_pose_ros = pose_ros
                    self.__local_dataset_pose_pub.publish(self.__local_dataset_pose_ros)
                    movement_fail_times_pub.publish(Int32(movement_fail_times))
                elif apply_movement_flag:
                    if apply_movement_result:
                        movement_fail_times += 1
                    movement_fail_times_pub.publish(Int32(movement_fail_times))
                self.__local_dataset_condition.notify_all()
            self.__local_dataset.close()
        
    def __update_ui_topdown(self):
        topdown_free_map_o3d = None
        if self.__topdown_info['free_map_cv2'] is not None:
            topdown_free_map = self.__topdown_info['free_map_cv2'].copy()
            if self.__topdown_info['rotation_vector'] is not None and self.__topdown_info['translation_pixel'] is not None:
                topdown_free_map = visualize_agent(
                    topdown_map=topdown_free_map,
                    meter_per_pixel=self.__topdown_info['meter_per_pixel'],
                    agent_translation=self.__topdown_info['translation_pixel'],
                    agent_rotation_vector=self.__topdown_info['rotation_vector'],
                    agent_color=(128, 255, 128),
                    agent_radius=self.__dataset_config.agent_radius,
                    rotation_vector_color=(0, 255, 0),
                    rotation_vector_thickness=2,
                    rotation_vector_length=20)
            topdown_free_map_o3d = o3d.geometry.Image(topdown_free_map)
            
        topdown_free_map_binary_o3d = None
        if self.__topdown_info['free_map_binary_cv2'] is not None:
            topdown_free_map_binary = self.__topdown_info['free_map_binary_cv2'].copy()
            if self.__topdown_info['rotation_vector'] is not None and self.__topdown_info['translation_pixel'] is not None:
                topdown_free_map_binary = visualize_agent(
                    topdown_map=topdown_free_map_binary,
                    meter_per_pixel=self.__topdown_info['meter_per_pixel'],
                    agent_translation=self.__topdown_info['translation_pixel'],
                    agent_rotation_vector=self.__topdown_info['rotation_vector'],
                    agent_color=(128, 255, 128),
                    agent_radius=self.__dataset_config.agent_radius,
                    rotation_vector_color=(0, 255, 0),
                    rotation_vector_thickness=2,
                    rotation_vector_length=20)
            topdown_free_map_binary_o3d = o3d.geometry.Image(topdown_free_map_binary)
            
        topdown_visible_map_o3d = None
        if self.__topdown_info['visible_map_cv2'] is not None:
            topdown_visible_map = self.__topdown_info['visible_map_cv2'].copy()
            if self.__topdown_info['rotation_vector'] is not None and self.__topdown_info['translation_pixel'] is not None:
                topdown_visible_map = visualize_agent(
                    topdown_map=topdown_visible_map,
                    meter_per_pixel=self.__topdown_info['meter_per_pixel'],
                    agent_translation=self.__topdown_info['translation_pixel'],
                    agent_rotation_vector=self.__topdown_info['rotation_vector'],
                    agent_color=(128, 255, 128),
                    agent_radius=self.__dataset_config.agent_radius,
                    rotation_vector_color=(0, 255, 0),
                    rotation_vector_thickness=2,
                    rotation_vector_length=20)
            if self.__current_horizon is not None:
                cv2.rectangle(
                    topdown_visible_map,
                    np.int32(self.__current_horizon[0]),
                    np.int32(self.__current_horizon[1]),
                    (255, 0, 0),
                    1)
            topdown_visible_map_o3d = o3d.geometry.Image(topdown_visible_map)
            
        topdown_visible_map_binary_o3d = None
        if self.__topdown_info['visible_map_binary_cv2'] is not None:
            topdown_visible_map_binary = self.__topdown_info['visible_map_binary_cv2'].copy()
            if self.__topdown_info['rotation_vector'] is not None and self.__topdown_info['translation_pixel'] is not None:
                topdown_visible_map_binary = visualize_agent(
                    topdown_map=topdown_visible_map_binary,
                    meter_per_pixel=self.__topdown_info['meter_per_pixel'],
                    agent_translation=self.__topdown_info['translation_pixel'],
                    agent_rotation_vector=self.__topdown_info['rotation_vector'],
                    agent_color=(128, 255, 128),
                    agent_radius=self.__dataset_config.agent_radius,
                    rotation_vector_color=(0, 255, 0),
                    rotation_vector_thickness=2,
                    rotation_vector_length=20)
            topdown_visible_map_binary_o3d = o3d.geometry.Image(topdown_visible_map_binary)
            
        precise_topdown_map_o3d = None
        if self.__precise_topdown_map_cv2 is not None:
            precise_topdown_map_cv2 = self.__precise_topdown_map_cv2.copy()
            if self.__topdown_info['rotation_vector'] is not None and self.__topdown_info['translation_pixel'] is not None:
                precise_topdown_map_cv2 = visualize_agent(
                    topdown_map=precise_topdown_map_cv2,
                    meter_per_pixel=self.__topdown_info['meter_per_pixel'],
                    agent_translation=self.__topdown_info['translation_pixel'],
                    agent_rotation_vector=self.__topdown_info['rotation_vector'],
                    agent_color=(128, 255, 128),
                    agent_radius=self.__dataset_config.agent_radius,
                    rotation_vector_color=(0, 255, 0),
                    rotation_vector_thickness=2,
                    rotation_vector_length=20,
                    resize_scale=self.__topdown_map_resize_scale)
            precise_topdown_map_o3d = o3d.geometry.Image(precise_topdown_map_cv2)
            
        if not self.__hide_windows:
            timing_update_render = start_timing()
            gui.Application.instance.post_to_main_thread(
                self.__window,
                lambda: self.__update_main_thread_ui_topdown(
                    topdown_free_map_o3d,
                    topdown_free_map_binary_o3d,
                    topdown_visible_map_o3d,
                    topdown_visible_map_binary_o3d,
                    precise_topdown_map_o3d))
            rospy.logdebug(f'Update ui of topdown used {end_timing(*timing_update_render):.2f} ms')
        
    def __update_main_thread_ui_topdown(
        self,
        topdown_free_map_o3d:o3d.geometry.Image,
        topdown_free_map_binary_o3d:o3d.geometry.Image,
        topdown_visible_map_o3d:o3d.geometry.Image,
        topdown_visible_map_binary_o3d:o3d.geometry.Image,
        precise_topdown_map_o3d:o3d.geometry.Image):
        if topdown_free_map_o3d is not None:
            self.__topdown_free_map_image.update_image(topdown_free_map_o3d)
            
        if topdown_free_map_binary_o3d is not None:
            self.__topdown_free_map_image_binary.update_image(topdown_free_map_binary_o3d)
            
        if topdown_visible_map_o3d is not None:
            self.__topdown_visible_map_image.update_image(topdown_visible_map_o3d)
            
        if topdown_visible_map_binary_o3d is not None:
            self.__topdown_visible_map_image_binary.update_image(topdown_visible_map_binary_o3d)
            
        if precise_topdown_map_o3d is not None:
            self.__precise_topdown_map_image.update_image(precise_topdown_map_o3d)
        
    def __update_ui_precise(self):
        precise_free_map_o3d = None
        if self.__precise_info['free_map_cv2'] is not None:
            precise_free_map_o3d = o3d.geometry.Image(self.__precise_info['free_map_cv2'])
            
        precise_topdown_map_o3d = None
        if self.__precise_info['free_map_sdf'] is not None and\
            self.__precise_info['free_map_x'] is not None and\
                self.__precise_info['free_map_y'] is not None and\
                    self.__topdown_info['free_map_sdf'] is not None:
            precise_map = self.__precise_info['free_map_sdf'].copy()
            precise_map_x = self.__precise_info['free_map_x'].copy()
            precise_map_y = self.__precise_info['free_map_y'].copy()
            precise_map_center_world = self.__precise_info['center']
            precise_map_center_topdown = translations_world_to_topdown(
                precise_map_center_world,
                self.__topdown_info,
                np.float64).reshape(-1)
            topdown_map = self.__topdown_info['free_map_sdf'].copy()
            map_sdf_max = max(np.abs(precise_map).max(), np.abs(topdown_map).max())
            precise_map_vis = np.uint8(precise_map / map_sdf_max * 127.5 + 127.5)
            topdown_map_vis = np.uint8(topdown_map / map_sdf_max * 127.5 + 127.5)
            precise_topdown_map_vis, precise_map_rect = visualize_topdown_and_precise_map(
                topdown_map_vis,
                self.__topdown_map_resize_scale,
                precise_map_vis,
                precise_map_x,
                precise_map_y)
            precise_topdown_map_cv2 = cv2.cvtColor(
                cv2.applyColorMap(
                    precise_topdown_map_vis,
                    cv2.COLORMAP_TWILIGHT_SHIFTED),
                cv2.COLOR_BGR2RGB)
            rect_thickness = 3
            rect_edge_expand = np.int32(np.ceil(rect_thickness / 2))
            cv2.rectangle(
                precise_topdown_map_cv2,
                precise_map_rect[0] - rect_edge_expand,
                precise_map_rect[1] + rect_edge_expand,
                (0, 255, 0),
                rect_thickness)
            cv2.rectangle(
                precise_topdown_map_cv2,
                np.int32((precise_map_center_topdown - self.__precise_info['core_size'] / 2 / self.__topdown_info['meter_per_pixel']) * self.__topdown_map_resize_scale) - rect_edge_expand,
                np.int32((precise_map_center_topdown + self.__precise_info['core_size'] / 2 / self.__topdown_info['meter_per_pixel']) * self.__topdown_map_resize_scale) + rect_edge_expand,
                (0, 128, 0),
                rect_thickness)
            
            self.__precise_topdown_map_cv2 = precise_topdown_map_cv2.copy()
            
            precise_topdown_map_cv2 = visualize_agent(
                topdown_map=precise_topdown_map_cv2,
                meter_per_pixel=self.__topdown_info['meter_per_pixel'],
                agent_translation=self.__topdown_info['translation_pixel'],
                agent_rotation_vector=self.__topdown_info['rotation_vector'],
                agent_color=(128, 255, 128),
                agent_radius=self.__dataset_config.agent_radius,
                rotation_vector_color=(0, 255, 0),
                rotation_vector_thickness=2,
                rotation_vector_length=20,
                resize_scale=self.__topdown_map_resize_scale)
                
            precise_topdown_map_o3d = o3d.geometry.Image(precise_topdown_map_cv2)
            
        if not self.__hide_windows:
            timing_update_render = start_timing()
            gui.Application.instance.post_to_main_thread(
                self.__window,
                lambda: self.__update_main_thread_ui_precise(
                    precise_free_map_o3d,
                    precise_topdown_map_o3d))
            rospy.logdebug(f'Update ui of precise used {end_timing(*timing_update_render):.2f} ms')
        
    def __update_main_thread_ui_precise(
        self,
        precise_free_map_o3d:o3d.geometry.Image,
        precise_topdown_map_o3d:o3d.geometry.Image):
        if precise_free_map_o3d is not None:
            self.__precise_free_map_image.update_image(precise_free_map_o3d)
            
        if precise_topdown_map_o3d is not None:
            self.__precise_topdown_map_image.update_image(precise_topdown_map_o3d)
        
    # NOTE: callback functions for GUI
        
    def __window_on_layout(self, ctx:gui.LayoutContext):
        em = ctx.theme.font_size

        panel_width = 23 * em
        rect:gui.Rect = self.__window.content_rect

        self.__panel_control.frame = gui.Rect(rect.x, rect.y, panel_width, rect.height)
        x = self.__panel_control.frame.get_right()

        self.__widget_3d.frame = gui.Rect(x, rect.y, rect.get_right() - 2*panel_width, rect.height)
        self.__panel_visualize.frame = gui.Rect(self.__widget_3d.frame.get_right(), rect.y, panel_width, rect.height)

        panel_timing_width = 15 * em
        panel_timing_height = 3 * em
        self.__panel_timing.frame = gui.Rect(
            rect.get_right() - panel_timing_width,
            rect.y,
            panel_timing_width,
            panel_timing_height
        )

        panel_warning_width = 16 * em
        panel_warning_height = 4 * em
        self.__panel_warning.frame = gui.Rect(
            rect.get_right() // 2 - panel_warning_width + panel_width // 2,
            rect.get_bottom() // 2 - panel_warning_height,
            panel_warning_width,
            panel_warning_height
        )
        return
        
    def __window_on_close(self) -> bool:
        self.__global_state = GlobalState.QUIT
        if self.__local_dataset is not None:
            with self.__local_dataset_condition:
                self.__local_dataset_condition.notify_all()
            self.__local_dataset_thread.join()
        self.__update_main_thread.join()
        gui.Application.instance.quit()
        return True
        
    def __widget_3d_on_key(self, event:gui.KeyEvent):
        if self.__global_state in [GlobalState.MANUAL_CONTROL, GlobalState.MANUAL_PLANNING] and event.type == gui.KeyEvent.Type.DOWN:
            if event.key == gui.KeyName.UP:
                twist_current = {
                    'linear': np.array([0.1, 0.0, 0.0]),
                    'angular': np.zeros(3)
                }
            elif event.key == gui.KeyName.LEFT:
                twist_current = {
                    'linear': np.zeros(3),
                    'angular': np.array([0.0, 0.0, 0.1])
                }
            elif event.key == gui.KeyName.RIGHT:
                twist_current = {
                    'linear': np.zeros(3),
                    'angular': np.array([0.0, 0.0, -0.1])
                }
            elif event.key == gui.KeyName.PAGE_UP:
                twist_current = {
                    'linear': np.zeros(3),
                    'angular': np.array([0.0, -0.1, 0.0])
                }
            elif event.key == gui.KeyName.PAGE_DOWN:
                twist_current = {
                    'linear': np.zeros(3),
                    'angular': np.array([0.0, 0.1, 0.0])
                }
            else:
                return gui.Widget.IGNORED
            self.__apply_movement(twist_current)
            return gui.Widget.HANDLED
        return gui.Widget.IGNORED
    
    def __update_mesh(self, mesh_name:str, show:bool, material:Union[str, o3d.visualization.rendering.MaterialRecord]=None, update:bool=True) -> o3d.geometry.TriangleMesh:
        if isinstance(material, str):
            material = self.__o3d_materials[material]
        elif isinstance(material, o3d.visualization.rendering.MaterialRecord):
            pass
        else:
            material = self.__o3d_materials['lit_mat']
        mesh, vertices_condition = self.__cut_mesh_by_height(o3d.geometry.TriangleMesh(self.__o3d_meshes[mesh_name]))
        # mesh = o3d.geometry.TriangleMesh(self.__o3d_meshes[mesh_name])
        # vertices = np.array(mesh.vertices)
        # vertices_condition = np.logical_or(
        #     vertices[:, self.__height_direction[0]] < (self.__open3d_gui_widget_last_state['height_direction_bound_slider'][0] if self.__hide_windows else self.__height_direction_lower_bound_slider.double_value),
        #     vertices[:, self.__height_direction[0]] > (self.__open3d_gui_widget_last_state['height_direction_bound_slider'][1] if self.__hide_windows else self.__height_direction_upper_bound_slider.double_value))
        # mesh.remove_vertices_by_mask(vertices_condition)
        if update:
            self.__widget_3d.scene.remove_geometry(mesh_name)
            self.__widget_3d.scene.add_geometry(mesh_name, mesh, material)
            self.__widget_3d.scene.show_geometry(mesh_name, show)
        return mesh
    
    def __update_pcd(self, pointcloud_name:str, show:bool, material:Union[str, o3d.visualization.rendering.MaterialRecord]=None, update:bool=True) -> o3d.t.geometry.PointCloud:
        if isinstance(material, str):
            material = self.__o3d_materials[material]
        elif isinstance(material, o3d.visualization.rendering.MaterialRecord):
            pass
        else:
            material = self.__o3d_materials['unlit_mat']
        pcd = o3d.t.geometry.PointCloud(self.__o3d_pcd[pointcloud_name])
        points = pcd.point.positions.numpy()
        points_condition = np.logical_or(
            points[:, self.__height_direction[0]] < (self.__open3d_gui_widget_last_state['height_direction_bound_slider'][0] if self.__hide_windows else self.__height_direction_lower_bound_slider.double_value),
            points[:, self.__height_direction[0]] > (self.__open3d_gui_widget_last_state['height_direction_bound_slider'][1] if self.__hide_windows else self.__height_direction_upper_bound_slider.double_value))
        pcd = pcd.select_by_index(np.where(~points_condition)[0])
        if update:
            if self.__widget_3d.scene.has_geometry(pointcloud_name):
                self.__widget_3d.scene.scene.update_geometry(pointcloud_name,
                                                            pcd,
                                                            rendering.Scene.UPDATE_POINTS_FLAG +\
                                                                rendering.Scene.UPDATE_COLORS_FLAG +\
                                                                    rendering.Scene.UPDATE_NORMALS_FLAG +\
                                                                        rendering.Scene.UPDATE_UV0_FLAG)
            else:
                self.__widget_3d.scene.add_geometry(pointcloud_name, pcd, material)
            self.__widget_3d.scene.show_geometry(pointcloud_name, show)
        return pcd
    
    def __update_clusters_targets_frustums(self, show:bool, update:bool=True) -> Tuple[o3d.geometry.TriangleMesh, List[o3d.geometry.LineSet]]:
        clusters_meshes_o3d = None
        targets_frustums_o3d = []
        if (self.__clusters_sort_type == self.ClustersSortType.AREA) and\
            (self.__clusters_info['clusters_area'] is not None):
            clusters_sort_values = self.__clusters_info['clusters_area']
        elif (self.__clusters_sort_type == self.ClustersSortType.TRIANGLES_N) and\
            (self.__clusters_info['clusters_n_triangles'] is not None):
            clusters_sort_values = self.__clusters_info['clusters_n_triangles']
        elif (self.__clusters_sort_type == self.ClustersSortType.UNCERTAINTY_MAX) and\
            (self.__clusters_info['clusters_uncertainty_max'] is not None):
            clusters_sort_values = self.__clusters_info['clusters_uncertainty_max']
        elif (self.__clusters_sort_type == self.ClustersSortType.UNCERTAINTY_MEAN) and\
            (self.__clusters_info['clusters_uncertainty_mean'] is not None):
            clusters_sort_values = self.__clusters_info['clusters_uncertainty_mean']
        else:
            return clusters_meshes_o3d, targets_frustums_o3d
        
        assert len(clusters_sort_values) ==\
            len(self.__clusters_info['targets_frustums_transform']) ==\
                len(self.__clusters_info['clusters_meshes']), 'Clusters sort values and clusters info are not consistent'
        
        clusters_number = len(clusters_sort_values)
        clusters_index_sorted = np.argsort(clusters_sort_values)
        clusters_targets_frustums_colormap = cm.get_cmap(TARGETS_FRUSTUMS['colormap'])
        if self.__clusters_info['targets_frustums_transform'] is not None:
            cluster_count = 0
            if update:
                while self.__widget_3d.scene.has_geometry(f'target_frustum_{cluster_count}'):
                    self.__widget_3d.scene.remove_geometry(f'target_frustum_{cluster_count}')
                    cluster_count += 1
            for cluster_count, cluster_index in enumerate(clusters_index_sorted):
                target_frustum_transform = self.__clusters_info['targets_frustums_transform'][cluster_index]
                target_frustum_transform_o3d = OPENCV_TO_OPENGL @ target_frustum_transform @ OPENCV_TO_OPENGL
                target_frustum_o3d = o3d.geometry.LineSet.create_camera_visualization(
                    self.__o3d_const_camera_intrinsics,
                    np.linalg.inv(target_frustum_transform_o3d),
                    TARGETS_FRUSTUMS['scale'])
                target_frustum_o3d.paint_uniform_color(
                    clusters_targets_frustums_colormap(
                        cluster_count / clusters_number)[:3])
                targets_frustums_o3d.append(target_frustum_o3d)
                if update:
                    self.__widget_3d.scene.add_geometry(
                        f'target_frustum_{cluster_count}',
                        target_frustum_o3d,
                        self.__o3d_materials[TARGETS_FRUSTUMS['material']])
                    self.__widget_3d.scene.show_geometry(
                        f'target_frustum_{cluster_count}',
                        show)
        if self.__clusters_info['clusters_meshes'] is not None:
            clusters_meshes_o3d = o3d.geometry.TriangleMesh()
            for cluster_count, cluster_index in enumerate(clusters_index_sorted):
                cluster_mesh = o3d.geometry.TriangleMesh(self.__clusters_info['clusters_meshes'][cluster_index])
                cluster_mesh.paint_uniform_color(
                    clusters_targets_frustums_colormap(
                        cluster_count / clusters_number)[:3])
                clusters_meshes_o3d += cluster_mesh
            clusters_meshes_o3d.remove_unreferenced_vertices()
            clusters_meshes_o3d.compute_vertex_normals()
            self.__o3d_meshes['clusters_meshes'] = o3d.geometry.TriangleMesh(clusters_meshes_o3d)
            clusters_meshes_o3d = self.__update_mesh(
                'clusters_meshes',
                False if self.__hide_windows else self.__visualize_clusters_box.checked,
                self.__o3d_materials['unlit_mat'],
                update)
        return clusters_meshes_o3d, targets_frustums_o3d
    
    def __height_direction_bound_slider_callback(self, value:float, is_upper:int):
        if value == self.__open3d_gui_widget_last_state['height_direction_bound_slider'][is_upper]:
            return gui.Slider.HANDLED
        else:
            if self.__scene_mesh_box is not None:
                self.__update_mesh('scene_mesh', self.__scene_mesh_box.checked, self.__o3d_materials['lit_mat'])
            if self.__o3d_meshes['construct_mesh'] is not None:
                self.__update_mesh('construct_mesh', self.__visualize_construct_mesh_box.checked, self.__o3d_materials['unlit_mat'])
            if self.__o3d_meshes['uncertainty_mesh'] is not None:
                self.__update_mesh('uncertainty_mesh', self.__visualize_uncertainty_mesh_box.checked, self.__o3d_materials['unlit_mat'])
            if self.__o3d_meshes['clusters_meshes'] is not None:
                self.__update_mesh('clusters_meshes', self.__visualize_clusters_box.checked, self.__o3d_materials['unlit_mat'])
            if self.__o3d_pcd['current_pcd'] is not None:
                current_pcd:o3d.t.geometry.PointCloud = self.__update_pcd(
                    'current_pcd',
                    self.__current_pcd_box.checked,
                    self.__o3d_materials['unlit_mat'])
                current_pcd_legacy:o3d.geometry.PointCloud = current_pcd.to_legacy()
                current_horizon:o3d.geometry.AxisAlignedBoundingBox = current_pcd_legacy.get_axis_aligned_bounding_box()
                current_horizon.color = CURRENT_HORIZON['color']
                self.__widget_3d.scene.remove_geometry('current_horizon')
                self.__widget_3d.scene.add_geometry('current_horizon', current_horizon, self.__o3d_materials['unlit_line_mat'])
                self.__widget_3d.scene.show_geometry('current_horizon', self.__current_horizon_box.checked)
            # TODO: Crop other elements here
            self.__open3d_gui_widget_last_state['height_direction_bound_slider'][is_upper] = value
        if is_upper:
            self.__height_direction_lower_bound_slider.set_limits(self.__bbox_visualize[self.__height_direction[0]][0], value)
        else:
            self.__height_direction_upper_bound_slider.set_limits(value, self.__bbox_visualize[self.__height_direction[0]][1])
        return gui.Slider.HANDLED
            
    def __global_state_callback(self, global_state_str:str, global_state_index:int):
        global_state = GlobalState(global_state_str)
        set_planner_state_response:SetPlannerStateResponse = self.__set_planner_state_service(SetPlannerStateRequest(global_state_str))
        if global_state == self.__global_state:
            return gui.Combobox.HANDLED
        elif self.__global_state == GlobalState.REPLAY and not self.__hide_windows:
            self.__global_states_selectable.remove(GlobalState.REPLAY)
            self.__global_state_combobox.remove_item(GlobalState.REPLAY.value)
        self.__global_state = global_state
        return gui.Combobox.HANDLED
    
    def __clusters_sort_by_combobox_callback(self, clusters_sort_type_str:str, clusters_sort_type_index:int):
        self.__clusters_sort_type = self.ClustersSortType(clusters_sort_type_str)
        self.__update_clusters_targets_frustums(self.__visualize_clusters_box.checked)
        return gui.Combobox.HANDLED
    
    # NOTE: ros functions
    
    def __frame_callback(self, msg:frame):
        frame_quaternion = np.array([msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z])
        frame_quaternion = quaternion.from_float_array(frame_quaternion)
        frame_translation = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        frame_c2w = np.eye(4)
        frame_c2w[:3, :3] = quaternion.as_rotation_matrix(frame_quaternion)
        frame_c2w[:3, 3] = frame_translation
        frame_c2w = convert_to_c2w_opencv(frame_c2w, self.__pose_data_type)
        if self.__is_pose_changed(frame_c2w) == PoseChangeType.NONE:
            return
        
        frame_rotation_vector = np.degrees(quaternion.as_rotation_vector(frame_quaternion))
        rospy.loginfo(f'Agent:\n\tX: {frame_translation[0]:.2f}, Y: {frame_translation[1]:.2f}, Z: {frame_translation[2]:.2f}\n\tX_angle: {frame_rotation_vector[0]:.2f}, Y_angle: {frame_rotation_vector[1]:.2f}, Z_angle: {frame_rotation_vector[2]:.2f}')
        
        if msg.rgb.encoding in ['rgb8', 'bgr8', 'rgba8', 'bgra8']:
            frame_rgb = np.frombuffer(msg.rgb.data, dtype=np.uint8).reshape(msg.rgb.height, msg.rgb.width, 3)
            if msg.rgb.encoding == 'bgr8':
                frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)
            elif msg.rgb.encoding == 'rgba8':
                frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGBA2RGB)
            elif msg.rgb.encoding == 'bgra8':
                frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGRA2RGB)
            frame_rgb = frame_rgb.astype(np.float32) / 255.0
        else:
            raise NotImplementedError(f'Unsupported RGB encoding: {msg.rgb.encoding}')
        if msg.depth.encoding == '32FC1':
            frame_depth = np.frombuffer(msg.depth.data, dtype=np.float32).reshape(msg.depth.height, msg.depth.width)
            assert self.__rgbd_sensor.depth_scale == 1, 'Depth scale is not 1'
        elif msg.depth.encoding == '16UC1':
            frame_depth = np.frombuffer(msg.depth.data, dtype=np.uint16).reshape(msg.depth.height, msg.depth.width).astype(np.float32)
            assert self.__rgbd_sensor.depth_scale == 1000, 'Depth scale is not 1000'
        else:
            raise NotImplementedError(f'Unsupported Depth encoding: {msg.depth.encoding}')
        frame_depth = frame_depth / self.__rgbd_sensor.depth_scale
        if np.any(np.isnan(frame_depth)):
            rospy.logwarn('Depth contains NaN')
            return
        if self.__frames_cache.empty():
            self.__frame_c2w_last = frame_c2w
            frame_current = {
                'rgb': torch.from_numpy(frame_rgb),
                'depth': torch.from_numpy(frame_depth.copy()),
                'c2w': torch.from_numpy(frame_c2w).float()}
            self.__update_ui_frame(frame_current)
            self.__frames_cache.put(frame_current)
        return
        
    def __apply_movement(self, twist:Dict[str, np.ndarray]):
        if self.__local_dataset is None:
            twist_msg = Twist()
            twist_msg.linear.x = twist['linear'][0]
            twist_msg.linear.y = twist['linear'][1]
            twist_msg.linear.z = twist['linear'][2]
            twist_msg.angular.x = twist['angular'][0]
            twist_msg.angular.y = twist['angular'][1]
            twist_msg.angular.z = twist['angular'][2]
            self.__cmd_vel_publisher.publish(twist_msg)
        else:
            if not self.__local_dataset_parallelized and not self.__frames_cache.empty():
                return
            with self.__local_dataset_condition:
                self.__local_dataset_twist = twist
                self.__local_dataset_condition.notify_all()
                self.__local_dataset_condition.wait()
        return
        
    def __cmd_vel_callback(self, twist:Twist):
        twist_current = {
            'linear': np.array([
                twist.linear.x,
                twist.linear.y,
                twist.linear.z]),
            'angular': np.array([
                twist.angular.x,
                twist.angular.y,
                twist.angular.z])}
        self.__apply_movement(twist_current)
        
    def __get_dataset_config(self, req:GetDatasetConfigRequest) -> GetDatasetConfigResponse:
        return self.__dataset_config
    
    def __get_topdown(self, req:GetTopdownRequest) -> GetTopdownResponse:
        with self.__get_topdown_condition:
            if req.arrived_flag:
                self.__get_topdown_flag = self.QueryTopdownFlag.ARRIVED
            elif self.__get_topdown_flag == self.QueryTopdownFlag.NONE:
                self.__get_topdown_flag = self.QueryTopdownFlag.RUNNING
            self.__get_topdown_condition.wait()
            if self.__global_state == GlobalState.QUIT:
                self.__get_topdown_condition.notify_all()
                return None
            free_map_binary:np.ndarray = self.__topdown_info['free_map_binary'].copy()
            visible_map_binary:np.ndarray = self.__topdown_info['visible_map_binary'].copy()
            if req.arrived_flag:
                targets_frustums_transform:List[np.ndarray] = self.__clusters_info['targets_frustums_transform'].copy()
                clusters_area:np.ndarray = self.__clusters_info['clusters_area'].copy()
                clusters_n_triangles:np.ndarray = self.__clusters_info['clusters_n_triangles'].copy()
                clusters_uncertainty_max:np.ndarray = self.__clusters_info['clusters_uncertainty_max'].copy()
                clusters_uncertainty_mean:np.ndarray = self.__clusters_info['clusters_uncertainty_mean'].copy()
            self.__get_topdown_condition.notify_all()
        topdown_response = GetTopdownResponse()
        topdown_response.free_map = free_map_binary.flatten().tolist()
        topdown_response.visible_map = visible_map_binary.flatten().tolist()
        if req.arrived_flag:
            topdown_response.clusters_area = clusters_area.tolist()
            topdown_response.clusters_n_triangles = clusters_n_triangles.tolist()
            topdown_response.clusters_uncertainty_max = clusters_uncertainty_max.tolist()
            topdown_response.clusters_uncertainty_mean = clusters_uncertainty_mean.tolist()
            topdown_response.targets_frustums = []
            for target_frustum_transform in targets_frustums_transform:
                target_frustum = Pose()
                target_frustum_quaternion = quaternion.from_rotation_matrix(target_frustum_transform[:3, :3])
                target_frustum_quaternion = quaternion.as_float_array(target_frustum_quaternion)
                target_frustum.position.x = target_frustum_transform[0, 3]
                target_frustum.position.y = target_frustum_transform[1, 3]
                target_frustum.position.z = target_frustum_transform[2, 3]
                target_frustum.orientation.w = target_frustum_quaternion[0]
                target_frustum.orientation.x = target_frustum_quaternion[1]
                target_frustum.orientation.y = target_frustum_quaternion[2]
                target_frustum.orientation.z = target_frustum_quaternion[3]
                topdown_response.targets_frustums.append(target_frustum)
        topdown_response.horizon_bound_min.x = self.__topdown_info['horizon_bbox'][0][0]
        topdown_response.horizon_bound_min.y = self.__topdown_info['horizon_bbox'][0][1]
        topdown_response.horizon_bound_min.z = self.__topdown_info['horizon_bbox'][0][2]
        topdown_response.horizon_bound_max.x = self.__topdown_info['horizon_bbox'][1][0]
        topdown_response.horizon_bound_max.y = self.__topdown_info['horizon_bbox'][1][1]
        topdown_response.horizon_bound_max.z = self.__topdown_info['horizon_bbox'][1][2]
        return topdown_response
    
    def __get_precise(self, req:GetPreciseRequest) -> GetPreciseResponse:
        with self.__get_precise_condition:
            self.__query_precise_center = c2w_topdown_to_world(
                (req.center_x, req.center_y),
                self.__topdown_info,
                0)
            self.__get_precise_condition.wait()
            if self.__global_state == GlobalState.QUIT:
                self.__get_precise_condition.notify_all()
                return None
            free_map_x:np.ndarray = self.__precise_info['free_map_x'].copy()
            free_map_y:np.ndarray = self.__precise_info['free_map_y'].copy()
            free_map_binary:np.ndarray = self.__precise_info['free_map_binary'].copy()
            self.__get_precise_condition.notify_all()
        precise_response = GetPreciseResponse()
        precise_response.precise_map = free_map_binary.flatten().tolist()
        precise_response.precise_map_x = free_map_x.flatten().tolist()
        precise_response.precise_map_y = free_map_y.flatten().tolist()
        return precise_response
    
    def __get_topdown_config(self, req:GetTopdownConfigRequest) -> GetTopdownConfigResponse:
        topdown_config_response = GetTopdownConfigResponse()
        topdown_config_response.topdown_x_world_dim_index = self.__topdown_info['world_dim_index'][0]
        topdown_config_response.topdown_y_world_dim_index = self.__topdown_info['world_dim_index'][1]
        topdown_config_response.topdown_x_world_lower_bound = self.__topdown_info['world_2d_bbox'][0][0]
        topdown_config_response.topdown_x_world_upper_bound = self.__topdown_info['world_2d_bbox'][0][1]
        topdown_config_response.topdown_y_world_lower_bound = self.__topdown_info['world_2d_bbox'][1][0]
        topdown_config_response.topdown_y_world_upper_bound = self.__topdown_info['world_2d_bbox'][1][1]
        topdown_config_response.topdown_x_length = self.__topdown_info['grid_map_shape'][0]
        topdown_config_response.topdown_y_length = self.__topdown_info['grid_map_shape'][1]
        topdown_config_response.topdown_meter_per_pixel = self.__topdown_info['meter_per_pixel']
        assert self.__precise_info['grid_map_shape'][0] == self.__precise_info['grid_map_shape'][1], 'Precise map is not square'
        topdown_config_response.precise_size = self.__precise_info['grid_map_shape'][0]
        topdown_config_response.precise_core_size = self.__precise_info['core_size'] / self.__topdown_info['meter_per_pixel']
        topdown_config_response.precise_meter_per_pixel = self.__precise_info['meter_per_pixel']
        return topdown_config_response
    
    def __set_mapper(self, req:SetMapperRequest) -> SetMapperResponse:
        kf_every = req.kf_every
        kf_every_old = self.__mapper.get_kf_every()
        self.__mapper.set_kf_every(kf_every)
        if not self.__hide_windows:
            self.__kf_every_slider.int_value = kf_every
        return SetMapperResponse(kf_every_old)
    
    def __set_kf_trigger_poses(self, req:SetKFTriggerPosesRequest) -> SetKFTriggerPosesResponse:
        self.__kf_trigger = [
            np.array([req.x, req.y]),
            np.array(req.theta)]
        return SetKFTriggerPosesResponse()
        
    # NOTE: Common Funtions
    
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
            
    def __cut_mesh_by_height(self, mesh:o3d.geometry.TriangleMesh) -> Tuple[o3d.geometry.TriangleMesh, np.ndarray]:
        vertices = np.array(mesh.vertices)
        vertices_condition = np.logical_or(
            vertices[:, self.__height_direction[0]] < (self.__open3d_gui_widget_last_state['height_direction_bound_slider'][0] if self.__hide_windows else self.__height_direction_lower_bound_slider.double_value),
            vertices[:, self.__height_direction[0]] > (self.__open3d_gui_widget_last_state['height_direction_bound_slider'][1] if self.__hide_windows else self.__height_direction_upper_bound_slider.double_value))
        mesh.remove_vertices_by_mask(vertices_condition)
        return mesh, vertices_condition