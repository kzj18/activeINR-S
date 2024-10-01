import os
import sys
import re
from typing import Tuple, Union
import json
from enum import Enum

import cv2
import numpy as np
import torch
import quaternion
import trimesh
from trimesh import visual
import open3d as o3d

import rospy

from geometry_msgs.msg import Point, Pose, Quaternion

from scripts import OPENCV_TO_OPENGL
from scripts.entry_points.nodes import GetDatasetConfigResponse

HABITAT_TRANSFORM_MATRIX = np.array([
    [1, 0, 0, 0],
    [0, 0, -1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
])
        
class PoseDataType(Enum):
    C2W_OPENCV = 'C2W_OPENCV'
    C2W_OPENGL = 'C2W_OPENGL'
    W2C_OPENCV = 'W2C_OPENCV'
    W2C_OPENGL = 'W2C_OPENGL'
    
class HeightDirection(Enum):
    X_NEGATIVE = 0
    X_POSITIVE = 1
    Y_NEGATIVE = 2
    Y_POSITIVE = 3
    Z_NEGATIVE = 4
    Z_POSITIVE = 5

class DatasetFormats(Enum):
    REALSENSE = 'realsense'
    MP3D = 'mp3d'
    GIBSON = 'gibson'
    REPLICA = 'replica'
    GOAT = 'goat'
    
def convert_to_c2w_opencv(c2w:np.ndarray, pose_data_type:PoseDataType) -> np.ndarray:
    if pose_data_type in [PoseDataType.C2W_OPENGL, PoseDataType.W2C_OPENGL]:
        # NOTE: Convert to OpenCV coordinate system
        c2w = OPENCV_TO_OPENGL @ c2w @ OPENCV_TO_OPENGL
    if pose_data_type in [PoseDataType.W2C_OPENCV, PoseDataType.W2C_OPENGL]:
        # NOTE: Convert to c2w
        c2w = np.linalg.inv(c2w)
    return c2w
    
def get_dataset_format() -> DatasetFormats:
    for arg_index, arg in enumerate(sys.argv):
        if arg == '--config':
            with open(sys.argv[arg_index + 1]) as f:
                config = json.load(f)
                dataset_format = config['dataset']['format']
                return DatasetFormats(dataset_format)
    raise ValueError("Dataset format not found in config file.")
    
def get_scene_mesh_url(dataset_format:DatasetFormats, dataset_root_url:str, scene_id:str) -> Tuple[Union[str, None], Union[str, None]]:
    assert dataset_format in DatasetFormats, f"Invalid dataset format: {dataset_format.name}"
    if dataset_format == DatasetFormats.REALSENSE:
        habitat_mesh_url = scene_mesh_url = None
    elif dataset_format == DatasetFormats.MP3D:
        habitat_mesh_url = os.path.join(dataset_root_url, 'v1', 'tasks', scene_id, f'{scene_id}.glb')
        scene_mesh_url = os.path.join(dataset_root_url, 'v1', 'tasks', scene_id, f'{scene_id}_semantic.ply')
    elif dataset_format == DatasetFormats.GIBSON:
        habitat_mesh_url = scene_mesh_url = os.path.join(dataset_root_url, f'{scene_id}.glb')
    elif dataset_format == DatasetFormats.REPLICA:
        habitat_mesh_url = scene_mesh_url = os.path.join(dataset_root_url, scene_id, 'mesh.ply')
    elif dataset_format == DatasetFormats.GOAT:
        habitat_mesh_url = scene_mesh_url = None
    else:
        raise NotImplementedError(f"Dataset format: {dataset_format.name} not implemented")
    return habitat_mesh_url, scene_mesh_url

CV_FILENODE_TYPE = {
    cv2.FILE_NODE_NONE: 'FILE_NODE_NONE',
    cv2.FILE_NODE_INT: 'FILE_NODE_INT',
    cv2.FILE_NODE_REAL: 'FILE_NODE_REAL',
    cv2.FILE_NODE_STRING: 'FILE_NODE_STRING',
    cv2.FILE_NODE_SEQ: 'FILE_NODE_SEQ',
    cv2.FILE_NODE_MAP: 'FILE_NODE_MAP',
    cv2.FILE_NODE_TYPE_MASK: 'FILE_NODE_TYPE_MASK',
    cv2.FILE_NODE_FLOAT: 'FILE_NODE_FLOAT',
    cv2.FILE_NODE_FLOW: 'FILE_NODE_FLOW',
    cv2.FILE_NODE_STR: 'FILE_NODE_STR',
    cv2.FILE_NODE_UNIFORM: 'FILE_NODE_UNIFORM',
    cv2.FILE_NODE_EMPTY: 'FILE_NODE_EMPTY',
    cv2.FILE_NODE_NAMED: 'FILE_NODE_NAMED'
}
print(json.dumps(CV_FILENODE_TYPE, indent=4))

def readSeqFileNode(file_node:cv2.FileNode):
    assert(file_node.isSeq())
    res = []
    for i in range(file_node.size()):
        res.append(file_node.at(i).real())
    return res

def readMapFileNode(file_root:cv2.FileNode, deep=False):
    res = dict()
    for key in file_root.keys():
        file_node = file_root.getNode(key)
        file_node_type = file_node.type()
        rospy.logdebug(key, CV_FILENODE_TYPE[file_node_type], ":")
        if file_node_type == cv2.FILE_NODE_INT:
            res[key] = int(file_node.real())
        elif file_node_type == cv2.FILE_NODE_REAL:
            res[key] = file_node.real()
        elif file_node_type == cv2.FILE_NODE_STRING:
            res[key] = file_node.string()
        elif file_node_type == cv2.FILE_NODE_SEQ:
            res[key] = readSeqFileNode(file_node)
        elif file_node_type == cv2.FILE_NODE_MAP:
            res[key] = readMapFileNode(file_node)
            if not deep:
                res[key] = np.reshape(np.array(res[key]["data"]), (int(res[key]["rows"]), int(res[key]["cols"])))
        elif file_node_type == cv2.FILE_NODE_TYPE_MASK:
            raise NotImplementedError
        elif file_node_type == cv2.FILE_NODE_FLOAT:
            res[key] = file_node.real()
        elif file_node_type == cv2.FILE_NODE_FLOW:
            raise NotImplementedError
        elif file_node_type == cv2.FILE_NODE_STR:
            res[key] = file_node.string()
        elif file_node_type == cv2.FILE_NODE_UNIFORM:
            raise NotImplementedError
        elif file_node_type == cv2.FILE_NODE_EMPTY:
            raise NotImplementedError
        elif file_node_type == cv2.FILE_NODE_NAMED:
            raise NotImplementedError
        else:
            raise NotImplementedError
        rospy.logdebug(res[key])
    return res

def readCameraCfg(yamlpath):
    cv_file = cv2.FileStorage(yamlpath, cv2.FILE_STORAGE_READ)
    res = readMapFileNode(cv_file.root())
    return res
    
def load_scene_mesh(scene_mesh_url:str, transform_matrix:np.ndarray) -> Tuple[o3d.geometry.TriangleMesh, np.ndarray]:
    scene_mesh_trimesh = trimesh.load(scene_mesh_url)
    if isinstance(scene_mesh_trimesh, trimesh.Scene):
        scene_mesh_trimesh:trimesh.Trimesh = scene_mesh_trimesh.dump(concatenate=True)
        
    scene_mesh_trimesh.apply_transform(transform_matrix)
    
    scene_mesh_visual = scene_mesh_trimesh.visual
    if isinstance(scene_mesh_visual, visual.TextureVisuals):
        scene_mesh_visual = scene_mesh_visual.to_color()
    scene_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(scene_mesh_trimesh.vertices),
        o3d.utility.Vector3iVector(scene_mesh_trimesh.faces))
    scene_mesh.vertex_colors = o3d.utility.Vector3dVector(scene_mesh_visual.vertex_colors[:, :3] / 255.0)
    scene_mesh.compute_vertex_normals()
    
    scene_bbox_o3d:o3d.geometry.AxisAlignedBoundingBox = scene_mesh.get_axis_aligned_bounding_box()
    scene_bbox = np.array(
        [scene_bbox_o3d.get_min_bound(), scene_bbox_o3d.get_max_bound()]).T
    
    return scene_mesh, scene_bbox

class RGBDSensor:
    
    def __init__(self,
                 height:int,
                 width:int,
                 fx:float,
                 fy:float,
                 cx:float,
                 cy:float,
                 depth_min:float,
                 depth_max:float,
                 depth_scale:float,
                 position:np.ndarray,
                 downsample_factor:float=1.0):
        '''
        depth_max: unit in meters
        depth_min: unit in meters
        '''
        if downsample_factor > 1.0:
            self.height = int(np.ceil(height / downsample_factor))
            self.width = int(np.ceil(width / downsample_factor))
            self.fx = fx * self.width / width
            self.fy = fy * self.height / height
            self.cx = cx * self.width / width
            self.cy = cy * self.height / height
        elif downsample_factor == 1.0:
            self.height = height
            self.width = width
            self.fx = fx
            self.fy = fy
            self.cx = cx
            self.cy = cy
        else:
            raise ValueError(f"Invalid downsample factor: {downsample_factor}")
        self.depth_max = depth_max
        self.depth_min = depth_min
        self.depth_scale = depth_scale
        self.position = position
        self.downsample_factor = downsample_factor
        self.directions = get_camera_rays(self.height, self.width, self.fx, self.fy, self.cx, self.cy)
        
# ROS conversion functions
        
def dataset_config_to_ros(dataset_config:dict) -> GetDatasetConfigResponse:
    dataset_config_ros = dataset_config.copy()
    for key, value in dataset_config.items():
        if issubclass(type(value), (int, float, str)):
            pass
        elif isinstance(value, np.ndarray):
            if value.shape == (3, ):
                dataset_config_ros[key] = Point(*value)
            elif value.shape == (4, 4):
                value_ros = Pose()
                value_ros.position = Point(*value[:3, 3])
                value_ros.orientation = Quaternion(
                    *np.roll(
                        quaternion.as_float_array(quaternion.from_rotation_matrix(value[:3, :3])),
                        -1))
                dataset_config_ros[key] = value_ros
            else:
                raise ValueError(f'Invalid shape of {key}, get {type(value)}')
        else:
            raise ValueError(f'Invalid type of {key}, get {type(value)}')
    return GetDatasetConfigResponse(**dataset_config_ros)

# Camera intrinsics conversion functions

def as_intrinsics_matrix(intrinsics):
    """
    Get matrix representation of intrinsics.

    """
    K = np.eye(3)
    K[0, 0] = intrinsics[0]
    K[1, 1] = intrinsics[1]
    K[0, 2] = intrinsics[2]
    K[1, 2] = intrinsics[3]
    return K

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [int(x) if x.isdigit() else x for x in re.split('([0-9]+)', s)]

def get_camera_rays(H, W, fx, fy=None, cx=None, cy=None, type='OpenGL'):
    """Get ray origins, directions from a pinhole camera."""
    #  ----> i
    # |
    # |
    # X
    # j
    i, j = torch.meshgrid(torch.arange(W, dtype=torch.float32),
                       torch.arange(H, dtype=torch.float32), indexing='xy')
    
    # View direction (X, Y, Lambda) / lambda
    # Move to the center of the screen
    #  -------------
    # |      y      |
    # |      |      |
    # |      .-- x  |
    # |             |
    # |             |
    #  -------------

    if cx is None:
        cx, cy = 0.5 * W, 0.5 * H

    if fy is None:
        fy = fx
    if type == 'OpenGL':
        dirs = torch.stack([(i - cx)/fx, -(j - cy)/fy, -torch.ones_like(i)], -1)
    elif type == 'OpenCV':
        dirs = torch.stack([(i - cx)/fx, (j - cy)/fy, torch.ones_like(i)], -1)
    else:
        raise NotImplementedError()

    rays_d = dirs
    return rays_d