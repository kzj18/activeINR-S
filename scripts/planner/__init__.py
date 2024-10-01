from typing import Dict, Union, List

import cv2
import numpy as np
import networkx as nx

def get_update_precise_map_flag(
    precise_map_last:Dict[str, Union[np.ndarray, Dict[str, Union[nx.Graph, np.ndarray, cv2.Mat, List[List[np.ndarray]]]]]],
    precise_config:Dict[str, float],
    pose_topdown_translation:np.ndarray):
    if precise_map_last['map'] is None or\
        precise_map_last['map_cv2'] is None or\
            precise_map_last['map_x'] is None or\
                precise_map_last['map_y'] is None or\
                    precise_map_last['center_topdown'] is None:
        update_precise_map_flag = True
    elif np.any(np.abs(pose_topdown_translation - precise_map_last['center_topdown']) > precise_config['core_size_topdown'] / 2):
        update_precise_map_flag = True
    else:
        update_precise_map_flag = False
    return update_precise_map_flag