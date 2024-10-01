from enum import Enum
from typing import List, Dict, Union, Tuple, Callable
import os
import time

import cv2
import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist
import networkx as nx
from matplotlib import colors, cm

import rospy

from scripts import start_timing, end_timing

def is_line_segment_out_of_circle(
    line_segment_start:np.ndarray,
    line_segment_end:np.ndarray,
    circle_center:np.ndarray,
    circle_radius:float) -> np.ndarray:
    line_segment_start_to_circle_center = circle_center - line_segment_start
    line_segment_end_to_circle_center = circle_center - line_segment_end
    line_segment_start_to_end = line_segment_end - line_segment_start
    line_segment_end_to_start = -line_segment_start_to_end
    line_segment_start_dot:np.ndarray = np.einsum('ij,ij->i', line_segment_start_to_circle_center, line_segment_start_to_end)
    line_segment_end_dot:np.ndarray = np.einsum('ij,ij->i', line_segment_end_to_circle_center, line_segment_end_to_start)
    vertical_foot_on_line_segment = np.logical_and(
        line_segment_start_dot >= 0,
        line_segment_end_dot >= 0)
    line_segment_length = np.linalg.norm(line_segment_start_to_end, axis=1)
    assert np.all(line_segment_length > 0), 'Line segment length should be greater than zero.'
    line_segment_vertical_foot_distance:np.ndarray = np.abs(np.cross(line_segment_start_to_end, line_segment_start_to_circle_center)) / line_segment_length
    line_segment_vertical_foot_in_circle_condition = np.logical_and(
        vertical_foot_on_line_segment,
        line_segment_vertical_foot_distance <= circle_radius)
    line_segment_start_in_circle_condition = np.linalg.norm(line_segment_start_to_circle_center, axis=1) <= circle_radius
    line_segment_end_in_circle_condition = np.linalg.norm(line_segment_end_to_circle_center, axis=1) <= circle_radius
    line_segment_out_of_circle_condition = np.logical_and(
        np.logical_not(line_segment_vertical_foot_in_circle_condition),
        np.logical_and(
            np.logical_not(line_segment_start_in_circle_condition),
            np.logical_not(line_segment_end_in_circle_condition)))
    return line_segment_out_of_circle_condition

def splat_inaccessible_database(
    agent_position:np.ndarray,
    global_obstacle_map:np.ndarray,
    inaccessible_database:Dict[Tuple[float, float], np.ndarray],
    splat_size_pixel:float) -> np.ndarray:
    global_obstacle_map_ = global_obstacle_map.copy()
    global_obstacle_map_vis = cv2.cvtColor(global_obstacle_map_, cv2.COLOR_GRAY2BGR)
    splat_radius = max(np.int32(np.round(splat_size_pixel / 2)), 1)
    splat_flag = False
    for translation, rotation_vectors in inaccessible_database.items():
        translation_np = np.array(translation)
        splat_centers:np.ndarray = translation_np + rotation_vectors / np.linalg.norm(rotation_vectors, axis=1)[:, np.newaxis] * splat_size_pixel
        splat_centers = np.int32(np.round(splat_centers))
        splat_centers_condition = np.logical_and(
            np.logical_and(
                0 <= splat_centers[:, 0],
                splat_centers[:, 0] < global_obstacle_map_.shape[1]),
            np.logical_and(
                0 <= splat_centers[:, 1],
                splat_centers[:, 1] < global_obstacle_map_.shape[0]))
        splat_centers = splat_centers[splat_centers_condition]
        for splat_center in splat_centers.tolist():
            global_obstacle_map_ = cv2.circle(
                global_obstacle_map_,
                splat_center,
                splat_radius,
                0,
                -1)
            global_obstacle_map_vis = cv2.circle(
                global_obstacle_map_vis,
                splat_center,
                splat_radius,
                (0, 0, 255),
                -1)
            splat_flag = True
    if splat_flag:
        test_splat_dir = os.path.join(os.getcwd(), 'test', 'test_splat')
        os.makedirs(test_splat_dir, exist_ok=True)
        current_time = time.strftime('%Y-%m-%d_%H-%M-%S')
        global_obstacle_map_vis = cv2.circle(
            global_obstacle_map_vis,
            np.int32(agent_position),
            np.int32(np.ceil(splat_size_pixel / 2)),
            (0, 255, 0),
            -1)
        cv2.imwrite(os.path.join(test_splat_dir, current_time + '_raw.png'), global_obstacle_map)
        cv2.imwrite(os.path.join(test_splat_dir, current_time + '_splat.png'), global_obstacle_map_vis)
    return global_obstacle_map_

def get_obstacle_map(
    global_obstacle_map:np.ndarray,
    agent_position:np.ndarray,
    approx_precision:float,
    accept_no_approx_children:bool=False) -> Tuple[cv2.Mat, np.ndarray, List[np.ndarray]]:
    # FIXME: Consider when there are multiple white ring outside the robot contour
    get_obstacle_map_timing = start_timing()
    global_obstacle_map_contours, _ = cv2.findContours(
        global_obstacle_map,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    agent_to_contours = np.array([cv2.pointPolygonTest(contour, agent_position, True) for contour in global_obstacle_map_contours])
    agent_to_contours_condition = agent_to_contours >= 0
    if np.any(agent_to_contours_condition):
        contours_index_selected = np.where(agent_to_contours_condition)[0]
        agent_to_contours_selected = agent_to_contours[contours_index_selected]
        contour_index_selected = contours_index_selected[np.argmin(agent_to_contours_selected)]
        local_obstacle_map_contour = global_obstacle_map_contours[contour_index_selected]
    else:
        rospy.logwarn('Robot position is not in the obstacle map.')
        if len(global_obstacle_map_contours) > 1:
            global_obstacle_map_contours_area = np.array([cv2.contourArea(contour) for contour in global_obstacle_map_contours])
            contour_index_selected = np.where(global_obstacle_map_contours_area > np.mean(global_obstacle_map_contours_area))[0]
            global_obstacle_map_contours = np.array(global_obstacle_map_contours, dtype=object)
            global_obstacle_map_contours_selected = global_obstacle_map_contours[contour_index_selected]
            agent_to_contours_selected = agent_to_contours[contour_index_selected]
            local_obstacle_map_contour = global_obstacle_map_contours_selected[np.argmax(agent_to_contours_selected)]
        elif len(global_obstacle_map_contours) == 1:
            local_obstacle_map_contour = global_obstacle_map_contours[0]
        else:
            raise ValueError('There is no contour in the obstacle map.')
    local_obstacle_map_approx_contour = cv2.approxPolyDP(
        local_obstacle_map_contour,
        approx_precision,
        True)
    white = np.ones_like(global_obstacle_map) * 255
    black = np.zeros_like(global_obstacle_map)
    
    local_obstacle_map_approx_inverse = cv2.drawContours(white.copy(), [local_obstacle_map_approx_contour], -1, 0, -1)
    local_obstacle_map_inverse = cv2.drawContours(white.copy(), [local_obstacle_map_contour], -1, 0, -1)
    local_obstacle_map_approx:cv2.Mat = cv2.drawContours(black.copy(), [local_obstacle_map_approx_contour], -1, 255, -1)
    obstacle_map_of_children = cv2.bitwise_or(
        cv2.bitwise_or(
            local_obstacle_map_inverse,
            local_obstacle_map_approx_inverse),
        global_obstacle_map)
    obstacle_map_of_children_inverse = cv2.bitwise_not(obstacle_map_of_children)
    obstacle_map_children_contours, _ = cv2.findContours(
        obstacle_map_of_children_inverse,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    obstacle_map_children_approx_contours = []
    children_contours_num = len(obstacle_map_children_contours)
    children_contours_area_bigger_than_zero_num = 0
    for contour in obstacle_map_children_contours:
        if cv2.contourArea(contour) > 0:
            children_contours_area_bigger_than_zero_num += 1
            approx_contour = cv2.approxPolyDP(contour, approx_precision, True)
            if cv2.contourArea(approx_contour) > 0:
                obstacle_map_children_approx_contours.append(approx_contour)
            elif accept_no_approx_children:
                obstacle_map_children_approx_contours.append(contour)
    if len(obstacle_map_children_approx_contours) == 0:
        if children_contours_num > 0 and children_contours_area_bigger_than_zero_num > 0 and accept_no_approx_children:
            rospy.logwarn(f'There is no children contour in the obstacle map, bigger than zero: {children_contours_area_bigger_than_zero_num}/{children_contours_num}.')
        obstacle_map = local_obstacle_map_approx.copy()
    else:
        obstacle_map = cv2.drawContours(local_obstacle_map_approx.copy(), obstacle_map_children_approx_contours, -1, 0, -1)
    rospy.logdebug(f'Get obstacle map timing: {end_timing(*get_obstacle_map_timing)} ms')
    return obstacle_map, local_obstacle_map_approx_contour, obstacle_map_children_approx_contours

def get_voronoi_graph(
    obstacle_map:cv2.Mat,
    obstacle_map_approx_contour:np.ndarray,
    obstacle_map_children_approx_contours:List[np.ndarray],
    edge_sample_num:int,
    agent_radius_pixel:float,
    inaccessible_points:np.ndarray,
    precise_min_coord:np.ndarray=None,
    precise_max_coord:np.ndarray=None,
    precise_map_approx_contour:np.ndarray=None,
    precise_map_children_approx_contours:List[np.ndarray]=None,
    topdown_to_precise_ratio:float=None) -> Dict[str, Union[nx.Graph, np.ndarray, cv2.Mat, List[List[np.ndarray]]]]:
    '''
    Given the obstacle map and the robot position, return the voronoi graph.
    '''
    edge_length_min = np.inf
    obstacle_map_approx_contours_vertices = []
    obstacle_map_approx_contours_edges_length = []
    for contour in ([obstacle_map_approx_contour] + obstacle_map_children_approx_contours):
        contour_vertices = contour.reshape(-1, 2)
        contour_edges_length = np.linalg.norm(contour_vertices - np.roll(contour_vertices, 1, axis=0), axis=1)
        obstacle_map_approx_contours_vertices.append(contour_vertices)
        obstacle_map_approx_contours_edges_length.append(contour_edges_length)
        contour_edges_length_min = np.min(contour_edges_length)
        if contour_edges_length_min > 0:
            edge_length_min = min(edge_length_min, contour_edges_length_min)
    assert edge_length_min != np.inf, 'Edge length min should be less than infinity.'
    edge_sample_resolution = edge_length_min / edge_sample_num
    
    obstacle_points = np.array([]).reshape(0, 2)
    for contour_vertices, contour_edges_length in zip(
        obstacle_map_approx_contours_vertices,
        obstacle_map_approx_contours_edges_length):
        for vertex_start, vertex_end, edge_length in zip(
            contour_vertices,
            np.roll(contour_vertices, 1, axis=0),
            contour_edges_length):
            edge_sample_num = int(edge_length / edge_sample_resolution)
            edge_sample = np.linspace(vertex_start, vertex_end, edge_sample_num, endpoint=False)
            obstacle_points = np.vstack((obstacle_points, edge_sample))
            
    if (precise_min_coord is not None) and\
        (precise_max_coord is not None) and\
            (precise_map_approx_contour is not None) and\
                (precise_map_children_approx_contours is not None) and\
                    (topdown_to_precise_ratio is not None):
        obstacle_points_inside_condition = is_positions_in_precise_range(
            obstacle_points,
            precise_min_coord,
            precise_max_coord)
        obstacle_points = obstacle_points[np.logical_not(obstacle_points_inside_condition)]
        precise_edge_length_min = np.inf
        precise_map_approx_contours_vertices = []
        precise_map_approx_contours_edges_length = []
        for precise_contour in ([precise_map_approx_contour] + precise_map_children_approx_contours):
            precise_contour_vertices = precise_contour.reshape(-1, 2)
            precise_contour_edges_length = np.linalg.norm(precise_contour_vertices - np.roll(precise_contour_vertices, 1, axis=0), axis=1)
            precise_map_approx_contours_vertices.append(precise_contour_vertices)
            precise_map_approx_contours_edges_length.append(precise_contour_edges_length)
            precise_edge_length_min = np.min(precise_contour_edges_length)
            if precise_edge_length_min > 0:
                precise_edge_length_min = min(precise_edge_length_min, precise_edge_length_min)
        assert precise_edge_length_min != np.inf, 'Precise edge length min should be less than infinity.'
        precise_edge_sample_resolution = precise_edge_length_min / edge_sample_num
        
        precise_obstacle_points = np.array([]).reshape(0, 2)
        for precise_contour_vertices, precise_contour_edges_length in zip(
            precise_map_approx_contours_vertices,
            precise_map_approx_contours_edges_length):
            for vertex_start, vertex_end, edge_length in zip(
                precise_contour_vertices,
                np.roll(precise_contour_vertices, 1, axis=0),
                precise_contour_edges_length):
                edge_sample_num = int(edge_length / precise_edge_sample_resolution)
                edge_sample = np.linspace(vertex_start, vertex_end, edge_sample_num, endpoint=False)
                precise_obstacle_points = np.vstack((precise_obstacle_points, edge_sample))
                
        precise_obstacle_points = translations_precise_to_topdown(
            precise_obstacle_points,
            precise_min_coord,
            1 / topdown_to_precise_ratio)
        
        precise_obstacle_points_inside_condition = is_positions_in_precise_range(
            precise_obstacle_points,
            precise_min_coord,
            precise_max_coord)
        
        precise_obstacle_points = precise_obstacle_points[precise_obstacle_points_inside_condition]
                
        obstacle_points = np.vstack([
            obstacle_points,
            precise_obstacle_points])
        
    voronoi_graph = Voronoi(obstacle_points)
    
    voronoi_graph_ridge_vertices = np.array(voronoi_graph.ridge_vertices)
    voronoi_graph_ridge_vertices = voronoi_graph_ridge_vertices[np.all(voronoi_graph_ridge_vertices >= 0, axis=1)]
    voronoi_graph_ridge_matrix = np.zeros((len(voronoi_graph.vertices), len(voronoi_graph.vertices)))
    voronoi_graph_vertices:np.ndarray = voronoi_graph.vertices
    if (precise_min_coord is not None) and\
        (precise_max_coord is not None) and\
            (precise_map_approx_contour is not None) and\
                (precise_map_children_approx_contours is not None) and\
                    (topdown_to_precise_ratio is not None):
        voronoi_graph_vertices_inside_condition = is_positions_in_precise_range(
            voronoi_graph_vertices,
            precise_min_coord,
            precise_max_coord)
        voronoi_graph_vertices_precise = translations_topdown_to_precise(
            voronoi_graph_vertices,
            precise_min_coord,
            topdown_to_precise_ratio)
    else:
        voronoi_graph_vertices_inside_condition = np.zeros(len(voronoi_graph_vertices), dtype=bool)
        voronoi_graph_vertices_precise = voronoi_graph_vertices.copy()
    voronoi_graph_ridge_matrix[voronoi_graph_ridge_vertices[:, 0], voronoi_graph_ridge_vertices[:, 1]] = 1
    voronoi_graph_ridge_matrix[voronoi_graph_ridge_vertices[:, 1], voronoi_graph_ridge_vertices[:, 0]] = 1
        
    vertices_indexes = []
    for vertex_index, (vertex, vertex_precise, vertex_inside) in enumerate(zip(voronoi_graph_vertices, voronoi_graph_vertices_precise, voronoi_graph_vertices_inside_condition)):
        if vertex_inside:
            if cv2.pointPolygonTest(precise_map_approx_contour, vertex_precise, True) > agent_radius_pixel * topdown_to_precise_ratio:
                in_children_obstacle_flag = False
                for precise_map_children_approx_contour in precise_map_children_approx_contours:
                    if cv2.pointPolygonTest(precise_map_children_approx_contour, vertex_precise, True) > -agent_radius_pixel * topdown_to_precise_ratio:
                        in_children_obstacle_flag = True
                        break
                if not in_children_obstacle_flag:
                    vertices_indexes.append(vertex_index)
        else:
            if cv2.pointPolygonTest(obstacle_map_approx_contour, vertex, True) > agent_radius_pixel:
                in_children_obstacle_flag = False
                for obstacle_map_children_approx_contour in obstacle_map_children_approx_contours:
                    if cv2.pointPolygonTest(obstacle_map_children_approx_contour, vertex, True) > -agent_radius_pixel:
                        in_children_obstacle_flag = True
                        break
                if not in_children_obstacle_flag:
                    vertices_indexes.append(vertex_index)
    voronoi_graph_vertices = voronoi_graph_vertices[vertices_indexes]
    voronoi_graph_ridge_matrix = voronoi_graph_ridge_matrix[vertices_indexes][:, vertices_indexes]
    
    voronoi_graph_vertices_connectivity = np.sum(voronoi_graph_ridge_matrix, axis=1)
    voronoi_graph_vertices_condition = voronoi_graph_vertices_connectivity > 0
    voronoi_graph_vertices = voronoi_graph_vertices[voronoi_graph_vertices_condition]
    voronoi_graph_ridge_matrix = voronoi_graph_ridge_matrix[voronoi_graph_vertices_condition][:, voronoi_graph_vertices_condition]
    voronoi_graph_vertices_connectivity = np.sum(voronoi_graph_ridge_matrix, axis=1)
    
    voronoi_graph_vertices_fix_condition = voronoi_graph_vertices_connectivity >= 3
    
    if len(inaccessible_points) > 0:
        inaccessible_points_distance_to_voronoi_graph_vertices = cdist(inaccessible_points, voronoi_graph_vertices)
        close_vertices_sorted_index = np.argsort(inaccessible_points_distance_to_voronoi_graph_vertices, axis=1)
        close_vertices_start_index = close_vertices_sorted_index[:, 0]
        close_vertices_end_index = close_vertices_sorted_index[:, 1]
        close_vertices_connectivity = np.bool8(voronoi_graph_ridge_matrix[close_vertices_start_index, close_vertices_end_index])
        inaccessible_points_selected = inaccessible_points[close_vertices_connectivity]
        close_vertices_start_index_selected = close_vertices_start_index[close_vertices_connectivity]
        close_vertices_end_index_selected = close_vertices_end_index[close_vertices_connectivity]
        close_vertices_start_selected = voronoi_graph_vertices[close_vertices_start_index_selected]
        close_vertices_end_selected = voronoi_graph_vertices[close_vertices_end_index_selected]
        line_segement_selected_out_of_circle_condition = is_line_segment_out_of_circle(
            close_vertices_start_selected,
            close_vertices_end_selected,
            inaccessible_points_selected,
            agent_radius_pixel)
        line_segement_selected_not_out_of_circle_condition = np.logical_not(line_segement_selected_out_of_circle_condition)
        pruned_vertex_start_index = close_vertices_start_index_selected[line_segement_selected_not_out_of_circle_condition]
        pruned_vertex_end_index = close_vertices_end_index_selected[line_segement_selected_not_out_of_circle_condition]
        pruned_vertex_index = np.unique(np.hstack((pruned_vertex_start_index, pruned_vertex_end_index)))
                
        voronoi_graph_vertices_inaccessible_condition = np.zeros(len(voronoi_graph_vertices), dtype=bool)
        voronoi_graph_vertices_inaccessible_condition[pruned_vertex_index] = True
        voronoi_graph_vertices_inaccessible_condition = np.logical_and(
            voronoi_graph_vertices_inaccessible_condition,
            np.logical_not(voronoi_graph_vertices_fix_condition))
        if np.any(voronoi_graph_vertices_inaccessible_condition):
            rospy.logwarn('Inaccessible points are detected in the voronoi graph.')
        voronoi_graph_vertices_accessibile_condition = np.logical_not(voronoi_graph_vertices_inaccessible_condition)
        voronoi_graph_vertices = voronoi_graph_vertices[voronoi_graph_vertices_accessibile_condition]
        voronoi_graph_ridge_matrix = voronoi_graph_ridge_matrix[voronoi_graph_vertices_accessibile_condition][:, voronoi_graph_vertices_accessibile_condition]
        voronoi_graph_vertices_connectivity = np.sum(voronoi_graph_ridge_matrix, axis=1)
        voronoi_graph_vertices_fix_condition = voronoi_graph_vertices_fix_condition[voronoi_graph_vertices_accessibile_condition]
        voronoi_graph_vertices_nodes_index = np.where(voronoi_graph_vertices_fix_condition)[0]
    
    pruned_chains:List[List[np.ndarray]] = []
    while True:
        voronoi_graph_vertices_nodes_index = np.where(voronoi_graph_vertices_fix_condition)[0]
        voronoi_graph_vertices_connectivity = np.sum(voronoi_graph_ridge_matrix, axis=1)
        pruned_vertices_index = np.where(voronoi_graph_vertices_connectivity <= 1)[0]
        pruned_vertices_index = np.setdiff1d(pruned_vertices_index, voronoi_graph_vertices_nodes_index)
        remain_vertices_index = np.setdiff1d(np.arange(len(voronoi_graph_vertices)), pruned_vertices_index)
        if len(pruned_vertices_index) == 0:
            break
        else:
            if len(pruned_chains) == 0:
                for pruned_vertex_index in pruned_vertices_index:
                    if np.sum(voronoi_graph_ridge_matrix[pruned_vertex_index]) == 0:
                        continue
                    else:
                        pruned_vertex_index_next = np.where(voronoi_graph_ridge_matrix[pruned_vertex_index])[0][0]
                        pruned_chain = [
                            voronoi_graph_vertices[pruned_vertex_index],
                            voronoi_graph_vertices[pruned_vertex_index_next]]
                        pruned_chains.append(pruned_chain)
            else:
                isolated_chains_index = []
                for pruned_vertex_index in pruned_vertices_index:
                    if np.sum(voronoi_graph_ridge_matrix[pruned_vertex_index]) == 0:
                        for pruned_chain_index, pruned_chain in enumerate(pruned_chains):
                            if np.allclose(pruned_chain[-1], voronoi_graph_vertices[pruned_vertex_index]):
                                isolated_chains_index.append(pruned_chain_index)
                                break
                    else:
                        pruned_vertex_index_next = np.where(voronoi_graph_ridge_matrix[pruned_vertex_index])[0][0]
                        for pruned_chain_index, pruned_chain in enumerate(pruned_chains):
                            if np.allclose(pruned_chain[-1], voronoi_graph_vertices[pruned_vertex_index]):
                                pruned_chains[pruned_chain_index].append(voronoi_graph_vertices[pruned_vertex_index_next])
                                break
                
                isolated_chains_index = np.unique(isolated_chains_index)
                pruned_chains = [pruned_chain for pruned_chain_index, pruned_chain in enumerate(pruned_chains) if pruned_chain_index not in isolated_chains_index]
                
            voronoi_graph_vertices = voronoi_graph_vertices[remain_vertices_index]
            voronoi_graph_ridge_matrix = voronoi_graph_ridge_matrix[remain_vertices_index][:, remain_vertices_index]
            voronoi_graph_vertices_fix_condition = voronoi_graph_vertices_fix_condition[remain_vertices_index]
    
    assert np.sum(voronoi_graph_ridge_matrix[np.diag_indices(len(voronoi_graph_vertices))]) == 0, 'Diagonal elements of the voronoi graph ridge matrix should be zero.'
    voronoi_graph_ridge_vertices = np.argwhere(np.triu(voronoi_graph_ridge_matrix))
    voronoi_graph_ridge_edges_length = np.linalg.norm(
        voronoi_graph_vertices[voronoi_graph_ridge_vertices[:, 0]] -\
            voronoi_graph_vertices[voronoi_graph_ridge_vertices[:, 1]],
            axis=1)
    voronoi_graph_ridge_matrix[voronoi_graph_ridge_vertices[:, 0], voronoi_graph_ridge_vertices[:, 1]] =\
        voronoi_graph_ridge_edges_length
    voronoi_graph_ridge_matrix[voronoi_graph_ridge_vertices[:, 1], voronoi_graph_ridge_vertices[:, 0]] =\
        voronoi_graph_ridge_edges_length
    
    return {
        'graph': nx.from_numpy_array(voronoi_graph_ridge_matrix),
        'vertices': voronoi_graph_vertices,
        'obstacle_map': obstacle_map,
        'pruned_chains': pruned_chains,
        'nodes_index': voronoi_graph_vertices_nodes_index}
    
def anchor_targets_frustums_to_voronoi_graph(
    vertices:np.ndarray,
    nodes_index:np.ndarray,
    targets_frustums_translation:np.ndarray) -> Tuple[Dict[int, List[int]], Dict[int, int]]:
    nodes_index_anchor_targets_frustums_index:Dict[int, List[int]] = {}
    for node_index in nodes_index:
        nodes_index_anchor_targets_frustums_index[node_index] = []
    targets_frustums_index_to_nodes_index:Dict[int, int] = {}
    nodes = np.reshape(vertices[nodes_index], (-1, 2))
    if len(nodes) == 0 or len(targets_frustums_translation) == 0:
        return nodes_index_anchor_targets_frustums_index, targets_frustums_index_to_nodes_index
    nodes_to_targets_frustums = cdist(nodes, targets_frustums_translation)
    node_count_closest_to_every_target_frustum = np.argmin(nodes_to_targets_frustums, axis=0)
    for target_frustum_index, node_count in enumerate(node_count_closest_to_every_target_frustum):
        node_index = nodes_index[node_count]
        nodes_index_anchor_targets_frustums_index[node_index].append(target_frustum_index)
        targets_frustums_index_to_nodes_index[target_frustum_index] = node_index
    return nodes_index_anchor_targets_frustums_index, targets_frustums_index_to_nodes_index
    
def draw_voronoi_graph(
    background:cv2.Mat,
    voronoi_graph_vertices:np.ndarray,
    voronoi_graph_ridge_matrix:np.ndarray,
    voronoi_graph_nodes_index:np.ndarray,
    voronoi_graph_nodes_score:np.ndarray,
    voronoi_graph_nodes_score_max:int,
    voronoi_graph_nodes_score_min:int,
    pruned_chains:List[List[np.ndarray]],
    nodes_index_anchor_targets_frustums_index:Dict[int, List[int]],
    targets_frustums_translation:np.ndarray,
    voronoi_graph_ridge_color:List[int],
    voronoi_graph_ridge_thickness:int,
    voronoi_graph_nodes_colormap:colors.Colormap,
    voronoi_graph_nodes_radius:int,
    pruned_chains_color:List[int],
    pruned_chains_thickness:int,
    targets_frustums_color:List[int],
    targets_frustums_radius:int,
    anchor_lines_color:List[int],
    anchor_lines_thickness:int) -> cv2.Mat:
    
    voronoi_graph_image = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
    for pruned_chain in pruned_chains:
        cv2.polylines(voronoi_graph_image, [np.array(pruned_chain).astype(np.int32)], False, pruned_chains_color, pruned_chains_thickness)
        
    voronoi_graph_ridge_vertices = np.argwhere(np.triu(voronoi_graph_ridge_matrix))
    for voronoi_graph_ridge_vertex in voronoi_graph_ridge_vertices:
        vertex_start:np.ndarray = voronoi_graph_vertices[voronoi_graph_ridge_vertex[0]]
        vertex_end:np.ndarray = voronoi_graph_vertices[voronoi_graph_ridge_vertex[1]]
        cv2.line(
            voronoi_graph_image,
            vertex_start.astype(np.int32),
            vertex_end.astype(np.int32),
            voronoi_graph_ridge_color,
            voronoi_graph_ridge_thickness)
        
    for voronoi_graph_node_index, voronoi_graph_node_score in zip(voronoi_graph_nodes_index, voronoi_graph_nodes_score):
        voronoi_graph_node = voronoi_graph_vertices[voronoi_graph_node_index]
        targets_frustums_index = nodes_index_anchor_targets_frustums_index[voronoi_graph_node_index]
        for target_frustum_index in targets_frustums_index:
            target_frustum_translation = targets_frustums_translation[target_frustum_index]
            cv2.line(
                voronoi_graph_image,
                np.int32(voronoi_graph_node),
                np.int32(target_frustum_translation),
                anchor_lines_color,
                anchor_lines_thickness)
            cv2.circle(
                voronoi_graph_image,
                np.int32(target_frustum_translation),
                targets_frustums_radius,
                targets_frustums_color,
                -1)
        cv2.circle(
            voronoi_graph_image,
            np.int32(voronoi_graph_node),
            voronoi_graph_nodes_radius,
            np.uint8(
                np.array(voronoi_graph_nodes_colormap(
                    (voronoi_graph_node_score - voronoi_graph_nodes_score_min) / (voronoi_graph_nodes_score_max - voronoi_graph_nodes_score_min))[:3]) * 255).tolist()[::-1],
            -1)
        
    return voronoi_graph_image

def get_closest_vertex_index(
    voronoi_graph_vertices:np.ndarray,
    obstacle_map:cv2.Mat,
    agent_position:np.ndarray,
    agent_radius_pixel:float) -> int:
    voronoi_graph_vertices_to_agent_distance = np.linalg.norm(voronoi_graph_vertices - agent_position, axis=1)
    voronoi_graph_vertices_index_sorted = np.argsort(voronoi_graph_vertices_to_agent_distance)
    voronoi_graph_vertices_sorted = voronoi_graph_vertices[voronoi_graph_vertices_index_sorted]
    free_space_pixels_num = cv2.countNonZero(obstacle_map)
    agent_mask = cv2.circle(
        np.zeros_like(obstacle_map),
        np.int32(agent_position),
        int(np.ceil(agent_radius_pixel)),
        255,
        -1)
    for vertex_index, vertex in enumerate(voronoi_graph_vertices_sorted):
        line_test_result = cv2.line(
            obstacle_map.copy(),
            np.int32(agent_position),
            np.int32(vertex),
            255,
            int(np.ceil(agent_radius_pixel * 2)))
        line_test_result[agent_mask > 0] = obstacle_map[agent_mask > 0]
        if cv2.countNonZero(line_test_result) == free_space_pixels_num:
            return voronoi_graph_vertices_index_sorted[vertex_index]
    line_test_results = []
    for vertex_index, vertex in enumerate(voronoi_graph_vertices_sorted):
        line_test_result = cv2.line(
            obstacle_map.copy(),
            np.int32(agent_position),
            np.int32(vertex),
            255,
            1)
        line_test_results.append(cv2.countNonZero(line_test_result))
        if line_test_results[-1] == free_space_pixels_num:
            return voronoi_graph_vertices_index_sorted[vertex_index]
    vertex_index = np.argmin(line_test_results)
    return voronoi_graph_vertices_index_sorted[vertex_index]

def get_safe_dijkstra_path(
    voronoi_graph:nx.Graph,
    vertex_start_index:int,
    vertex_end_index:int,
    voronoi_graph_vertices:np.ndarray,
    obstacle_map:cv2.Mat,
    agent_position:np.ndarray,
    agent_radius_pixel:float,
    fast_forward_radius_ratio:float=1.0) -> Tuple[np.ndarray, np.ndarray, bool]:
    try:
        navigation_path_index = nx.dijkstra_path(voronoi_graph, vertex_start_index, vertex_end_index)
    except nx.NetworkXNoPath:
        return None, None, False
    free_space_pixels_num = cv2.countNonZero(obstacle_map)
    navigation_path = voronoi_graph_vertices[navigation_path_index]
        
    navigation_path = optimize_navigation_path_using_fast_forward(
        navigation_path=navigation_path,
        obstacle_map=obstacle_map,
        agent_position=agent_position,
        agent_radius_pixel=agent_radius_pixel * fast_forward_radius_ratio)
    
    line_test_result = cv2.polylines(
        obstacle_map.copy(),
        [np.int32(navigation_path)],
        False,
        255,
        int(np.ceil(agent_radius_pixel * 2)))
    if cv2.countNonZero(line_test_result) == free_space_pixels_num:
        return navigation_path_index, navigation_path, True
    else:
        return None, None, True
    
def is_positions_in_precise_range(
    positions_topdown:np.ndarray,
    precise_min_coord:np.ndarray,
    precise_max_coord:np.ndarray) -> np.ndarray:
    positions_topdown_ = positions_topdown.reshape(-1, 2)
    return np.logical_and(
        np.logical_and(
            precise_min_coord[0] < positions_topdown_[:, 0],
            precise_min_coord[1] < positions_topdown_[:, 1]),
        np.logical_and(
            positions_topdown_[:, 0] < precise_max_coord[0],
            positions_topdown_[:, 1] < precise_max_coord[1]))
    
def optimize_navigation_path_using_precise_info(
    agent_position_topdown:np.ndarray,
    navigation_path:np.ndarray,
    agent_radius_pixel_topdown:float,
    topdown_to_precise_ratio:float,
    precise_min_coord:np.ndarray,
    precise_max_coord:np.ndarray,
    precise_voronoi_graph:nx.Graph,
    precise_voronoi_graph_vertices:np.ndarray,
    precise_strict_voronoi_graph:nx.Graph,
    precise_strict_voronoi_graph_vertices:np.ndarray,
    precise_map:np.ndarray) -> np.ndarray:
    navigation_path_strict, navigation_path_success = optimize_navigation_path_using_precise_map(
        agent_position_topdown,
        navigation_path,
        agent_radius_pixel_topdown,
        topdown_to_precise_ratio,
        precise_min_coord,
        precise_max_coord,
        precise_strict_voronoi_graph,
        precise_strict_voronoi_graph_vertices,
        precise_map)
    if navigation_path_success:
        return navigation_path_strict
    else:
        return optimize_navigation_path_using_precise_map(
            agent_position_topdown,
            navigation_path,
            agent_radius_pixel_topdown,
            topdown_to_precise_ratio,
            precise_min_coord,
            precise_max_coord,
            precise_voronoi_graph,
            precise_voronoi_graph_vertices,
            precise_map)[0]
    
def optimize_navigation_path_using_precise_map(
    agent_position_topdown:np.ndarray,
    navigation_path:np.ndarray,
    agent_radius_pixel_topdown:float,
    topdown_to_precise_ratio:float,
    precise_min_coord:np.ndarray,
    precise_max_coord:np.ndarray,
    precise_voronoi_graph:nx.Graph,
    precise_voronoi_graph_vertices:np.ndarray,
    precise_map:np.ndarray) -> Tuple[np.ndarray, bool]:
    
    translations_topdown_to_precise_:Callable[[np.ndarray], np.ndarray] = lambda t_t: translations_topdown_to_precise(t_t, precise_min_coord, topdown_to_precise_ratio)
    translations_precise_to_topdown_:Callable[[np.ndarray], np.ndarray] = lambda t_p: translations_precise_to_topdown(t_p, precise_min_coord, 1 / topdown_to_precise_ratio)

    navigation_path_inside_condition = is_positions_in_precise_range(
        navigation_path,
        precise_min_coord,
        precise_max_coord)
    
    navigation_path_first_outside_index = np.argmin(navigation_path_inside_condition)
    
    if navigation_path_first_outside_index == 0:
        return navigation_path, True
    
    agent_position_precise = translations_topdown_to_precise_(agent_position_topdown)
    agent_radius_pixel_precise = agent_radius_pixel_topdown * topdown_to_precise_ratio
    agent_radius_pixel_precise_int = int(agent_radius_pixel_precise)
    
    agent_mask = cv2.circle(
        np.zeros_like(precise_map),
        np.int32(agent_position_precise),
        int(np.ceil(agent_radius_pixel_precise)),
        255,
        -1)
    free_space_pixels_num = cv2.countNonZero(precise_map)
    
    navigation_path_inside_precise = translations_topdown_to_precise_(navigation_path[:navigation_path_first_outside_index].reshape(-1, 2))
    
    for navigation_path_test_index in range(navigation_path_first_outside_index - 1, -1, -1):
        navigation_path_test_precise = navigation_path_inside_precise[navigation_path_test_index:navigation_path_first_outside_index]
        line_test_result = cv2.polylines(
            precise_map.copy(),
            [np.int32(navigation_path_test_precise)],
            False,
            255,
            int(np.ceil(agent_radius_pixel_precise * 3)))
        line_test_result[agent_mask > 0] = precise_map[agent_mask > 0]
        if cv2.countNonZero(line_test_result) != free_space_pixels_num:
            if navigation_path_test_index + 1 == navigation_path_first_outside_index:
                return navigation_path, False
            break
        else:
            line_test_result = cv2.polylines(
                precise_map.copy(),
                [np.int32(np.vstack([agent_position_precise, navigation_path_test_precise]))],
                False,
                255,
                int(np.ceil(agent_radius_pixel_precise * 3)))
            line_test_result[agent_mask > 0] = precise_map[agent_mask > 0]
            if cv2.countNonZero(line_test_result) == free_space_pixels_num:
                return navigation_path[navigation_path_test_index:], True
    
    target_precise = navigation_path_inside_precise[navigation_path_first_outside_index - 1]
    
    vertex_start_index = get_closest_vertex_index(
        precise_voronoi_graph_vertices,
        precise_map,
        agent_position_precise,
        agent_radius_pixel_precise)
    
    vertex_end_index = get_closest_vertex_index(
        precise_voronoi_graph_vertices,
        precise_map,
        target_precise,
        agent_radius_pixel_precise)
    
    navigation_path_index, navigation_path_precise, navigation_path_success = get_safe_dijkstra_path(
        precise_voronoi_graph,
        vertex_start_index,
        vertex_end_index,
        precise_voronoi_graph_vertices,
        precise_map,
        agent_position_precise,
        agent_radius_pixel_precise * 0.5,
        fast_forward_radius_ratio=2)
    
    if navigation_path_precise is not None:
        navigation_path_precise = np.vstack([navigation_path_precise, target_precise.reshape(1, 2)])
        
        test_optimize_dir = os.path.join(os.getcwd(), 'test', 'test_optimize')
        os.makedirs(test_optimize_dir, exist_ok=True)
        current_time = time.strftime('%Y-%m-%d_%H-%M-%S')
        precise_map_vis = cv2.cvtColor(precise_map, cv2.COLOR_GRAY2BGR)
        
        navigation_path_topdown_np = [np.int32(np.vstack([
            agent_position_precise,
            navigation_path_inside_precise]))]
        navigation_path_precise_np = [np.int32(np.vstack([
            agent_position_precise,
            navigation_path_precise]))]
        
        for navigation_path_np, file_name_ext in zip([navigation_path_topdown_np, navigation_path_precise_np], ['topdown', 'precise']):
            navigation_path_vis = cv2.polylines(
                precise_map_vis,
                navigation_path_np,
                False,
                (0, 0, 255),
                agent_radius_pixel_precise_int)
            navigation_path_vis = cv2.circle(
                navigation_path_vis,
                np.int32(target_precise),
                agent_radius_pixel_precise_int,
                (0, 255, 0),
                -1)
            navigation_path_vis = cv2.circle(
                navigation_path_vis,
                np.int32(agent_position_precise),
                agent_radius_pixel_precise_int,
                (255, 0, 0),
                -1)
            cv2.imwrite(os.path.join(test_optimize_dir, current_time + f'_navigation_path_{file_name_ext}.png'), navigation_path_vis)
            
        navigation_path_optimized = translations_precise_to_topdown_(navigation_path_precise)
        return np.vstack([navigation_path_optimized, navigation_path[navigation_path_first_outside_index:]]), True
    else:
        return navigation_path, False
    
def optimize_navigation_path_using_precise_map_and_fast_forward(
    navigation_path_topdown:np.ndarray,
    precise_map:cv2.Mat,
    agent_position_topdown:np.ndarray,
    agent_radius_pixel_topdown:float,
    topdown_to_precise_ratio:float,
    precise_min_coord:np.ndarray,
    precise_max_coord:np.ndarray) -> np.ndarray:
    navigation_path_inside_condition = is_positions_in_precise_range(
        navigation_path_topdown,
        precise_min_coord,
        precise_max_coord)
    
    navigation_path_first_outside_index = np.argmin(navigation_path_inside_condition)
    if navigation_path_first_outside_index == 0:
        return navigation_path_topdown
    
    navigation_path_topdown_inside = navigation_path_topdown[:navigation_path_first_outside_index]
    navigation_path_topdown_outside = navigation_path_topdown[navigation_path_first_outside_index:]
    navigation_path_precise = translations_topdown_to_precise(
        navigation_path_topdown_inside,
        precise_min_coord,
        topdown_to_precise_ratio)
    agent_position_precise = translations_topdown_to_precise(
        agent_position_topdown,
        precise_min_coord,
        topdown_to_precise_ratio)
    agent_radius_pixel_precise = agent_radius_pixel_topdown * topdown_to_precise_ratio
    navigation_path_precise = optimize_navigation_path_using_fast_forward(
        navigation_path=navigation_path_precise,
        obstacle_map=precise_map,
        agent_position=agent_position_precise,
        agent_radius_pixel=agent_radius_pixel_precise)
    navigation_path_topdown_inside_optimized = translations_precise_to_topdown(
        navigation_path_precise,
        precise_min_coord,
        1 / topdown_to_precise_ratio)
    navigation_path_topdown_optimized = np.vstack([navigation_path_topdown_inside_optimized, navigation_path_topdown_outside])
    return navigation_path_topdown_optimized

def optimize_navigation_path_using_fast_forward(
    navigation_path:np.ndarray,
    obstacle_map:cv2.Mat,
    agent_position:np.ndarray,
    agent_radius_pixel:float) -> np.ndarray:
    free_space_pixels_num = cv2.countNonZero(obstacle_map)
    
    navigation_point_last_distance = np.inf
    for navigation_point_index, navigation_point in enumerate(navigation_path[::-1]):
        line_test_result = cv2.line(
            obstacle_map.copy(),
            np.int32(agent_position),
            np.int32(navigation_point),
            255,
            int(np.ceil(agent_radius_pixel * 3)))
        if cv2.countNonZero(line_test_result) != free_space_pixels_num:
            continue
        navigation_point_distance = np.linalg.norm(agent_position - navigation_point)
        if navigation_point_distance > navigation_point_last_distance:
            break
        navigation_point_last_distance = navigation_point_distance
        
    return navigation_path[-(navigation_point_index + 1):]

def check_path_using_topdown_and_precise_map(
    whole_path_topdown:np.ndarray,
    topdown_map:cv2.Mat,
    agent_radius_pixel_topdown:float,
    precise_map:cv2.Mat=None,
    precise_min_coord:np.ndarray=None,
    precise_max_coord:np.ndarray=None,
    topdown_to_precise_ratio:float=None) -> bool:
    if (precise_map is not None) and\
        (precise_min_coord is not None) and\
            (precise_max_coord is not None) and\
                (topdown_to_precise_ratio is not None):
        whole_path_inside_condition = is_positions_in_precise_range(
            whole_path_topdown,
            precise_min_coord,
            precise_max_coord)
        whole_path_first_outside_index = np.argmin(whole_path_inside_condition)
        if whole_path_first_outside_index != 0:
            whole_path_inside = whole_path_topdown[:whole_path_first_outside_index]
            whole_path_outside = whole_path_topdown[whole_path_first_outside_index:]
            whole_path_precise = translations_topdown_to_precise(
                whole_path_inside,
                precise_min_coord,
                topdown_to_precise_ratio)
            agent_radius_pixel_precise = agent_radius_pixel_topdown * topdown_to_precise_ratio
            free_space_pixels_num_precise = cv2.countNonZero(precise_map)
            agent_mask_precise = cv2.circle(
                np.zeros_like(precise_map),
                np.int32(whole_path_precise[0]),
                int(np.ceil(agent_radius_pixel_precise)),
                255,
                -1)
            line_test_result_precise = cv2.polylines(
                precise_map.copy(),
                [np.int32(whole_path_precise)],
                False,
                255,
                1)
            line_test_result_precise[agent_mask_precise > 0] = precise_map[agent_mask_precise > 0]
            if cv2.countNonZero(line_test_result_precise) != free_space_pixels_num_precise:
                rospy.logdebug('Precise map check fail.')
                return False
        else:
            whole_path_outside = whole_path_topdown
    else:
        whole_path_outside = whole_path_topdown
    free_space_pixels_num_topdown = cv2.countNonZero(topdown_map)
    agent_mask_topdown = cv2.circle(
        np.zeros_like(topdown_map),
        np.int32(whole_path_topdown[0]),
        int(np.ceil(agent_radius_pixel_topdown)),
        255,
        -1)
    line_test_result_topdown = cv2.polylines(
        topdown_map.copy(),
        [np.int32(whole_path_outside)],
        False,
        255,
        1)
    line_test_result_topdown[agent_mask_topdown > 0] = topdown_map[agent_mask_topdown > 0]
    check_result = cv2.countNonZero(line_test_result_topdown) == free_space_pixels_num_topdown
    if not check_result:
        rospy.logdebug('Topdown map check fail.')
    return check_result
    
def translations_topdown_to_precise(
    translations_topdown:np.ndarray,
    precise_origin:np.ndarray,
    topdown_to_precise_ratio:float) -> np.ndarray:
    translations_precise:np.ndarray = (translations_topdown.reshape(-1, 2) - precise_origin) * topdown_to_precise_ratio
    return translations_precise.reshape(translations_topdown.shape)

def translations_precise_to_topdown(
    translations_precise:np.ndarray,
    precise_origin:np.ndarray,
    precise_to_topdown_ratio:float) -> np.ndarray:
    translations_topdown:np.ndarray = (translations_precise.reshape(-1, 2) * precise_to_topdown_ratio) + precise_origin
    return translations_topdown.reshape(translations_precise.shape)
    
class TurnLineTestResult(Enum):
    BOTH_FREE_SPACE = 0
    LEFT_FREE_SPACE = 1
    RIGHT_FREE_SPACE = -1
    LEFT_MORE_FREE_SPACE = 2
    RIGHT_MORE_FREE_SPACE = -2
    RIGHT_TRY_FAILED = 3
    LEFT_TRY_FAILED = -3
    BOTH_FREE_SPACE_WITH_OBSTACLE = 4
    BOTH_TRY_FAILED = 5
    
def visualize_directions(
    directions:np.ndarray,
    figsize:int=101) -> cv2.Mat:
    assert figsize % 2 == 1, 'figsize should be odd.'
    figsize_half = figsize // 2
    directions_ = directions.reshape(-1, 2)
    directions_ = directions_ / np.linalg.norm(directions_, axis=1)[:, np.newaxis]
    black = cv2.cvtColor(np.zeros((figsize, figsize), dtype=np.uint8), cv2.COLOR_GRAY2BGR)
    directions_ = np.int32(directions_ * figsize_half) + figsize_half
    directions_colormap = cm.get_cmap('tab20')
    directions_number = len(directions_)
    for direction_id, direction in enumerate(directions_):
        direction_color = np.uint8(np.array(directions_colormap(direction_id / directions_number)[:3]) * 255)
        direction_color:np.ndarray = direction_color[::-1]
        cv2.line(black, (figsize_half, figsize_half), direction, direction_color.tolist(), 1)
    return black
    
def get_escape_plan(
    obstacle_map:cv2.Mat,
    agent_position:np.ndarray,
    agent_rotation_vector:np.ndarray,
    agent_turn_angle:float,
    agent_step_size_pixel:float,
    inaccessible_database:np.ndarray) -> Tuple[int, np.ndarray]:
    # FIXME: Fix the bug in escape plan and visualize it.
    
    # XXX: Remove after debug
    test_escape_dir = os.path.join(os.getcwd(), 'test', 'test_escape')
    os.makedirs(test_escape_dir, exist_ok=True)
    inaccessible_database_vis = visualize_directions(inaccessible_database)
    cv2.imwrite(os.path.join(test_escape_dir, time.strftime('%Y-%m-%d_%H-%M-%S') + '_escape_inaccessible_database.png'), inaccessible_database_vis)
    
    agent_turn_angle_rad = np.radians(agent_turn_angle)
    turn_times_half = int(np.ceil(180 / agent_turn_angle))
    turn_left_theta = (np.arange(turn_times_half) + 1) * agent_turn_angle_rad
    turn_right_theta = -turn_left_theta
    agent_rotation_theta = np.arctan2(agent_rotation_vector[1], agent_rotation_vector[0])
    turn_left_rotation_vectors = np.vstack((
        np.cos(turn_left_theta + agent_rotation_theta),
        np.sin(turn_left_theta + agent_rotation_theta))).T
    turn_right_rotation_vectors = np.vstack((
        np.cos(turn_right_theta + agent_rotation_theta),
        np.sin(turn_right_theta + agent_rotation_theta))).T
    assert np.allclose(np.linalg.norm(turn_left_rotation_vectors, axis=1), 1), 'Turn left rotation vectors should be normalized.'
    assert np.allclose(np.linalg.norm(turn_right_rotation_vectors, axis=1), 1), 'Turn right rotation vectors should be normalized.'
    free_space_pixels_num = cv2.countNonZero(obstacle_map)
    if len(inaccessible_database) > 0:
        turn_left_rotation_vectors_inaccessible = cdist(turn_left_rotation_vectors, inaccessible_database)
        turn_right_rotation_vectors_inaccessible = cdist(turn_right_rotation_vectors, inaccessible_database)
        turn_left_rotation_vectors_inaccessible = np.any(turn_left_rotation_vectors_inaccessible < agent_turn_angle_rad * 0.1, axis=1)
        turn_right_rotation_vectors_inaccessible = np.any(turn_right_rotation_vectors_inaccessible < agent_turn_angle_rad * 0.1, axis=1)
    else:
        turn_left_rotation_vectors_inaccessible = np.zeros(turn_times_half, dtype=bool)
        turn_right_rotation_vectors_inaccessible = np.zeros(turn_times_half, dtype=bool)
    line_test_results = []
    for turn_left_rotation_vector, turn_left_rotation_vector_inaccessible, turn_right_rotation_vector, turn_right_rotation_vector_inaccessible in zip(turn_left_rotation_vectors, turn_left_rotation_vectors_inaccessible, turn_right_rotation_vectors, turn_right_rotation_vectors_inaccessible):
        if turn_left_rotation_vector_inaccessible:
            turn_left_free_space_pixels_num = np.inf
        else:
            turn_left_line_test_result = cv2.line(
                obstacle_map.copy(),
                np.int32(agent_position),
                np.int32(agent_position + turn_left_rotation_vector * agent_step_size_pixel),
                255,
                1)
            turn_left_free_space_pixels_num = cv2.countNonZero(turn_left_line_test_result)
        if turn_right_rotation_vector_inaccessible:
            turn_right_free_space_pixels_num = np.inf
        else:
            turn_right_line_test_result = cv2.line(
                obstacle_map.copy(),
                np.int32(agent_position),
                np.int32(agent_position + turn_right_rotation_vector * agent_step_size_pixel),
                255,
                1)
            turn_right_free_space_pixels_num = cv2.countNonZero(turn_right_line_test_result)
        assert turn_left_free_space_pixels_num >= free_space_pixels_num, 'Turn left line test result should have more free space pixels than the obstacle map.'
        assert turn_right_free_space_pixels_num >= free_space_pixels_num, 'Turn right line test result should have more free space pixels than the obstacle map.'
        if turn_left_free_space_pixels_num == free_space_pixels_num == turn_right_free_space_pixels_num:
            line_test_results.append(TurnLineTestResult.BOTH_FREE_SPACE.value)
        elif turn_left_free_space_pixels_num == free_space_pixels_num:
            line_test_results.append(TurnLineTestResult.LEFT_FREE_SPACE.value)
        elif turn_right_free_space_pixels_num == free_space_pixels_num:
            line_test_results.append(TurnLineTestResult.RIGHT_FREE_SPACE.value)
        elif turn_left_free_space_pixels_num == turn_right_free_space_pixels_num == np.inf:
            line_test_results.append(TurnLineTestResult.BOTH_TRY_FAILED.value)
        elif turn_right_free_space_pixels_num == np.inf:
            line_test_results.append(TurnLineTestResult.RIGHT_TRY_FAILED.value)
        elif turn_left_free_space_pixels_num == np.inf:
            line_test_results.append(TurnLineTestResult.LEFT_TRY_FAILED.value)
        elif turn_left_free_space_pixels_num - free_space_pixels_num < turn_right_free_space_pixels_num - free_space_pixels_num:
            line_test_results.append(TurnLineTestResult.LEFT_MORE_FREE_SPACE.value)
        elif turn_left_free_space_pixels_num - free_space_pixels_num > turn_right_free_space_pixels_num - free_space_pixels_num:
            line_test_results.append(TurnLineTestResult.RIGHT_MORE_FREE_SPACE.value)
        elif turn_left_free_space_pixels_num == turn_right_free_space_pixels_num:
            line_test_results.append(TurnLineTestResult.BOTH_FREE_SPACE_WITH_OBSTACLE.value)
        else:
            raise ValueError('Invalid turn line test result.')
    line_test_results = np.array(line_test_results)
    line_test_results_abs = np.abs(line_test_results)
    if 1 in line_test_results_abs:
        indices = np.argwhere(line_test_results_abs == 1)
        first_index = indices[0, 0]
        rotation_direction = line_test_results[first_index]
    else:
        line_test_results_condition = np.logical_or(
            line_test_results_abs == TurnLineTestResult.BOTH_TRY_FAILED.value,
            line_test_results_abs == TurnLineTestResult.BOTH_FREE_SPACE_WITH_OBSTACLE.value)
        line_test_results[line_test_results_condition] = 0
        rotation_direction = np.sign(np.sum(line_test_results))
        if rotation_direction == 0:
            rotation_direction = np.random.choice([-1, 1])
        
    turn_times = int(np.ceil(360 / agent_turn_angle))
    turn_test_condition = np.zeros(turn_times, dtype=bool)
    if rotation_direction == TurnLineTestResult.LEFT_FREE_SPACE.value:
        turn_test_condition[:turn_times_half] = line_test_results != TurnLineTestResult.LEFT_TRY_FAILED.value
    elif rotation_direction == TurnLineTestResult.RIGHT_FREE_SPACE.value:
        turn_test_condition[:turn_times_half] = line_test_results != TurnLineTestResult.RIGHT_TRY_FAILED.value
    else:
        raise ValueError('Invalid rotation direction.')
    
    turn_test_condition_index_remain = np.arange(turn_times_half, turn_times)
    turn_theta_remain = (turn_test_condition_index_remain + 1) * agent_turn_angle_rad * rotation_direction
    turn_rotation_vectors_remain = np.vstack((
        np.cos(turn_theta_remain + agent_rotation_theta),
        np.sin(turn_theta_remain + agent_rotation_theta))).T
    if len(inaccessible_database) > 0:
        turn_rotation_vectors_remain_inaccessible = cdist(turn_rotation_vectors_remain, inaccessible_database)
        turn_rotation_vectors_remain_inaccessible = np.any(turn_rotation_vectors_remain_inaccessible < agent_turn_angle_rad * 0.1, axis=1)
    else:
        turn_rotation_vectors_remain_inaccessible = np.zeros(turn_times_half, dtype=bool)
    for turn_test_condition_index, turn_rotation_vector_inaccessible in zip(turn_test_condition_index_remain, turn_rotation_vectors_remain_inaccessible):
        if turn_rotation_vector_inaccessible:
            continue
        else:
            turn_test_condition[turn_test_condition_index] = True
    assert np.any(turn_test_condition), 'No valid turn test condition.'
    # XXX: Remove after debug
    turn_theta = (np.arange(len(turn_test_condition)) + 1) * agent_turn_angle_rad * rotation_direction
    turn_rotation_vectors = np.vstack((
        np.cos(turn_theta + agent_rotation_theta),
        np.sin(turn_theta + agent_rotation_theta))).T
    turn_rotation_vectors = turn_rotation_vectors[turn_test_condition]
    turn_test_condition_vis = visualize_directions(turn_rotation_vectors)
    cv2.imwrite(os.path.join(test_escape_dir, time.strftime('%Y-%m-%d_%H-%M-%S') + '_escape_turn_test_condition.png'), turn_test_condition_vis)
    return rotation_direction, turn_test_condition
    