import copy
import gc
from typing import List, Dict, Tuple, Union

import numpy as np
from box import Box
from shapely.geometry import Polygon, Point
from sklearn.decomposition import PCA
import random

config = {
    'group_distance_theshold': 8,
    'lane_angle_threshold': np.pi / 6,
    'pca_weight': 0.5,
    'tri_weight': 0.5,
}
config = Box(config)


def dict_find(w_dict, data_dict):
    for i in data_dict:
        if i['token'] == w_dict['token']:
            return True
    return False


def average_direction(dir_a: np.ndarray, dir_b: np.ndarray) -> np.ndarray:
    dir_a_norm = dir_a / np.linalg.norm(dir_a)
    dir_b_norm = dir_b / np.linalg.norm(dir_b)
    avg_dir = (dir_a_norm + dir_b_norm) / 2.0
    avg_dir_norm = avg_dir / np.linalg.norm(avg_dir)
    return avg_dir_norm


def calculate_lane_width(lane_points: np.ndarray) -> float:
    if len(lane_points) < 2:
        return 0.0
    direction = calculate_lane_direction(lane_points)
    if np.linalg.norm(direction) < 1e-6:
        return 0.0
    normal = np.array([-direction[1], direction[0]])
    center = np.mean(lane_points, axis=0)
    projections = [np.dot(point - center, normal) for point in lane_points]
    width = max(projections) - min(projections)
    return width


def calculate_lane_length(lane_points: np.ndarray) -> float:
    if len(lane_points) < 2:
        return 0.0
    direction = calculate_lane_direction(lane_points)
    if np.linalg.norm(direction) < 1e-6:
        return 0.0
    center = np.mean(lane_points, axis=0)
    projections = [np.dot(point - center, direction) for point in lane_points]
    length = max(projections) - min(projections)
    return length


def fix_direction_xf(direction):
    if direction[0] < 0:
        direction = -direction
    return direction


def rotate_points(points: np.ndarray, theta: float) -> np.ndarray:
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])
    return points @ rotation_matrix.T


def calculate_lane_direction(lane_points: np.ndarray) -> np.ndarray:
    if len(lane_points) < 2:
        return np.array([1.0, 0.0])

    pca = PCA(n_components=1)
    pca.fit(lane_points)
    pca_direction = pca.components_[0]
    pca_direction = pca_direction / np.linalg.norm(pca_direction)

    if len(lane_points) <= 3:
        return pca_direction
    polygon = Polygon(lane_points)
    min_bbox = polygon.minimum_rotated_rectangle
    bbox_ext = np.array(min_bbox.exterior.coords[:-1])
    d1 = np.linalg.norm(bbox_ext[1] - bbox_ext[0])
    d2 = np.linalg.norm(bbox_ext[2] - bbox_ext[1])

    if d1 > d2:
        bbox_direction = bbox_ext[1] - bbox_ext[0]
    else:
        bbox_direction = bbox_ext[2] - bbox_ext[1]
    bbox_direction = bbox_direction / np.linalg.norm(bbox_direction)

    angle = np.arccos(np.clip(np.dot(pca_direction, bbox_direction), -1.0, 1.0))
    if angle < config.lane_angle_threshold:
        weighted_direction = config.pca_weight * pca_direction + config.tri_weight * bbox_direction
        combined_direction = weighted_direction / np.linalg.norm(weighted_direction)
    else:
        combined_direction = pca_direction

    return combined_direction


def calculate_polygon_area(points: np.ndarray) -> float:
    points = np.asarray(points)

    if points.shape[1] != 2:
        raise ValueError("input must be shape (n,2)")
    if not np.allclose(points[0], points[-1]):
        raise ValueError("polygon open")
    
    x = points[:, 0]
    y = points[:, 1]
    area = 0.5 * np.abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))

    return area


def group_lanes_and_lines(lanes_with_tokens: List[Dict[str, Union[str, np.ndarray]]],
                          lines_with_tokens: Dict[str, List[Dict[str, Union[str, np.ndarray]]]],
                          distance_threshold: float = config.group_distance_theshold) -> List[
    Dict[str, Union[List[Dict[str, Union[str, np.ndarray]]], int]]]:
    groups = []
    n_lanes = len(lanes_with_tokens)
    directions = []
    centers = []
    for lane in lanes_with_tokens:
        direction = calculate_lane_direction(lane['points'])
        direction = fix_direction_xf(direction)
        directions.append(direction)
        centers.append(np.mean(lane['points'], axis=0))

    directions = np.array(directions)
    centers = np.array(centers)

    parent = list(range(n_lanes))
    rank = [0] * n_lanes

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1

    for i in range(n_lanes):
        for j in range(i + 1, n_lanes):
            angle = np.arccos(np.clip(np.dot(directions[i], directions[j]), -1.0, 1.0))
            distance = np.linalg.norm(centers[i] - centers[j])
            if angle < np.pi / 12 and distance < distance_threshold:
                union(i, j)
    group_dict = {}
    for i in range(n_lanes):
        root = find(i)
        if root not in group_dict:
            group_dict[root] = {
                'lane': [],
                'lane_divider': [],
                'road_divider': [],
                'lane_number': 0,
                'direction': []
            }
        group_dict[root]['lane'].append(lanes_with_tokens[i])
        group_dict[root]['direction'].append(directions[i])
    for group_data in group_dict.values():
        group_data['c'] = len(group_data['lane'])
        group_data['direction'] = np.mean(np.array(group_data['direction']), axis=0)
        groups.append(group_data)
    for group in groups:
        group['lane_number'] = len(group['lane'])
    used_token = set()
    for group in groups:
        all_points = np.vstack([lane['points'] for lane in group['lane']])
        min_x, min_y = np.min(all_points, axis=0)
        max_x, max_y = np.max(all_points, axis=0)
        expand_x = (max_x - min_x) * 0.05
        expand_y = (max_y - min_y) * 0.05
        min_x -= expand_x
        min_y -= expand_y
        max_x += expand_x
        max_y += expand_y

        bounding_box = Polygon([(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y)])
        local_center_point = Point(np.array([0, 0]))
        for line_type in ['lane_divider', 'road_divider']:
            this_type_number = 0
            for line in lines_with_tokens[line_type]:
                line_center = np.mean(line['points'], axis=0)
                point = Point(line_center)
                if bounding_box.contains(point) and line['token'] not in used_token:
                    group[line_type].append(line)
                    used_token.add(line['token'])
                    this_type_number += 1
        if bounding_box.contains(local_center_point):
            group['is_center_in'] = True
        else:
            group['is_center_in'] = False
    remaining_lines = {
        'lane_divider': [line for line in lines_with_tokens['lane_divider'] if
                         line not in [item for group in groups for item in group['lane_divider']]],
        'road_divider': [line for line in lines_with_tokens['road_divider'] if
                         line not in [item for group in groups for item in group['road_divider']]]
    }

    for line_type in ['lane_divider', 'road_divider']:
        for line in remaining_lines[line_type]:
            pca_line = PCA(n_components=1)
            pca_line.fit(line['points'])
            direction_line = pca_line.components_[0]
            best_group = None
            best_similarity = -1
            for group in groups:
                all_points = np.vstack([lane['points'] for lane in group['lane']])
                pca_group = PCA(n_components=1)
                pca_group.fit(all_points)
                direction_group = pca_group.components_[0]
                similarity = np.dot(direction_line, direction_group)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_group = group
            best_group[line_type].append(line)

    return groups


def scale_lane_to_width(lane_points: np.ndarray, target_width: float, reference_center: np.ndarray = None) -> np.ndarray:
    current_width = calculate_lane_width(lane_points)
    scale_ratio = target_width / current_width if current_width > 0 else 1.0
    if reference_center is None:
        reference_center = np.mean(lane_points, axis=0)
    
    direction = calculate_lane_direction(lane_points)
    if np.linalg.norm(direction) < 1e-6:
        return lane_points
    
    normal = np.array([-direction[1], direction[0]])
    transformed_points = []
    for point in lane_points:
        delta_longitudinal = np.dot(point - reference_center, direction)
        delta_transverse = np.dot(point - reference_center, normal)
        
        new_delta_transverse = delta_transverse * scale_ratio
        
        new_point = reference_center + delta_longitudinal * direction + new_delta_transverse * normal
        transformed_points.append(new_point)
    
    return np.array(transformed_points)


def create_lane_with_shape(reference_lane_points: np.ndarray, offset: float, width: float) -> np.ndarray:
    if len(reference_lane_points) < 2:
        return np.array([])
    
    direction = calculate_lane_direction(reference_lane_points)
    if np.linalg.norm(direction) < 1e-6:
        return np.array([])
    
    normal = np.array([-direction[1], direction[0]])
    
    center = np.mean(reference_lane_points, axis=0)
    new_center = center + offset * normal
    
    aligned_points = []
    for point in reference_lane_points:
        delta = point - center
        aligned_point = new_center + delta
        aligned_points.append(aligned_point)
    
    aligned_points = np.array(aligned_points)
    return scale_lane_to_width(aligned_points, width, new_center)


def generate_lane_divider(lane_a_points: np.ndarray, lane_b_points: np.ndarray) -> np.ndarray:
    if len(lane_a_points) < 2 or len(lane_b_points) < 2:
        return np.array([])
    
    dir_a = calculate_lane_direction(lane_a_points)
    dir_b = calculate_lane_direction(lane_b_points)
    
    if np.linalg.norm(dir_a) < 1e-6 or np.linalg.norm(dir_b) < 1e-6:
        return np.array([])
    
    center_a = np.mean(lane_a_points, axis=0)
    center_b = np.mean(lane_b_points, axis=0)
    midpoint = (center_a + center_b) / 2.0
    
    avg_dir = average_direction(dir_a, dir_b)
    norm_dir = np.array([-avg_dir[1], avg_dir[0]])
    
    len_a = calculate_lane_length(lane_a_points)
    len_b = calculate_lane_length(lane_b_points)
    avg_len = (len_a + len_b) / 2.0
    
    proj_a = [np.dot(p - center_a, dir_a) for p in lane_a_points]
    proj_b = [np.dot(p - center_b, dir_b) for p in lane_b_points]
    
    min_proj_a, max_proj_a = min(proj_a), max(proj_a)
    min_proj_b, max_proj_b = min(proj_b), max(proj_b)
    
    min_proj = (min_proj_a + min_proj_b) / 2.0
    max_proj = (max_proj_a + max_proj_b) / 2.0
    
    line_start = midpoint + min_proj * avg_dir
    line_end = midpoint + max_proj * avg_dir
    
    return np.array([line_start, line_end])


def modify_lane_groups(
        lane_groups: List[Dict[str, Union[str, np.ndarray]]],
        group: Dict[str, Union[str, np.ndarray]],
        group_index: int,
        road_boundary: Dict[str, Union[str, np.ndarray]],
        original_boundary: Dict[str, Union[str, np.ndarray]],
        operation: str,
        scale_factor: float = 0.5) -> Tuple[
    List[Dict[str, Union[str, np.ndarray]]], List[Dict[str, Union[str, np.ndarray]]]]:
    modified_groups = []
    new_lanes = []
    total_width = sum(calculate_lane_width(lane['points']) for lane in group['lane'])

    if operation == "add_lane":
        if len(group['lane']) <= 0:
            modified_groups = lane_groups
            return modified_groups, road_boundary, False, 0
        
        group_direction = np.mean([calculate_lane_direction(lane['points']) for lane in group['lane']], axis=0)
        group_direction = group_direction / np.linalg.norm(group_direction)
        
        total_change = 0
        
        lane_width = total_width / len(group['lane'])
        lane_normal = np.array([-group_direction[1], group_direction[0]])
        
        reference_center = np.mean([np.mean(lane['points'], axis=0) for lane in group['lane']], axis=0)
        
        if len(group['lane']) >= 1:
            reference_lane = group['lane'][-1]
            new_lane_points = create_lane_with_shape(reference_lane['points'], lane_width, lane_width)
            
            if len(new_lane_points) > 0:
                new_lane = {'token': f'new_lane_{random.randint(0, 9999)}', 'points': new_lane_points}
                
                if 'lane_divider' in group and len(group['lane_divider']) > 0:
                    divider_points = generate_lane_divider(reference_lane['points'], new_lane['points'])
                    if len(divider_points) > 0:
                        new_divider = {'token': f'new_divider_{random.randint(0, 9999)}', 'points': divider_points}
                        group['lane_divider'].append(new_divider)
                        total_change += 1
                
                total_change += 1
                new_lanes.append(new_lane)
        
        if total_change == 0:
            modified_groups = lane_groups
            return modified_groups, road_boundary, False, 0
        
        groups = lane_groups
        groups[group_index] = group
        groups[group_index]['lane'].extend(new_lanes)
        modified_groups = copy.deepcopy(groups)
        groups = None
        change_number = 2 + total_change
        return modified_groups, road_boundary, True, change_number
    else:
        return lane_groups, road_boundary, False, 0


def add_lane(input_data_: Dict[str, List[List[List[float]]]], max_changes: float):
    input_data = copy.deepcopy(input_data_)
    lanes_to_modify = []
    dividers_to_modify = {}
    total_instance = 0
    if 'lane' in input_data:
        for idx, lane_points in enumerate(input_data['lane']):
            lanes_to_modify.append({
                'token': f'lane_{idx}',
                'points': np.array(lane_points)
            })
            total_instance += 1

    if 'lane_divider' in input_data:
        dividers_to_modify['lane_divider'] = []
        for idx, divider_points in enumerate(input_data['lane_divider']):
            dividers_to_modify['lane_divider'].append({
                'token': f'lane_divider_{idx}',
                'points': np.array(divider_points)
            })
            total_instance += 1
    
    if 'road_divider' in input_data:
        dividers_to_modify['road_divider'] = []
        for idx, divider_points in enumerate(input_data['road_divider']):
            dividers_to_modify['road_divider'].append({
                'token': f'road_divider_{idx}',
                'points': np.array(divider_points)
            })
            total_instance += 1
    
    if 'road_boundary' in input_data:
        dividers_to_modify['road_boundary'] = []
        for idx, divider_points in enumerate(input_data['road_boundary']):
            dividers_to_modify['road_boundary'].append({
                'token': f'road_boundary_{idx}',
                'points': np.array(divider_points)
            })
            total_instance += 1
    
    original_lanes_ = copy.deepcopy(lanes_to_modify)
    original_dividers_ = copy.deepcopy(dividers_to_modify)
    
    lane_groups = group_lanes_and_lines(lanes_to_modify, dividers_to_modify)
    road_boundary = dividers_to_modify['road_boundary']
    original_boundary = copy.deepcopy(road_boundary)
    
    group_number = len(lane_groups)
    max_group_change_number = int(max_changes*group_number)
    change_instance_number = 0
    modified_groups = copy.deepcopy(lane_groups)
    group_change_number = 0 
    for index in range(group_number):
        if group_change_number >= max_group_change_number:
            break
        group = copy.deepcopy(modified_groups[index])
        modified_groups, road_boundary, change_success, change_number = modify_lane_groups(modified_groups, group, index, road_boundary, original_boundary, "add_lane")
        if change_success:
            group_change_number += 1
            change_instance_number += change_number
    
    modified_lane_groups = modified_groups
    if total_instance > 0:
        total_change_rate = change_instance_number / total_instance
    else:
        total_change_rate = 0
    modified_lanes_dict = {
        'lane': [],
        'lane_divider': [],
        'road_divider': [],
        'road_boundary': [],
        'ped_crossing': input_data.get('ped_crossing', []),
        'road_segment': input_data.get('road_segment', [])
    }
    
    for group in modified_lane_groups:
        if group != {}:
            for sig_line in group.get('lane', []):
                modified_lanes_dict['lane'].append(sig_line['points'].tolist())
            
            for divider_info in group.get('lane_divider', []):
                modified_lanes_dict['lane_divider'].append(divider_info['points'].tolist())
            
            for divider_info in group.get('road_divider', []):
                modified_lanes_dict['road_divider'].append(divider_info['points'].tolist())
            
    for _boundary in road_boundary:
        modified_lanes_dict['road_boundary'].append(_boundary['points'].tolist())
    
    return modified_lanes_dict, {k: input_data[k] for k in input_data}, lane_groups, total_change_rate
