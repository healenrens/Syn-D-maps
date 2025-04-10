import copy
import gc
from typing import List, Dict, Tuple, Union

import numpy as np
from box import Box
from shapely.geometry import Polygon, Point
from sklearn.decomposition import PCA

config = {
    'group_distance_theshold': 10,
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


def find_and_replace_overlapping(lanes_with_tokens, lines_with_tokens, road_boundary_with_tokens, groups_to_delete):
    if not groups_to_delete:
        return lanes_with_tokens, lines_with_tokens, road_boundary_with_tokens

    non_deleted_lanes = []
    tokens_deleted = []

    deleted_lane_tokens = set()
    for group in groups_to_delete:
        for lane in group.get('lane', []):
            deleted_lane_tokens.add(lane['token'])
            tokens_deleted.append(lane['token'])

        for line_type in ['lane_divider', 'road_divider']:
            for line in group.get(line_type, []):
                tokens_deleted.append(line['token'])

    for lane in lanes_with_tokens:
        if lane['token'] not in deleted_lane_tokens:
            non_deleted_lanes.append(lane)

    updated_lines = {line_type: [] for line_type in lines_with_tokens}
    for line_type, lines in lines_with_tokens.items():
        for line in lines:
            if line['token'] not in tokens_deleted:
                updated_lines[line_type].append(line)

    updated_road_boundary = []
    for boundary in road_boundary_with_tokens:
        updated_road_boundary.append(boundary)

    return non_deleted_lanes, updated_lines, updated_road_boundary


def segment_polygon_overlap_restrict(segment_points, polygon_points):
    if len(segment_points) < 2 or len(polygon_points) < 3:
        return False

    from shapely.geometry import LineString, Polygon
    try:
        line = LineString(segment_points)
        polygon = Polygon(polygon_points)
    except:
        return False

    if polygon.contains(line):
        return True
    if line.intersects(polygon):
        return True

    return False


def modify_lane_groups(
        lane_groups: List[Dict[str, Union[str, np.ndarray]]],
        group_index: int,
        operation: str) -> Tuple[
    List[Dict[str, Union[str, np.ndarray]]], bool, int]:

    modified_groups = copy.deepcopy(lane_groups)
    group_to_delete = modified_groups[group_index]
    token_deleted = 0
    if operation == "delete_group":
        if len(group_to_delete.get('lane', [])) == 0:
            return lane_groups, False, 0

        token_deleted = len(group_to_delete.get('lane', [])) + len(group_to_delete.get('lane_divider', [])) + len(
            group_to_delete.get('road_divider', []))
        modified_groups[group_index] = {}

        return modified_groups, True, token_deleted
    else:
        return lane_groups, False, 0


def delete_group(input_data_: Dict[str, List[List[List[float]]]], max_changes: float):
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

    lane_groups = group_lanes_and_lines(lanes_to_modify, dividers_to_modify)
    road_boundary = dividers_to_modify['road_boundary']
    
    group_number = len(lane_groups)
    max_group_change_number = int(max_changes*group_number)
    
    modified_groups = copy.deepcopy(lane_groups)
    group_change_number = 0 
    
    token_deleted = 0
    change_instance_number = 0
    success_groups = []
    
    for index in range(group_number):
        if group_change_number >= max_group_change_number:
            break
        if len(modified_groups[index].get('lane', [])) <= 1:
            continue
            
        modified_groups, change_success, change_number = modify_lane_groups(modified_groups, index, "delete_group")
        
        if change_success:
            token_deleted += change_number
            success_groups.append(index)
            group_change_number += 1
            change_instance_number += change_number
    modified_lane_groups = modified_groups
    if total_instance > 0:
        total_change_rate = change_instance_number / total_instance
    else:
        total_change_rate = 0
    lanes_remain, lines_remain, road_boundary_remain = find_and_replace_overlapping(
        lanes_to_modify, dividers_to_modify, road_boundary, [modified_lane_groups[i] for i in success_groups])
    
    modified_lanes_dict = {
        'lane': [],
        'lane_divider': [],
        'road_divider': [],
        'road_boundary': [],
        'ped_crossing': input_data.get('ped_crossing', []),
        'road_segment': input_data.get('road_segment', [])
    }
    
    for lane in lanes_remain:
        modified_lanes_dict['lane'].append(lane['points'].tolist())
    
    for line_type in ['lane_divider', 'road_divider']:
        for line in lines_remain[line_type]:
            modified_lanes_dict[line_type].append(line['points'].tolist())
    
    for boundary in road_boundary_remain:
        modified_lanes_dict['road_boundary'].append(boundary['points'].tolist())
    
    return modified_lanes_dict, {k: input_data[k] for k in input_data}, lane_groups, total_change_rate
