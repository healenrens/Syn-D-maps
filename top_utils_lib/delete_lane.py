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


def find_and_replace_overlapping(boundaries, group):

    def point_distance(a, b):
        return np.linalg.norm(a - b)

    if 'road_divider' not in group and 'lane_divider' not in group:
        return boundaries, {}
    segments_to_remove = []
    dividers_to_remove = []
    for i, lane in enumerate(group['lane']):
        overlapping_boundary = []
        for boundary in boundaries:
            if segment_polygon_overlap(boundary['points'], lane['points']):
                overlapping_boundary.append(boundary)

        if overlapping_boundary != []:
            if 'lane_divider' in group:
                for idx, divider in enumerate(group['lane_divider']):
                    if segment_polygon_overlap(divider['points'],lane['points']):
                        segments_to_remove.append(overlapping_boundary)
                        dividers_to_remove.append(divider)
                            
            
            if  'road_divider' in group:
                for idx, divider in enumerate(group['road_divider']):
                    if segment_polygon_overlap(divider['points'],lane['points']):
                        segments_to_remove.append(overlapping_boundary)
                        dividers_to_remove.append(divider)
                            
    new_boundaries = []
    assert len(dividers_to_remove) == len(segments_to_remove)
    if segments_to_remove == []:
        return boundaries,[]
    
    random_idx = random.randint(0,len(dividers_to_remove)-1)
    dividers_to_remove = dividers_to_remove[random_idx]
    segments_to_remove = segments_to_remove[random_idx]
    last_index = 0
    just_add = 0
    
    for i, boundary in enumerate(boundaries):
        if i != 0:
            last_boundary = boundaries[i-1]
        else:
            last_boundary = None
        
        if dict_find(boundary,segments_to_remove):
            if just_add == 0:
                if last_boundary != None:
                    if np.linalg.norm(last_boundary['points'][-1] - boundary['points'][0])>1.0:
                        pass
                    else:
                        if point_distance(boundaries[last_index]['points'][-1],dividers_to_remove['points'][0]) < point_distance(boundaries[last_index]['points'][-1],dividers_to_remove['points'][-1]):
                            new_boundaries.append({'token':'smooth_line1','points':np.array([boundaries[last_index]['points'][-1],dividers_to_remove['points'][0]])})
                        else:
                            new_boundaries.append({'token':'smooth_line1','points':np.array([boundaries[last_index]['points'][-1],dividers_to_remove['points'][-1]])})
                new_boundaries.append(copy.deepcopy(dividers_to_remove))
                just_add = 1
        else:
            last_index = i
            new_boundaries.append(copy.deepcopy(boundary))
            next_i = i+1
            if next_i < len(boundaries):
                next_boundary = boundaries[next_i]
                if dict_find(next_boundary,segments_to_remove):
                    if np.linalg.norm(boundary['points'][-1] - next_boundary['points'][0]) < 1.0:
                        if point_distance(boundary['points'][-1],dividers_to_remove['points'][0]) < point_distance(boundary['points'][-1],dividers_to_remove['points'][-1]):
                            new_boundaries.append({'token':'smooth_line2','points':np.array([boundary['points'][-1],dividers_to_remove['points'][0]])})
                        else:
                            new_boundaries.append({'token':'smooth_line2','points':np.array([boundary['points'][-1],dividers_to_remove['points'][-1]])})
                        

    return new_boundaries, dividers_to_remove

def segment_polygon_overlap(segment_points, polygon_points):
    from shapely.geometry import LineString, Polygon, Point
    line = LineString(segment_points)
    try:
        polygon = Polygon(polygon_points)

        if polygon.contains(line):
            return True
        
        for point in segment_points:
            point_geom = Point(point)
            if point_geom.distance(polygon.boundary) < 2:
                return True                                            
        
        if line.intersects(polygon.boundary):
            intersection = line.intersection(polygon.boundary)
            if hasattr(intersection, 'length') and intersection.length > 0:
                return True
    except:
        pass
        
    return False

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

    if len(group['lane']) <= 1:
        modified_groups = lane_groups
        return  modified_groups, road_boundary, False, 0

    if operation == "delete_lane":
        selected_lane = group['lane'][-1]
        group['lane'] = group['lane'][:-1]
        if len(group['lane']) <= 1:
            lane_groups[group_index] = group
            modified_groups = lane_groups
            return modified_groups, road_boundary, False, 0

        number_lane_divider = len(group['lane_divider'])
        number_road_divider = len(group['road_divider'])
        if number_lane_divider > 0:
            group['lane_divider'] = group['lane_divider'][:number_lane_divider-1]
        else:
            if number_road_divider > 0:
                group['road_divider'] = group['road_divider'][:number_road_divider-1]
            else:
                modified_groups = lane_groups
                return modified_groups, road_boundary, False, 0

        new_boundaries, replace_divider = find_and_replace_overlapping(road_boundary, group)
        if replace_divider != {}:
            road_boundary = new_boundaries
            selected_lane = {}

        lane_groups[group_index] = group
        modified_groups = lane_groups
        return modified_groups, road_boundary, True, 1
    else:
        return lane_groups, road_boundary, False, 0


def delete_lane(input_data_: Dict[str, List[List[List[float]]]], max_changes: float):
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
    
    modified_groups = copy.deepcopy(lane_groups)
    group_change_number = 0 
    total_change_number = 0
    for index in range(group_number):
        if group_change_number >= max_group_change_number:
            break
        group = copy.deepcopy(modified_groups[index])
        if "lane" not in group:
            continue
        if len(group["lane"]) <= 1:
            continue
        modified_groups, road_boundary, change_success, change_number = modify_lane_groups(modified_groups,group,index,road_boundary, original_boundary, "delete_lane")
        if change_success:
            group_change_number+=1
            total_change_number += change_number
    
    total_change_rate = total_change_number / total_instance


    modified_lane_groups = modified_groups
    
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
            for sig_line in group.get('lane',[]):
                modified_lanes_dict['lane'].append(sig_line['points'].tolist())
            
            for divider_info in group.get('lane_divider',[]):
                modified_lanes_dict['lane_divider'].append(divider_info['points'].tolist())
            
            for divider_info in group.get('road_divider',[]):
                modified_lanes_dict['road_divider'].append(divider_info['points'].tolist())
            
    for _boundary in road_boundary:
        modified_lanes_dict['road_boundary'].append(_boundary['points'].tolist())
    
    if len(modified_lanes_dict['lane_divider']) == 0 and len(modified_lanes_dict['road_divider']) == 0:
        raise RuntimeError("delete lane not success")
    
    return modified_lanes_dict, {k: input_data[k] for k in input_data}, lane_groups, total_change_rate
