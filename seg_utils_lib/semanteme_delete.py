import copy
import numpy as np
from typing import Dict, List, Tuple, Union
from shapely.geometry import LineString, Polygon, Point


def dict_find(w_dict, data_dict):
    for i in data_dict:
        if i['token'] == w_dict['token']:
            return True
    return False


def segment_polygon_overlap_restrict(segment_points, polygon_points):
    from shapely.geometry import LineString, Polygon, Point
    
    line = LineString(segment_points)
    polygon = Polygon(polygon_points)
    
    if polygon.contains(line):
        return True
    
    for point in segment_points:
        point_geom = Point(point)
        if point_geom.distance(polygon.boundary) < 0.1:
            return True                                            
    
    if line.intersects(polygon.boundary):
        intersection = line.intersection(polygon.boundary)
        if hasattr(intersection, 'length') and intersection.length > 0:
            return True
        
    boundary_lines = list(map(LineString, zip(polygon_points, polygon_points[1:] + [polygon_points[0]])))
    for boundary_line in boundary_lines:
        if line.distance(boundary_line) < 2:
            v1 = np.array(segment_points[1]) - np.array(segment_points[0])
            v2 = np.array(boundary_line.coords[1]) - np.array(boundary_line.coords[0])
            v1 = v1 / np.linalg.norm(v1)
            v2 = v2 / np.linalg.norm(v2)
            if abs(np.dot(v1, v2)) > 0.98: 
                return True
    return False


def find_overelapping_delete_boundary(boundaries, group):
    overlapping_boundary = []
    remain_boundary = []
    use_hash = list(range(len(boundaries)))
    
    for i, lane in enumerate(group['lane']):
        for i, boundary in enumerate(boundaries):
            if segment_polygon_overlap_restrict(boundary['points'], lane['points']):
                overlapping_boundary.append(copy.deepcopy(boundary))
                try:
                    use_hash.remove(i)
                except:
                    pass
                
    for i in use_hash:
        remain_boundary.append(boundaries[i])
        
    return overlapping_boundary, remain_boundary


def group_lanes_and_lines(lanes_with_tokens, lines_with_tokens, distance_threshold=10):
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

    return groups


def calculate_lane_direction(lane_points: np.ndarray) -> np.ndarray:
    from sklearn.decomposition import PCA
    
    if len(lane_points) < 2:
        return np.array([1.0, 0.0])

    pca = PCA(n_components=1)
    pca.fit(lane_points)
    direction = pca.components_[0]
    return direction / np.linalg.norm(direction)


def fix_direction_xf(direction):
    if direction[0] < 0:
        direction = -direction
    return direction


def semanteme_delete(input_data_: Dict[str, List[List[List[float]]]], max_changes: float):
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
    
    modified_groups = copy.deepcopy(lane_groups)
    
    np.random.seed(42)
    group_indices = np.random.permutation(len(lane_groups))[:min(int(max_changes*len(lane_groups)), len(lane_groups))]
    change_number = 0
    for group_index in group_indices:
        group = copy.deepcopy(modified_groups[group_index])
        
        this_group_boundary, remain_boundary = find_overelapping_delete_boundary(road_boundary, group)
        group['road_boundary'] = this_group_boundary
        
        if 'null_line' not in group:
            group['null_line'] = []
        
        group['null_line'] = group.get('lane_divider', []) + group.get('road_boundary', []) + group.get('road_divider', [])
        group['lane_divider'] = []
        group['road_boundary'] = []
        group['road_divider'] = []
        change_number += len(group['lane_divider'])+len(group['road_divider'])+2
        road_boundary = copy.deepcopy(remain_boundary)
        lane_groups[group_index] = group
        modified_groups = copy.deepcopy(lane_groups)
    
    modified_lanes_dict = {
        'lane': [],
        'lane_divider': [],
        'road_divider': [],
        'road_boundary': [],
        'ped_crossing': input_data.get('ped_crossing', []),
        'road_segment': input_data.get('road_segment', [])
    }
    
    if any('null_line' in group for group in modified_groups):
        modified_lanes_dict['null_line'] = []
    total_change_rate = change_number / total_instance
    for group in modified_groups:
        if 'lane' in group:
            for sig_line in group['lane']:
                modified_lanes_dict['lane'].append(sig_line['points'].tolist())
            
            for divider_info in group.get('lane_divider', []):
                modified_lanes_dict['lane_divider'].append(divider_info['points'].tolist())
            
            for divider_info in group.get('road_divider', []):
                modified_lanes_dict['road_divider'].append(divider_info['points'].tolist())
            
            if 'road_boundary' in group:
                for divider_info in group['road_boundary']:
                    modified_lanes_dict['road_boundary'].append(divider_info['points'].tolist())
            
            if 'null_line' in group and 'null_line' in modified_lanes_dict:
                for divider_info in group['null_line']:
                    modified_lanes_dict['null_line'].append(divider_info['points'].tolist())
    
    for _boundary in road_boundary:
        modified_lanes_dict['road_boundary'].append(_boundary['points'].tolist())
    
    return modified_lanes_dict, {k: input_data[k] for k in input_data}, lane_groups, total_change_rate