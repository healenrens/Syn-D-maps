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


def find_approximate_inscribed_rectangle(polygon_points, direction_vector, num_points_per_side=5):
    normal_vector = np.array([-direction_vector[1], direction_vector[0]])

    proj_direction = np.dot(polygon_points, direction_vector)
    proj_normal = np.dot(polygon_points, normal_vector)

    min_dir, max_dir = np.min(proj_direction), np.max(proj_direction)
    min_norm, max_norm = np.min(proj_normal), np.max(proj_normal)

    shrink_factor = 0.1  
    dir_range = max_dir - min_dir
    norm_range = max_norm - min_norm

    min_dir += dir_range
    max_dir -= dir_range
    min_norm += norm_range * shrink_factor
    max_norm -= norm_range * shrink_factor

    base_corners = np.array([
        direction_vector * min_dir + normal_vector * min_norm,  
        direction_vector * max_dir + normal_vector * min_norm,  
        direction_vector * max_dir + normal_vector * max_norm,  
        direction_vector * min_dir + normal_vector * max_norm, 
        direction_vector * min_dir + normal_vector * min_norm,  
    ])

    dense_points = []
    for i in range(4): 
        start_point = base_corners[i]
        end_point = base_corners[i + 1]

       
        t = np.linspace(0, 1, num_points_per_side)
        edge_points = np.array([start_point + (end_point - start_point) * ti for ti in t])

     
        dense_points.extend(edge_points[:-1])

    dense_points.append(base_corners[-1])

    return np.array(dense_points)


def average_direction(dir_a: np.ndarray, dir_b: np.ndarray) -> np.ndarray:
    dir_a_norm = dir_a / np.linalg.norm(dir_a)
    dir_b_norm = dir_b / np.linalg.norm(dir_b)
    avg_dir = (dir_a_norm + dir_b_norm) / 2.0
    avg_dir_norm = avg_dir / np.linalg.norm(avg_dir)
    return avg_dir_norm


def scale_lane_to_width(lane_points: np.ndarray, target_width: float,
                        reference_center: np.ndarray = None) -> np.ndarray:
    if len(lane_points) < 2:
        return lane_points
    direction = calculate_lane_direction(lane_points)
    if np.linalg.norm(direction) < 1e-6:
        return lane_points
    normal = np.array([-direction[1], direction[0]])
    center = np.mean(lane_points, axis=0) if reference_center is None else reference_center
    projections = [np.dot(point - center, normal) for point in lane_points]
    current_width = max(projections) - min(projections)
    if current_width < 1e-6:
        return lane_points
    scale_factor = target_width / current_width
    scaled_lane = []
    for point in lane_points:
        offset = np.dot(point - center, normal)
        scaled_offset = offset * scale_factor
        scaled_point = center + scaled_offset * normal + np.dot(point - center, direction) * direction
        scaled_lane.append(scaled_point)
    return np.array(scaled_lane)


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

    return combined_direction  # / np.linalg.norm(combined_direction)


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


def calculate_center_line(points: np.ndarray, num_points: int) -> np.ndarray:
    sorted_indices = np.argsort(points[:, 0])
    sorted_points = points[sorted_indices]
    x_unique = np.linspace(sorted_points[0, 0], sorted_points[-1, 0], num_points)
    y_avg = np.interp(x_unique, sorted_points[:, 0], sorted_points[:, 1])
    center_line = np.column_stack((x_unique, y_avg))
    return center_line


def least_squares_fit(points, m):

    x = points[:, 0]
    y = points[:, 1]

    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)

    slope = numerator / denominator
    intercept = y_mean - slope * x_mean

    x_fit = np.linspace(np.min(x), np.max(x), m)

    y_fit = slope * x_fit + intercept

    fitted_points = np.column_stack((x_fit, y_fit))

    return fitted_points


def group_lanes_and_lines(lanes_with_tokens: List[Dict[str, Union[str, np.ndarray]]],
                          lines_with_tokens: Dict[str, List[Dict[str, Union[str, np.ndarray]]]],
                          distance_threshold: float = config.group_distance_theshold) -> List[
    Dict[str, Union[List[Dict[str, Union[str, np.ndarray]]], int]]]:  # 6for delete lane and other 10 for add
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



def find_center_line(group: Dict) -> Tuple[str, np.ndarray]:
    lanes = group.get('lane', [])
    if not lanes:
        return 'none', None

    lane_centers = []
    for lane in lanes:
        points = lane['points']  # (n, 2) array

        center = np.mean(points, axis=0)
        lane_centers.append(center)

    mean_center = np.mean(lane_centers, axis=0)

    divider_types = ['lane_divider', 'road_divider']
    min_distance = float('inf')
    best_divider_type = 'none'
    best_divider_points = None

    for divider_type in divider_types:
        dividers = group.get(divider_type, [])
        for divider in dividers:
            points = divider['points']  # (n, 2) array
            divider_center = np.mean(points, axis=0)
            distance = np.linalg.norm(divider_center - mean_center)
            if distance < min_distance:
                min_distance = distance
                best_divider_type = divider_type
                best_divider_points = points

    return best_divider_type, best_divider_points


def segment_polygon_overlap(segment_points, polygon_points):
    from shapely.geometry import LineString, Polygon, Point
    try:
        line = LineString(segment_points)
        polygon = Polygon(polygon_points)
    except:
        return False

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
    return False


def create_lane_with_shape(reference_lane_points: np.ndarray, reference_divider_direction: np.array, offset: float,
                           width: float) -> np.ndarray:
    if len(reference_lane_points) < 2:
        return np.array([])
    direction = calculate_lane_direction(reference_lane_points)
    divider_direction = reference_divider_direction
    approximate_rectangle = find_approximate_inscribed_rectangle(reference_lane_points, divider_direction)
    if np.linalg.norm(direction) < 1e-6:
        return np.array([])
    normal = np.array([-direction[1], direction[0]])
    new_lane_points = []
    for point in reference_lane_points:
        new_point = point + offset * normal
        new_lane_points.append(new_point)
    new_lane_points = scale_lane_to_width(np.array(approximate_rectangle), target_width=width)
    return np.array(new_lane_points)


def generate_divider(points_a: np.ndarray, points_b: np.ndarray, direction: np.array,
                     num_points: int = 100, ) -> np.ndarray:
    dir_a = calculate_lane_direction(points_a)
    dir_b = calculate_lane_direction(points_b)

    avg_dir = average_direction(dir_a, dir_b)

    theta = np.arctan2(avg_dir[1], avg_dir[0])

    rotated_points_a = rotate_points(points_a, -theta)
    rotated_points_b = rotate_points(points_b, -theta)

    center_line_a = calculate_center_line(rotated_points_a, num_points)
    center_line_b = calculate_center_line(rotated_points_b, num_points)
    center_line_a = least_squares_fit(center_line_a, num_points)
    center_line_b = least_squares_fit(center_line_b, num_points)

    divider_rotated = (center_line_a + center_line_b) / 2.0

    divider = rotate_points(divider_rotated, theta)
    center = np.mean(divider, axis=0)
    relative_points = divider - center
    projection_lengths = np.dot(relative_points, direction)
    divider = center + np.outer(projection_lengths, direction)

    return divider


def push_elements_from_center(
        center_divider: np.ndarray,
        groups: List[Dict[str, List[Dict]]],
        boundary: List[Dict],
        ignore_token: List,
        push_distance: float
) -> None:
    processed_tokens = set(ignore_token)

    direction = calculate_lane_direction(center_divider)

    normal = np.array([-direction[1], direction[0]])
    total_change = 0
    point_on_line = center_divider[0]

    def is_intersecting_with_infinite_line(points: np.ndarray) -> bool:
        if len(points) < 2:
            return False

        vectors = points - point_on_line
        signed_distances = np.dot(vectors, np.array([-direction[1], direction[0]]))

        for i in range(len(signed_distances) - 1):
            if signed_distances[i] * signed_distances[i + 1] <= 0:
                return True
        return False

    def determine_side(points: np.ndarray) -> int:
        center = np.mean(points, axis=0)

        vector_to_point = center - point_on_line
        signed_distance = np.dot(vector_to_point, normal)

        return 1 if signed_distance > 0 else -1

    def push_points(points: np.ndarray, side: int) -> np.ndarray:
        return points + side * push_distance * normal

    for group in groups:
        for key, elements in group.items():
            if type(elements) != list or not elements:
                continue
            for element in elements:
                if (type(element) != dict or
                        element['token'] in processed_tokens):
                    continue
                points = element['points']
               
                side = determine_side(points)
                element['points'] = push_points(points, side)
                processed_tokens.add(element['token'])
                if key in ['lane_divider', 'road_divider']:
                    total_change += 1

    for element in boundary:
        if element['token'] in processed_tokens:
            continue

        points = element['points']
        if is_intersecting_with_infinite_line(points):
            continue

        side = determine_side(points)
        element['points'] = push_points(points, side)
        processed_tokens.add(element['token'])
        total_change += 1

    return groups, boundary, total_change


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
    lane_length = sum(calculate_lane_length(lane['points']) for lane in group['lane']) / len(group['lane'])
    change_number = 0
    if operation == "single_lane":
        if len(group['lane']) <= 1:
            modified_groups = lane_groups
            return modified_groups, road_boundary, False, change_number
        number_lane_divider = len(group['lane_divider'])
        number_road_divider = len(group['road_divider'])
        if number_lane_divider > 0:
            group['lane_divider'] = group['lane_divider'][:number_lane_divider - 1]
        else:
            if number_road_divider > 0:
                group['road_divider'] = group['road_divider'][:number_road_divider - 1]
            else:
                modified_groups = lane_groups
                return modified_groups, road_boundary, False, change_number
        lane_groups[group_index] = group
        modified_groups = lane_groups
        change_number = 1
    return modified_groups, road_boundary, True, change_number


def divider_delete(input_data_: Dict[str, List[List[List[float]]]], max_changes: float):
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
        index = 0
        for idx, divider_points in enumerate(input_data['road_boundary']):
            last_point = 0
            for sig_point in divider_points:
                sig_point = np.array(sig_point, dtype=np.float64)
                if type(last_point) == int:
                    last_point = sig_point
                    continue
                dividers_to_modify['road_boundary'].append({
                    'token': f'road_boundary_{index}',
                    'points': np.vstack([last_point, sig_point])
                })
                index += 1
                last_point = sig_point
        original_boundary = []
        for idx, divider_points in enumerate(input_data['road_boundary']):
            original_boundary.append({
                'token': f'road_boundary_{idx}',
                'points': np.array(divider_points)
            })
            total_instance += 1
    lane_groups = group_lanes_and_lines(lanes_to_modify, dividers_to_modify)

    road_boundary = dividers_to_modify['road_boundary']
    group_number = len(lane_groups)
    max_group_change_number = int(group_number * max_changes)

    modified_groups = copy.deepcopy(lane_groups)
    lane_groups = None
    group_change_number = 0
    change_instance_number = 0
    group_area = []
    for group in modified_groups:
        _group_area = 0
        for lane in group['lane']:
            this_group_area = calculate_polygon_area(lane['points'])
            _group_area += this_group_area
        group_area.append(_group_area)
    rate_average = np.mean(group_area) * 0.3
    for index in range(group_number):
        if group_area[index] < rate_average or modified_groups[index]['lane_number'] < 2:
            continue
        gc.collect()
        group = copy.deepcopy(modified_groups[index])
        modified_groups, road_boundary, change_success, change_number = modify_lane_groups(modified_groups, group,
                                                                                           index, road_boundary,
                                                                                           original_boundary,
                                                                                           'single_lane')

        if change_success:

            group_change_number += 1
            change_instance_number += change_number
            if group_change_number >= int(max_group_change_number):
                break
    if total_instance > 0:
        total_change_rate = change_instance_number / total_instance
    else:
        total_change_rate = 0
    modified_lane_groups = modified_groups
    modified_lanes_dict = {
        'lane': [],
        'lane_divider': [],
        'road_divider': [],
        'road_boundary': [],
        'null_line': [],
        'ped_crossing': input_data.get('ped_crossing', []),
        'road_segment': input_data.get('road_segment', [])
    }

    for group in modified_lane_groups:
        if group != {}:
            for sig_line in group['lane']:
                modified_lanes_dict['lane'].append(sig_line['points'].tolist())
            for divider_info in group['lane_divider']:
                modified_lanes_dict['lane_divider'].append(divider_info['points'].tolist())
            for divider_info in group['road_divider']:
                modified_lanes_dict['road_divider'].append(divider_info['points'].tolist())
            if 'road_boundary' in group:
                for divider_info in group['road_boundary']:
                    modified_lanes_dict['road_boundary'].append(divider_info['points'].tolist())
            if 'null_line' in group:
                for divider_info in group['null_line']:
                    modified_lanes_dict['null_line'].append(divider_info['points'].tolist())

    for _boundary in road_boundary:
        modified_lanes_dict['road_boundary'].append(_boundary['points'].tolist())

    return modified_lanes_dict, {k: input_data[k] for k in input_data}, lane_groups, total_change_rate
