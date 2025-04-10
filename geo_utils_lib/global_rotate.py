import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import random

@dataclass
class MapElement:
    points: np.ndarray
    element_type: str
    instance_id: str

def _convert_to_elements(map_data: Dict[str, List[List[Tuple[float, float]]]]) -> List[MapElement]:
    elements = []
    for element_type, instances in map_data.items():
        for i, points in enumerate(instances):
            points_array = np.array(points)
            elements.append(MapElement(
                points=points_array,
                element_type=element_type,
                instance_id=f"{element_type}_{i}"
            ))
    return elements

def _convert_to_dict(elements: List[MapElement]) -> Dict[str, List[np.ndarray]]:
    result = {}
    for element in elements:
        if element.element_type not in result:
            result[element.element_type] = []
        result[element.element_type].append(element.points)
    return result

def global_rotate(
    map_data: Dict[str, List[List]], 
    target_EC: Optional[float] = None,
    target_ED: Optional[float] = None,
    distance_level: str = 'high',
    center_point: Optional[np.ndarray] = np.array([0,0])) -> Dict[str, List[np.ndarray]]:
    distance_thresholds = {
        'high': 1.5,
        'medium': 1.0,
        'low': 0.5
    }
    
    elements = _convert_to_elements(map_data)
    threshold = distance_thresholds[distance_level]
    
    if center_point is None or (center_point == np.array([0,0])).all():
        all_points = np.concatenate([e.points for e in elements])
        center_point = np.mean(all_points, axis=0)
    
    left_angle = 0
    right_angle = np.pi/2
    target_rate = target_EC if target_EC is not None else target_ED
    best_angle = None
    
    for _ in range(1000):
        mid_angle = (left_angle + right_angle) / 2
        rotation_matrix = np.array([
            [np.cos(mid_angle), -np.sin(mid_angle)],
            [np.sin(mid_angle), np.cos(mid_angle)]
        ])
        
        if target_EC is not None:
            total_points = 0
            error_points = 0
            
            for element in elements:
                centered_points = element.points - center_point
                rotated_points = centered_points @ rotation_matrix.T + center_point
                
                distances = np.linalg.norm(rotated_points - element.points, axis=1)
                
                total_points += len(distances)
                error_points += np.sum(distances > threshold)
            
            current_rate = error_points / total_points
            
        else:
            error_instances = 0
            
            for element in elements:
                centered_points = element.points - center_point
                rotated_points = centered_points @ rotation_matrix.T + center_point
                
                distances = np.linalg.norm(rotated_points - element.points, axis=1)
                if np.any(distances > threshold):
                    error_instances += 1
            
            current_rate = error_instances / len(elements)
        
        if abs(current_rate - target_rate) < 1e-3:
            best_angle = mid_angle
            break
        elif current_rate < target_rate:
            left_angle = mid_angle
        else:
            right_angle = mid_angle
            
        best_angle = mid_angle
    
    final_angle = best_angle if np.random.random() < 0.5 else -best_angle
    
    final_rotation_matrix = np.array([
        [np.cos(final_angle), -np.sin(final_angle)],
        [np.sin(final_angle), np.cos(final_angle)]
    ])
    
    deformed_elements = []
    for element in elements:
        centered_points = element.points - center_point
        rotated_points = centered_points @ final_rotation_matrix.T + center_point
        
        deformed_elements.append(MapElement(
            points=rotated_points,
            element_type=element.element_type,
            instance_id=element.instance_id
        ))
    
    return _convert_to_dict(deformed_elements) 