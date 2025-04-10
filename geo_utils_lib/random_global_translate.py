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

def calculate_deformation_params(
    target_ED: Optional[float], 
    target_EC: Optional[float],
    threshold: float,
    total_instances: int) -> Tuple[float, int]:
    if target_ED is not None:
        instances_to_deform = int(total_instances * target_ED)
        deform_strength = np.random.normal(loc=1.5*threshold, scale=threshold/6)
    elif target_EC is not None:
        instances_to_deform = total_instances
        deform_strength = threshold * target_EC
    else:
        raise ValueError("At least one of target_ED or target_EC must be provided")
        
    return deform_strength, instances_to_deform

def random_global_translate(
    map_data: Dict[str, List[List]], 
    target_ED: Optional[float] = None,
    target_EC: Optional[float] = None,
    distance_level: str = 'high') -> Dict[str, List[np.ndarray]]:
    distance_thresholds = {
        'high': 1.5,
        'medium': 1.0,
        'low': 0.5
    }
    
    elements = _convert_to_elements(map_data)
    threshold = distance_thresholds[distance_level]
    total_instances = len(elements)
    
    translation_distance, instances_to_deform = calculate_deformation_params(
        target_ED, target_EC, threshold, total_instances
    )
    
    deform_indices = np.random.choice(
        total_instances, 
        size=instances_to_deform,
        replace=False
    )
    
    deformed_elements = []
    for i, element in enumerate(elements):
        if i in deform_indices:
            translation = np.random.normal(loc=translation_distance/4, scale=translation_distance/4, size=element.points.shape)
            deformed_points = element.points + translation
        else:
            deformed_points = element.points.copy()
            
        deformed_elements.append(MapElement(
            points=deformed_points,
            element_type=element.element_type,
            instance_id=element.instance_id
        ))
    
    return _convert_to_dict(deformed_elements) 