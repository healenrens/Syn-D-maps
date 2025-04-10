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

def global_translate(
    map_data: Dict[str, List[List]], 
    distance_level: str = 'high') -> Dict[str, List[np.ndarray]]:
    distance_thresholds = {
        'high': 1.5,
        'medium': 1.0,
        'low': 0.5
    }
    
    elements = _convert_to_elements(map_data)
    
    translation_distance = distance_thresholds[distance_level] + random.uniform(0, 0.5)
    
    angle = np.random.uniform(0, 2 * np.pi)
    translation = translation_distance * np.array([np.cos(angle), np.sin(angle)])
    
    deformed_elements = []
    for element in elements:
        deformed_points = element.points + translation
        deformed_elements.append(MapElement(
            points=deformed_points,
            element_type=element.element_type,
            instance_id=element.instance_id
        ))
    
    return _convert_to_dict(deformed_elements) 