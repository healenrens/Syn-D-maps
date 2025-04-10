"""
Example script demonstrating how to use NuChanger map transformation libraries
to modify road network and map data from the nuScenes dataset.
"""

from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
import numpy as np
import os
# Import utility for map geometry extraction
from utils.patch_gemo import get_map_geom
from utils.road_edge import new_get_boundary

# Import visualization utility
from utils.fig_show import fig_show

# Import map transformation modules
from top_utils_lib.add_lane import add_lane
from top_utils_lib.delete_lane import delete_lane
from seg_utils_lib.semanteme_change import semanteme_change
from seg_utils_lib.semanteme_delete import semanteme_delete
from top_utils_lib.bezier_warp import bezier_warp
from top_utils_lib.lane_narrowing import lane_narrowing
from top_utils_lib.lane_widening import lane_widening
from top_utils_lib.delete_group import delete_group
from top_utils_lib.divider_delete import divider_delete
from geo_utils_lib.global_rotate import global_rotate
from geo_utils_lib.global_translate import global_translate
from geo_utils_lib.random_global_translate import random_global_translate

def main():
    """
    Main function to demonstrate map transformation capabilities.
    Loads a nuScenes scene, extracts map data, and applies various transformations.
    """
    # Initialize nuScenes dataset
    print("Initializing nuScenes dataset...")
    nusc = NuScenes(version='v1.0-mini', dataroot='_dataset')
    
    # Select a specific scene and sample
    scene_idx = 7  # Use scene index 7 for demonstration
    scene = nusc.scene[scene_idx]
    
    # Use a specific sample token for demonstration
    sample_token = '5998b71b64c146769bde1d5430741381'
    sample = nusc.get('sample', sample_token)
    
    print(f"Processing scene {scene_idx}, sample {sample_token}")
    
    # Get LIDAR_TOP sample data
    sample_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    
    # Get map location for the scene
    scene_log = nusc.get('log', scene['log_token'])
    location = scene_log['location']
    print(f"Map location: {location}")
    
    # Initialize map
    nusc_map = NuScenesMap(dataroot='_dataset', map_name=location)
    map_explorer = NuScenesMapExplorer(nusc_map)
    
    # Get ego vehicle pose
    ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
    ego_position = ego_pose['translation'][:2]
    ego_rotation = Quaternion(ego_pose['rotation'])
    patch_angle = quaternion_yaw(ego_rotation)/np.pi*180
    
    # Define search box around ego position
    x, y = ego_position[0], ego_position[1]
    h = 60  # height of search box
    w = 120  # width of search box
    search_box = (x, y, h, w)
    
    # Extract map geometries within search box
    print("Extracting map geometries...")
    all_line = get_map_geom(search_box, map_explorer, nusc_map, patch_angle)
    _all_line = all_line.copy()  # Keep a copy of original data
    
    # Get road boundaries
    edge = new_get_boundary(map_explorer, search_box, patch_angle, [h, w])
    all_line.update(edge)
    
    # Select relevant map features
    all_line = {k: all_line[k] for k in ['lane_divider', 'road_boundary', 'road_divider', 'lane']}
    _all_line = all_line.copy()
    
    # Create output directory
    output_dir = "transformation_examples/"
    os.makedirs(output_dir, exist_ok=True)
    print("Applying map transformations...")
    # Apply and visualize different map transformations
    
    # 1. Lane Widening: Increases the width of lanes
    print("1. Applying lane widening...")
    transformed_map, _, _, _ = lane_widening(all_line, 0.4)
    fig_show(_all_line, transformed_map, f"{output_dir}lane_widening_example")
    
    # 2. Semanteme Delete: Removes semantic elements from the map
    print("2. Applying semanteme deletion...")
    transformed_map, _, _, _ = semanteme_delete(all_line, 0.4)
    fig_show(_all_line, transformed_map, f"{output_dir}semanteme_delete_example")
    
    # 3. Divider Delete: Removes dividers from the map
    print("3. Applying divider deletion...")
    transformed_map, _, _, _ = divider_delete(all_line, 0.4)
    fig_show(_all_line, transformed_map, f"{output_dir}divider_delete_example")
    
    # 4. Bezier Warp: Applies bezier curve deformation to map elements
    print("4. Applying bezier warping...")
    transformed_map, _, _, _ = bezier_warp(all_line, 0.4)
    fig_show(_all_line, transformed_map, f"{output_dir}bezier_warp_example")
    
    # 5. Lane Narrowing: Decreases the width of lanes
    print("5. Applying lane narrowing...")
    transformed_map, _, _, _ = lane_narrowing(all_line, 0.4)
    fig_show(_all_line, transformed_map, f"{output_dir}lane_narrowing_example")
    
    # 6. Delete Group: Removes groups of map elements
    print("6. Applying group deletion...")
    transformed_map, _, _, _ = delete_group(all_line, 0.4)
    fig_show(_all_line, transformed_map, f"{output_dir}delete_group_example")
    
    # 7. Add Lane: Adds new lanes to the map
    print("7. Applying lane addition...")
    transformed_map, _, _, _ = add_lane(all_line, 0.4)
    fig_show(_all_line, transformed_map, f"{output_dir}add_lane_example")
    
    # 8. Delete Lane: Removes lanes from the map
    print("8. Applying lane deletion...")
    transformed_map, _, _, _ = delete_lane(all_line, 0.4)
    fig_show(_all_line, transformed_map, f"{output_dir}delete_lane_example")
    
    # 9. Semanteme Change: Changes semantic properties of map elements
    print("9. Applying semanteme changes...")
    transformed_map, _, _, _ = semanteme_change(all_line, 0.4)
    fig_show(_all_line, transformed_map, f"{output_dir}semanteme_change_example")
    
    # 10. Random Global Translate: Randomly translates all map elements
    print("10. Applying random global translation...")
    transformed_map = random_global_translate(all_line, target_ED=0.8, distance_level='medium')
    fig_show(_all_line, transformed_map, f"{output_dir}random_global_translate_example")
    
    # 11. Global Rotate: Rotates all map elements
    print("11. Applying global rotation...")
    transformed_map = global_rotate(all_line, target_EC=0.65, distance_level='medium')
    fig_show(_all_line, transformed_map, f"{output_dir}global_rotate_example")
    
    # 12. Global Translate: Translates all map elements
    print("12. Applying global translation...")
    transformed_map = global_translate(all_line, distance_level='medium')
    fig_show(_all_line, transformed_map, f"{output_dir}global_translate_example")
    
    print("All transformations completed. Results saved to 'transformation_examples/' directory.")
    
if __name__ == "__main__":
    main() 