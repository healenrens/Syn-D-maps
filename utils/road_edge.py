from shapely.geometry import Polygon, LineString, MultiLineString, Point, box,MultiPolygon
import shapely.ops as ops
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import numpy as np
def create_boundary_linestring(x_center, y_center,height,width):
    x_min = x_center - width / 2
    x_max = x_center + width / 2
    y_min = y_center - height / 2
    y_max = y_center + height / 2
    boundary_linestring = LineString([
        (x_min, y_min),
        (x_max, y_min),
        (x_max, y_max),
        (x_min, y_max),
        (x_min, y_min)
    ])
    return boundary_linestring

def extract_line_strings(geometry):
    line_strings = []
    if geometry.geom_type == 'LineString':
        line_strings.append(geometry)
    elif geometry.geom_type == 'MultiLineString':
        line_strings.extend(list(geometry.geoms))
    elif geometry.geom_type == 'GeometryCollection':
        for geom in geometry.geoms:
            line_strings.extend(extract_line_strings(geom))
    return line_strings
def process_boundary(boundary):
    boundary_segments = []
    if isinstance(boundary, LineString):
        boundary = MultiLineString([boundary])
    for line in boundary.geoms:
        coords = list(line.coords)
        for i in range(len(coords) - 1):
            segment = LineString([coords[i], coords[i + 1]])
            boundary_segments.append(segment)
    
    return boundary_segments

def extract_road_boundary(lanes, road_segments,width, height,x_center=0, y_center=0):

    boundary_linestring = create_boundary_linestring(x_center, y_center, width, height)
    

    lane_polys = [Polygon(lane) for lane in lanes]
    road_segment_polys = [Polygon(rs) for rs in road_segments]
    

    lanes_union = unary_union(lane_polys)
    road_segments_union = unary_union(road_segment_polys)
    

    intersection = lanes_union.intersection(road_segments_union)
    

    lanes_without_intersection = lanes_union.difference(intersection)
    

    road_segments_without_intersection = road_segments_union.difference(intersection)
    

    road_boundary = unary_union([lanes_without_intersection, road_segments_without_intersection])

    boundary = road_boundary.boundary

    boundary_segments = process_boundary(boundary)
    #print(boundary_segments)
    filtered_segments = []
    boundary_tol = boundary_linestring.buffer(1e-1)

    filtered_segments = []
    for line in boundary_segments:
        if not boundary_tol.contains(line):
            filtered_segments.append(line)


    boundary_coords = [list(seg.coords) for seg in filtered_segments]
    
    return {"road_boundary": boundary_coords}

def plot_boundaries(lanes, road_segments, road_boundary,  height, width, x_center=0, y_center=0):
    plt.figure(figsize=(10, 10))
    
    for lane in lanes:
        x_coords, y_coords = zip(*lane)
        plt.plot(x_coords, y_coords, 'b-', label='Lane' if 'Lane' not in plt.gca().get_legend_handles_labels()[1] else "")
    
    for rs in road_segments:
        x_coords, y_coords = zip(*rs)
        plt.plot(x_coords, y_coords, 'r-', label='Road Segment' if 'Road Segment' not in plt.gca().get_legend_handles_labels()[1] else "")
    
    for boundary in road_boundary["road_boundary"]:
        x_coords, y_coords = zip(*boundary)
        plt.plot(x_coords, y_coords, 'g-', label='Road Boundary' if 'Road Boundary' not in plt.gca().get_legend_handles_labels()[1] else "")
    
    x_min = x_center - width / 2
    x_max = x_center + width / 2
    y_min = y_center - height / 2
    y_max = y_center + height / 2
    boundary_coords = [
        (x_min, y_min),
        (x_max, y_min),
        (x_max, y_max),
        (x_min, y_max),
        (x_min, y_min)
    ]
    x_coords, y_coords = zip(*boundary_coords)
    plt.plot(x_coords, y_coords, 'k--', label='Region Boundary' if 'Region Boundary' not in plt.gca().get_legend_handles_labels()[1] else "")
    
    plt.axis('equal')
    plt.legend()
    plt.show()
def new_get_boundary(map_explore,patch_box, patch_angle,patch_size,layer_names=['road_segment','lane'],):
        map_geom = []
        for layer_name in layer_names:
            geoms = map_explore._get_layer_polygon(patch_box, patch_angle, layer_name)
            map_geom.append((layer_name, geoms))
        
        def poly_geoms_to_vectors(polygon_geom):
            roads = polygon_geom[0][1] 
            lanes = polygon_geom[1][1] 
            union_roads = ops.unary_union(roads) 
            union_lanes = ops.unary_union(lanes) 
            union_segments = ops.unary_union([union_roads, union_lanes])
            max_x = patch_size[0] #/ 2
            max_y = patch_size[1] #/ 2
            local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
            exteriors = []
            interiors = []
            if union_segments.geom_type != 'MultiPolygon':
                union_segments = MultiPolygon([union_segments])
            for poly in union_segments.geoms:
                exteriors.append(poly.exterior) 
                for inter in poly.interiors:
                    interiors.append(inter)
            results = []
            for ext in exteriors:
                if ext.is_ccw: 
                    ext.coords = list(ext.coords)[::-1]
                lines = ext.intersection(local_patch) 
                if isinstance(lines, MultiLineString):
                    lines = ops.linemerge(lines)
                results.append(lines)

            for inter in interiors:
                if not inter.is_ccw:
                    inter.coords = list(inter.coords)[::-1]
                lines = inter.intersection(local_patch)
                if isinstance(lines, MultiLineString):
                    lines = ops.linemerge(lines)
                results.append(lines)
            
            return _one_type_line_geom_to_vectors(results)
        def sample_pts_from_line(line):
            fixed_num = -1
            sample_dist = 1
            num_samples = 2
            if fixed_num < 0:
                distances = np.arange(0, line.length, sample_dist)
                sampled_points = np.array([list(line.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
            else:
                distances = np.linspace(0, line.length, fixed_num)
                sampled_points = np.array([list(line.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)

            num_valid = len(sampled_points)

            if fixed_num > 0:
                return sampled_points, num_valid

            num_valid = len(sampled_points)

            if fixed_num < 0:
                if num_valid < num_samples:
                    padding = np.zeros((num_samples - len(sampled_points), 2))
                    sampled_points = np.concatenate([sampled_points, padding], axis=0)
                else:
                    pass

            return sampled_points, num_valid
        def _one_type_line_geom_to_vectors(line_geom):
            line_vectors = []
            for line in line_geom:
                if not line.is_empty:
                    if line.geom_type == 'MultiLineString':
                        for single_line in line.geoms:
                            line_vectors.append(sample_pts_from_line(single_line))
                    elif line.geom_type == 'LineString':
                        line_vectors.append(sample_pts_from_line(line))
                    else:
                        raise NotImplementedError
            return line_vectors
        def split_points_at_boundary(points_array, boundary_box):
            boundary_polygon = Polygon(boundary_box)
            boundary_tol = boundary_box.buffer(1e-1)
            
            result_points = []
            
            for points in points_array:
                
                if len(points) < 2:
                    continue
                    
                current_segment = [] 
                segments = []       
                
                for i in range(len(points)-1):
                    p1 = points[i]
                    p2 = points[i+1]
                    
                    
                    point1 = Point(p1)
                    point2 = Point(p2)
                    
                    line_segment = LineString([p1, p2])
                    
                    
                    p1_on_boundary = boundary_tol.contains(point1)
                    p2_on_boundary = boundary_tol.contains(point2)
                    
                    
                    if line_segment.within(boundary_tol):
                        
                        if current_segment:
                            current_segment.append(p1)  
                            segments.append(np.array(current_segment))
                            current_segment = []
                        continue
                    
                    
                    if not current_segment:
                        current_segment.append(p1)
                    
                    
                    if p2_on_boundary:
                        current_segment.append(p2)
                        segments.append(np.array(current_segment))
                        current_segment = []
                    else:
                        current_segment.append(p2)
                
                
                if current_segment and len(current_segment) >= 2:
                    segments.append(np.array(current_segment))
                
                
                for segment in segments:
                    if len(segment) >= 2:
                        line = LineString(segment)
                        if line.intersects(boundary_polygon):
                            result_points.append(segment)
            
            return result_points
        end_result = poly_geoms_to_vectors(map_geom)
        result_point = []
        n_edge = {'road_boundary':[]}
        for boundary in end_result:
            result_point.append(boundary[0])
        result_point = split_points_at_boundary(result_point,create_boundary_linestring(0,0,patch_size[0],patch_size[1]))
        n_edge['road_boundary'] = result_point
        return n_edge