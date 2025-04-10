from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap,NuScenesMapExplorer
from shapely import affinity, ops
from shapely.geometry import LineString,Polygon
import numpy as np


def polygon_to_points(polygon: Polygon):
    if polygon is None:
        return np.array([])
    exterior_coords = polygon.exterior.coords
    return exterior_coords  
def get_ped_crossing_line(patch_box, patch_angle,nusc_maps,map_explorer):
    def add_line(poly_xy, idx, patch, patch_angle, patch_x, patch_y):
        points = [(p0, p1) for p0, p1 in zip(poly_xy[0, idx:idx + 2], poly_xy[1, idx:idx + 2])]
        line = LineString(points)
        line = line.intersection(patch)
        if not line.is_empty:
            line = affinity.rotate(line, patch_angle, origin=(patch_x, patch_y), use_radians=False)
            line = affinity.affine_transform(line, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
            return line
        else:
            return None

    patch_x = patch_box[0]
    patch_y = patch_box[1]

    patch = NuScenesMapExplorer.get_patch_coord(patch_box, patch_angle)
    line_list = []
    records = getattr(nusc_maps, 'ped_crossing')
    for record in records:
        polygon = map_explorer.extract_polygon(record['polygon_token'])
        poly_xy = np.array(polygon.exterior.xy)
        #print(poly_xy)
        dist = np.square(poly_xy[:, 1:] - poly_xy[:, :-1]).sum(0)
        x1, x2 = np.argsort(dist)[-2:]

        llx = add_line(poly_xy, x1, patch, patch_angle, patch_x, patch_y)
        lly = add_line(poly_xy, x2, patch, patch_angle, patch_x, patch_y)
        line_list.append([llx, lly]) if llx is not None else None
    return line_list
def get_map_geom(patch_box,map_explorer,nusc_map, patch_angle = 0, layer_names=['road_divider', 'lane_divider','ped_crossing','road_segment', 'lane'],):
    map_geom = {}
    for layer_name in layer_names:
        if layer_name in ['road_divider', 'lane_divider']:
            geomss = []
            geoms = map_explorer._get_layer_line(patch_box, patch_angle, layer_name)
            for geom in geoms:
                if geom.geom_type == 'MultiLineString':
                    for line in geom.geoms:
                        geomss.append(list(line.coords))
                else:
                    geomss.append(list(geom.coords))
            map_geom[layer_name] = geomss
        elif layer_name in ['road_segment', 'lane']:
            geoms = map_explorer._get_layer_polygon(patch_box, patch_angle, layer_name)
            point_sub_geoms = []
            for i in range(len(geoms)):
                sub_geoms = geoms[i].geoms
                # for j in range(len(sub_geoms)):
                point_sub_geoms.append(list(polygon_to_points(sub_geoms[0])))
            map_geom[layer_name] = point_sub_geoms
        elif layer_name in ['ped_crossing']:
            geoms = get_ped_crossing_line(patch_box, patch_angle,nusc_map,map_explorer)
            print(len(geoms))
            for i in range(len(geoms)):
                        try:
                            geoms[i][0] = list(geoms[i][0].coords)
                            geoms[i][1] = list(geoms[i][1].coords)
                        except:
                             pass
                            #  print(geoms[i])
                            #  geoms[i][0] = list(geoms[i][0].coords)
                            #  geoms[i][1] = list(geoms[i][1].coords)
            map_geom[layer_name] = geoms
    return map_geom


