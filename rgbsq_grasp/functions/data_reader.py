import open3d as o3d
import functions.primitives as primitives
from functions.object_class import Object

def create_pcd_from_data(data, plot=False):
    pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(data['partial_pc']))
    if plot:
        pcd.paint_uniform_color([0, 0, 0.9])
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.10)
        bbox = o3d.geometry.OrientedBoundingBox.create_from_points(pcd.points)
        o3d.visualization.draw_geometries([pcd, bbox, frame])
    return pcd

def create_set_of_prims_from_data(data):
    set_of_primitives = []
    for primitive in data['primitives']:
        set_of_primitives.append(primitives.gen_primitive[f"{primitive['type']}"](primitive['SE3'], primitive['parameters']))
    return set_of_primitives

def create_object_from_prim_list(set_of_primitives):
    obj = Object(set_of_primitives, transform=False)
    return obj

def create_object_from_data(data, plot=False):
    set_of_primitives = create_set_of_prims_from_data(data)
    obj = create_object_from_prim_list(set_of_primitives)
    if plot:
        o3d.visualization.draw_geometries([obj])
    return obj

