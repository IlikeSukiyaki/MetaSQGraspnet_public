import numpy as np
from copy import deepcopy
import open3d as o3d

def iou(mesh1, mesh2, voxel_resolution=40, camera_resolution=3):
    
    # calculate scale factor 
    bound_min = np.min(np.array([mesh1.get_min_bound(), mesh2.get_min_bound()]), axis=0)
    bound_max = np.max(np.array([mesh1.get_max_bound(), mesh2.get_max_bound()]), axis=0)
    scale = np.max(bound_max - bound_min)
    center = bound_min + (bound_max - bound_min) / 2

    # mesh scaling
    mesh1_preprocessed = preprocess(mesh1, scale, center)
    mesh2_preprocessed = preprocess(mesh2, scale, center)

    # recalculate bound
    bound_min = np.min(np.array([mesh1_preprocessed.get_min_bound(), mesh2_preprocessed.get_min_bound()]), axis=0)
    bound_max = np.max(np.array([mesh1_preprocessed.get_max_bound(), mesh2_preprocessed.get_max_bound()]), axis=0)
    voxel_size = np.min(bound_max - bound_min) / voxel_resolution
    bound_min = bound_min - 2 * voxel_size
    bound_max = bound_max + 2 * voxel_size
    grid_bound = np.array([bound_min, bound_max]).transpose()

    voxel1, _, _ = voxel_carving(mesh1_preprocessed, grid_bound, voxel_resolution + 4, camera_resolution)
    voxel2, _, _ = voxel_carving(mesh2_preprocessed, grid_bound, voxel_resolution + 4, camera_resolution)
    return voxel_iou(voxel1, voxel2)

def voxel_iou(voxel1, voxel2):
    indices1 = get_voxel_indices(voxel1)
    indices2 = get_voxel_indices(voxel2)
    
    dim = np.max(np.array([np.max(indices1, axis=0), np.max(indices2, axis=0)]), axis=0)

    voxel_bool1 = np.zeros(dim + 1)
    voxel_bool2 = np.zeros(dim + 1)

    try:
        voxel_bool1[indices1[:,0], indices1[:,1], indices1[:,2]] = 1
    except:
        print('voxel1 boolean error!!')

    try:
        voxel_bool2[indices2[:,0], indices2[:,1], indices2[:,2]] = 1
    except:
        print('voxel2 boolean error!!')
    
    intersection = np.sum(np.logical_and(voxel_bool1, voxel_bool2))
    union = np.sum(np.logical_or(voxel_bool1, voxel_bool2))
        
    return float(intersection) / float(union)

def voxel_carving(mesh,
                  grid_bound,
                  voxel_resolution = 0,
                  camera_resolution = 3,
                  w=300,
                  h=300,
                  use_depth=True,
                  surface_method='pointcloud'):
    mesh.compute_vertex_normals()
    cubic_size = grid_bound[:, 1] - grid_bound[:, 0]
    camera_sphere = o3d.geometry.TriangleMesh.create_sphere(1, resolution=camera_resolution)

    # setup dense voxel grid
    voxel_carving = o3d.geometry.VoxelGrid.create_dense(
        width=cubic_size[0],
        height=cubic_size[1],
        depth=cubic_size[2],
        voxel_size=np.min(cubic_size) / voxel_resolution,
        origin=grid_bound[:, 0].tolist(),
        color=[1.0, 0.7, 0.0]
        )

    # setup visualizer to render depthmaps
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=w, height=h, visible=False)
    vis.add_geometry(mesh)
    vis.get_render_option().mesh_show_back_face = True
    ctr = vis.get_view_control()
    param = ctr.convert_to_pinhole_camera_parameters()

    # carve voxel grid
    pcd_agg = o3d.geometry.PointCloud()
    centers_pts = np.zeros((len(camera_sphere.vertices), 3))
    for cid, xyz in enumerate(camera_sphere.vertices):

        # get new camera pose
        trans = get_extrinsic(xyz)
        param.extrinsic = trans
        c = np.linalg.inv(trans).dot(np.asarray([0, 0, 0, 1]).transpose())
        centers_pts[cid, :] = c[:3]
        ctr.convert_from_pinhole_camera_parameters(param)

        # capture depth image and make a point cloud
        vis.poll_events()
        vis.update_renderer()
        depth = vis.capture_depth_float_buffer(False)
        pcd_agg += o3d.geometry.PointCloud.create_from_depth_image(
            o3d.geometry.Image(depth),
            param.intrinsic,
            param.extrinsic,
            depth_scale=1)

        # depth map carving method
        if use_depth:
            voxel_carving.carve_depth_map(o3d.geometry.Image(depth), param)
        else:
            voxel_carving.carve_silhouette(o3d.geometry.Image(depth), param)
    vis.destroy_window()

    # add voxel grid survace
    if surface_method == 'pointcloud':
        voxel_surface = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
            pcd_agg,
            voxel_size=np.min(cubic_size) / voxel_resolution,
            min_bound=tuple(grid_bound[:, 0]),
            max_bound=tuple(grid_bound[:, 1]))
    elif surface_method == 'mesh':
        voxel_surface = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
            mesh,
            voxel_size=cubic_size / voxel_resolution,
            min_bound=tuple(grid_bound[:, 0]),
            max_bound=tuple(grid_bound[:, 1]))
    else:
        raise Exception('invalid surface method')
    voxel_carving_surface = voxel_surface + voxel_carving

    return voxel_carving_surface, voxel_carving, voxel_surface

def get_voxel_indices(voxel):
    indices = []
    for v in voxel.get_voxels():
        indices.append(v.grid_index)
    indices = np.array(indices)
    return indices

def xyz_spherical(xyz):
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    r = np.sqrt(x * x + y * y + z * z)
    r_x = np.arccos(y / r)
    r_y = np.arctan2(z, x)
    return [r, r_x, r_y]

def get_rotation_matrix(r_x, r_y):
    rot_x = np.asarray([[1, 0, 0], [0, np.cos(r_x), -np.sin(r_x)],
                        [0, np.sin(r_x), np.cos(r_x)]])
    rot_y = np.asarray([[np.cos(r_y), 0, np.sin(r_y)], [0, 1, 0],
                        [-np.sin(r_y), 0, np.cos(r_y)]])
    return rot_y.dot(rot_x)

def get_extrinsic(xyz):
    rvec = xyz_spherical(xyz)
    r = get_rotation_matrix(rvec[1], rvec[2])
    t = np.asarray([0, 0, 2]).transpose()
    trans = np.eye(4)
    trans[:3, :3] = r
    trans[:3, 3] = t
    return trans

def preprocess(model, scale, center):
    model_ = deepcopy(model)
    vertices = np.asarray(model_.vertices)
    vertices -= center
    model_.vertices = o3d.utility.Vector3dVector(vertices / scale)
    return model_