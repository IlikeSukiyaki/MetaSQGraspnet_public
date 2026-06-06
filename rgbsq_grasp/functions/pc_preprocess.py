import open3d as o3d
import numpy as np
from functions.utils_numpy import define_SE3, inverse_SE3
import copy
import torch
from sklearn.preprocessing import normalize

def voxel_downsample_npoints(pcd, num_pcd_points):
    v_size = 0.5
    v_min = 0
    v_max = 1
    v_last_met_requirements = None
    cnt = 0
    for trial in range(100):
        num_tmp_pcd_pnts = np.asarray(pcd.voxel_down_sample(v_size).points).shape[0]
        if num_tmp_pcd_pnts - num_pcd_points >= 0 and num_tmp_pcd_pnts - num_pcd_points < 10:
            break 
        if num_tmp_pcd_pnts > num_pcd_points:
            v_last_met_requirements = v_size
            v_min = v_size

        if num_tmp_pcd_pnts < num_pcd_points:
            v_max = v_size
        v_size = (v_min + v_max) / 2
        if trial == 99 and num_tmp_pcd_pnts > num_pcd_points:
            if v_last_met_requirements is not None:
                v_size = v_last_met_requirements
            else:
                v_size = v_min
        cnt += 1

    pcd = pcd.voxel_down_sample(v_size)
    pcd = pcd.select_by_index([*range(num_pcd_points)])

    return pcd

def resample_pc(pcd, num_processed_pc_points):

    # o3d to numpy
    pcd.estimate_normals()
    pcd_numpy = np.asarray(pcd.points)
    pcd_normals_numpy = np.asarray(pcd.normals)

    if len(pcd.points) > num_processed_pc_points:

        random_idx = np.random.choice(len(pcd_numpy), size=num_processed_pc_points, replace = False)
        pcd_numpy = pcd_numpy[random_idx].T
        pcd_numpy = pcd_numpy.T
        pcd.points = o3d.utility.Vector3dVector(pcd_numpy)

    elif len(pcd.points) < num_processed_pc_points:

        # noise parameter
        noise_std = 0.1

        random_idx = np.random.choice(len(pcd_numpy), size=num_processed_pc_points, replace = True)
        pcd_numpy = pcd_numpy[random_idx].T
        pcd_numpy = pcd_numpy.T
        pcd_normals_numpy = pcd_normals_numpy[random_idx].T
        pcd_normals_numpy = pcd_normals_numpy.T

        noise = np.random.uniform(-1, 1, size=pcd_numpy.shape)
        noise = normalize(noise, axis=0, norm='l2')
        noise = np.cross(pcd_normals_numpy, noise)
        noise = noise / np.linalg.norm(noise)
        scale = np.random.normal(loc=0, scale=noise_std, size=(1, pcd_numpy.shape[1])).repeat(pcd_numpy.shape[0], axis=0)
        pcd_numpy = pcd_numpy + noise * scale
        pcd.points = o3d.utility.Vector3dVector(pcd_numpy)
    
    return pcd

def upsampling(X, target_num_data = 1500, bandwidth=1, size=0.8):
    """ YHLee
    upsamplong using tangent space estimation 
    X : np.array, (num_points, 3)
    bandwidth : kernel bandwidth for tangent space estimation 
    size : sampling range
    """

    X = torch.tensor(X, dtype=torch.float)
    num_data = X.shape[0]

    dist_mat = torch.cdist(X, X)
    band_width = bandwidth*dist_mat.sort().values[:, 1].unsqueeze(1)
    Kernel = torch.exp(-dist_mat**2/(band_width**2))
    X_aug = X.unsqueeze(0).repeat(X.size(0), 1, 1)
    diff = X_aug - X.unsqueeze(1)
    V = torch.einsum('nij, ni, nik -> njk', diff, Kernel, diff)
    J = gram_schmidt(torch.svd(V.permute(0, 2, 1)@V).U[:, :, :2])
    num_add_data = target_num_data-num_data

    if num_add_data > 0:
        rand_idx = np.random.randint(0, num_data, num_add_data)
        x = X[rand_idx]
        j = J[rand_idx]
        e1 = j[:, :, 0]
        e2 = j[:, :, 1]
        rad = 0.8*dist_mat.sort().values[:, 1][rand_idx]

        rand1 = (torch.rand(num_add_data)-torch.rand(num_add_data)) * rad
        rand2 = (torch.rand(num_add_data)-torch.rand(num_add_data)) * rad

        x_ = x + rand1.unsqueeze(1)*e1 + rand2.unsqueeze(1)*e2
    else:
        raise ValueError('target number is larger than the num points')
    return torch.cat([X, x_], dim=0).detach().numpy()

def gram_schmidt(X):
    bs = X.size(0)
    dim_n = X.size(1)
    dim_p = X.size(2)
    output = []
    for p in range(dim_p):
        X_temp = X[:, :, p]
        for proj_dim in range(p):
            X_temp = proj(X_temp, X[:, :, proj_dim])
        output.append((X_temp/torch.norm(X_temp, dim=1).unsqueeze(1)).unsqueeze(-1))
    return torch.cat(output, dim=-1)

def proj(X, Y):
    return X - torch.einsum('ni, ni -> n', X, Y).unsqueeze(1)*Y/(torch.norm(Y, dim=1)**2).unsqueeze(1)

def resample_pc_modified(pcd, num_processed_pc_points):

    # o3d to numpy
    pcd.estimate_normals()
    pcd_numpy = np.asarray(pcd.points)
    pcd_normals_numpy = np.asarray(pcd.normals)

    if len(pcd.points) > num_processed_pc_points:

        try:
            pcd = voxel_downsample_npoints(pcd, num_processed_pc_points)
        except:
            pcd_numpy = upsampling(pcd_numpy, floor(1.5 * num_processed_pc_points))
            pcd.points = o3d.utility.Vector3dVector(pcd_numpy)
            pcd = voxel_downsample_npoints(pcd, num_processed_pc_points)

    elif len(pcd.points) < num_processed_pc_points:

        pcd_numpy = upsampling(pcd_numpy, num_processed_pc_points)
        pcd.points = o3d.utility.Vector3dVector(pcd_numpy)

    # pcd.points = o3d.utility.Vector3dVector(pcd_numpy * normalizer)
    # pcd.estimate_normals()
    # o3d.visualization.draw_geometries([pcd])

    
    return pcd

def process_pc(pcd, num_processed_pc_points=3000):

    # resample
    # pcd = resample_pc(pcd, num_processed_pc_points)
    pcd = resample_pc_modified(pcd, num_processed_pc_points)
    
    # transform to pc frame
    pcd, pcd_frame = transform_to_pc_frame(pcd)

    return pcd, pcd_frame

def normalize_numpy_pc(pnts):
    max_ = np.max(pnts, axis=0)
    min_ = np.min(pnts, axis=0)
    diagonal_len = np.linalg.norm(max_-min_)
    pnts = pnts / diagonal_len
    normalizer = diagonal_len

    return pnts, normalizer

def transform_to_pc_frame(pcd):
  
    try:
        bbox = o3d.geometry.OrientedBoundingBox.create_from_points(pcd.points)
    except:
        # print('----------2d convexhull error----------')
        # pnts = np.asarray(pcd.points)
        # pnts[0, 2] += 0.001
        # pcd.points = o3d.utility.Vector3dVector(pnts)
        # bbox = o3d.geometry.OrientedBoundingBox.create_from_points(pcd.points)
        return None, None

    if np.abs(np.linalg.det(bbox.R) + 1) < 1e-4:
        object_frame = define_SE3(-bbox.R, bbox.center)
    else:
        object_frame = define_SE3(bbox.R, bbox.center)

    T_xyz_to_zxy = define_SE3(np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]), [0, 0, 0])
    object_frame = np.matmul(object_frame, T_xyz_to_zxy)
    pcd_cand = copy.deepcopy(pcd)
    pcd_cand.transform(inverse_SE3(object_frame))
    
    # flip frame so that pnts below xy plane has larger volume
    pnts = np.asarray(pcd_cand.points)
    proj_xy_norms = np.linalg.norm(pnts[:, 0:2], axis=1)
    proj_xy_norms_upper_pnts = proj_xy_norms[np.where(pnts[:, 2] > 0)]
    proj_xy_norms_lower_pnts = proj_xy_norms[np.where(pnts[:, 2] < 0)]
    avg_xy_norm_upper = np.mean(proj_xy_norms_upper_pnts)
    avg_xy_norm_lower = np.mean(proj_xy_norms_lower_pnts)
    if avg_xy_norm_upper > avg_xy_norm_lower:
        T_flip_x = define_SE3(np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]), [0, 0, 0])
        object_frame = np.matmul(object_frame, T_flip_x)
        pcd_cand = copy.deepcopy(pcd)
        pcd_cand.transform(inverse_SE3(object_frame))

    pcd_cand2 = copy.deepcopy(pcd_cand)
    pnts = np.asarray(pcd_cand2.points)
    proj_yz_norms = np.linalg.norm(pnts[:, 1:3], axis=1)
    proj_yz_norms_upper_pnts = proj_yz_norms[np.where(pnts[:, 0] > 0)]
    proj_yz_norms_lower_pnts = proj_yz_norms[np.where(pnts[:, 0] < 0)]
    avg_yz_norm_upper = np.mean(proj_yz_norms_upper_pnts)
    avg_yz_norm_lower = np.mean(proj_yz_norms_lower_pnts)
    if avg_yz_norm_upper > avg_yz_norm_lower:
        T_flip_z = define_SE3(np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]), [0, 0, 0])
        object_frame = np.matmul(object_frame, T_flip_z)
        pcd_cand2 = copy.deepcopy(pcd)
        pcd_cand2.transform(inverse_SE3(object_frame))
    
    pcd = copy.deepcopy(pcd_cand2)
    return pcd, object_frame