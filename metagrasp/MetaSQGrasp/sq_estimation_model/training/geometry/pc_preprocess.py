import copy

import numpy as np
import open3d as o3d
import torch
from sklearn.preprocessing import normalize

from geometry.utils_numpy import define_SE3, inverse_SE3


def voxel_downsample_npoints(pcd, num_pcd_points):
    v_size = 0.5
    v_min = 0
    v_max = 1
    v_last_met_requirements = None
    for trial in range(100):
        num_tmp_pcd_pnts = np.asarray(pcd.voxel_down_sample(v_size).points).shape[0]
        if 0 <= num_tmp_pcd_pnts - num_pcd_points < 10:
            break
        if num_tmp_pcd_pnts > num_pcd_points:
            v_last_met_requirements = v_size
            v_min = v_size
        if num_tmp_pcd_pnts < num_pcd_points:
            v_max = v_size
        v_size = (v_min + v_max) / 2
        if trial == 99 and num_tmp_pcd_pnts > num_pcd_points:
            v_size = v_last_met_requirements if v_last_met_requirements is not None else v_min

    pcd = pcd.voxel_down_sample(v_size)
    return pcd.select_by_index([*range(num_pcd_points)])


def upsampling(points, target_num_data=1500, bandwidth=1):
    points = torch.tensor(points, dtype=torch.float)
    num_data = points.shape[0]

    dist_mat = torch.cdist(points, points)
    band_width = bandwidth * dist_mat.sort().values[:, 1].unsqueeze(1)
    kernel = torch.exp(-(dist_mat**2) / (band_width**2))
    points_aug = points.unsqueeze(0).repeat(points.size(0), 1, 1)
    diff = points_aug - points.unsqueeze(1)
    tangent = gram_schmidt(torch.svd(torch.einsum("nij, ni, nik -> njk", diff, kernel, diff).permute(0, 2, 1) @ torch.einsum("nij, ni, nik -> njk", diff, kernel, diff)).U[:, :, :2])
    num_add_data = target_num_data - num_data

    if num_add_data <= 0:
        raise ValueError("target number is smaller than or equal to the number of points")

    rand_idx = np.random.randint(0, num_data, num_add_data)
    x = points[rand_idx]
    j = tangent[rand_idx]
    e1 = j[:, :, 0]
    e2 = j[:, :, 1]
    rad = 0.8 * dist_mat.sort().values[:, 1][rand_idx]

    rand1 = (torch.rand(num_add_data) - torch.rand(num_add_data)) * rad
    rand2 = (torch.rand(num_add_data) - torch.rand(num_add_data)) * rad
    x_aug = x + rand1.unsqueeze(1) * e1 + rand2.unsqueeze(1) * e2
    return torch.cat([points, x_aug], dim=0).detach().numpy()


def gram_schmidt(x):
    output = []
    for p in range(x.size(2)):
        x_temp = x[:, :, p]
        for proj_dim in range(p):
            x_temp = project(x_temp, x[:, :, proj_dim])
        output.append((x_temp / torch.norm(x_temp, dim=1).unsqueeze(1)).unsqueeze(-1))
    return torch.cat(output, dim=-1)


def project(x, y):
    return x - torch.einsum("ni, ni -> n", x, y).unsqueeze(1) * y / (
        torch.norm(y, dim=1) ** 2
    ).unsqueeze(1)


def resample_pc_modified(pcd, num_processed_pc_points):
    pcd.estimate_normals()
    pcd_numpy = np.asarray(pcd.points)

    if len(pcd.points) > num_processed_pc_points:
        try:
            pcd = voxel_downsample_npoints(pcd, num_processed_pc_points)
        except Exception:
            pcd_numpy = upsampling(pcd_numpy, int(1.5 * num_processed_pc_points))
            pcd.points = o3d.utility.Vector3dVector(pcd_numpy)
            pcd = voxel_downsample_npoints(pcd, num_processed_pc_points)
    elif len(pcd.points) < num_processed_pc_points:
        pcd_numpy = upsampling(pcd_numpy, num_processed_pc_points)
        pcd.points = o3d.utility.Vector3dVector(pcd_numpy)

    return pcd


def process_pc(pcd, num_processed_pc_points=3000):
    pcd = resample_pc_modified(pcd, num_processed_pc_points)
    return transform_to_pc_frame(pcd)


def normalize_numpy_pc(points):
    max_ = np.max(points, axis=0)
    min_ = np.min(points, axis=0)
    diagonal_len = np.linalg.norm(max_ - min_)
    return points / diagonal_len, diagonal_len


def transform_to_pc_frame(pcd):
    try:
        bbox = o3d.geometry.OrientedBoundingBox.create_from_points(pcd.points)
    except Exception:
        return None, None

    if np.abs(np.linalg.det(bbox.R) + 1) < 1e-4:
        object_frame = define_SE3(-bbox.R, bbox.center)
    else:
        object_frame = define_SE3(bbox.R, bbox.center)

    t_xyz_to_zxy = define_SE3(np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]), [0, 0, 0])
    object_frame = np.matmul(object_frame, t_xyz_to_zxy)
    pcd_cand = copy.deepcopy(pcd)
    pcd_cand.transform(inverse_SE3(object_frame))

    points = np.asarray(pcd_cand.points)
    proj_xy_norms = np.linalg.norm(points[:, 0:2], axis=1)
    upper = proj_xy_norms[np.where(points[:, 2] > 0)]
    lower = proj_xy_norms[np.where(points[:, 2] < 0)]
    if np.mean(upper) > np.mean(lower):
        t_flip_x = define_SE3(np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]), [0, 0, 0])
        object_frame = np.matmul(object_frame, t_flip_x)
        pcd_cand = copy.deepcopy(pcd)
        pcd_cand.transform(inverse_SE3(object_frame))

    pcd_cand2 = copy.deepcopy(pcd_cand)
    points = np.asarray(pcd_cand2.points)
    proj_yz_norms = np.linalg.norm(points[:, 1:3], axis=1)
    upper = proj_yz_norms[np.where(points[:, 0] > 0)]
    lower = proj_yz_norms[np.where(points[:, 0] < 0)]
    if np.mean(upper) > np.mean(lower):
        t_flip_z = define_SE3(np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]), [0, 0, 0])
        object_frame = np.matmul(object_frame, t_flip_z)
        pcd_cand2 = copy.deepcopy(pcd)
        pcd_cand2.transform(inverse_SE3(object_frame))

    return copy.deepcopy(pcd_cand2), object_frame
