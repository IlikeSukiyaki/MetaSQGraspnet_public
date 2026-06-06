import torch

def get_device_info(x):
    cuda_check = x.is_cuda
    if cuda_check:
        device = "cuda:{}".format(x.get_device())
    else:
        device = 'cpu'
    return device

def quaternions_to_rotation_matrices_torch(quaternions):
    
    # initialize
    K = quaternions.shape[0]
    R = quaternions.new_zeros((K, 3, 3))

    # A unit quaternion is q = w + xi + yj + zk
    xx = quaternions[:, 1]**2
    yy = quaternions[:, 2]**2
    zz = quaternions[:, 3]**2
    ww = quaternions[:, 0]**2
    n = (ww + xx + yy + zz).unsqueeze(-1)
    s = quaternions.new_zeros((K, 1))
    s[n != 0] = 2 / n[n != 0]

    xy = s[:, 0] * quaternions[:, 1] * quaternions[:, 2]
    xz = s[:, 0] * quaternions[:, 1] * quaternions[:, 3]
    yz = s[:, 0] * quaternions[:, 2] * quaternions[:, 3]
    xw = s[:, 0] * quaternions[:, 1] * quaternions[:, 0]
    yw = s[:, 0] * quaternions[:, 2] * quaternions[:, 0]
    zw = s[:, 0] * quaternions[:, 3] * quaternions[:, 0]

    xx = s[:, 0] * xx
    yy = s[:, 0] * yy
    zz = s[:, 0] * zz

    idxs = torch.arange(K).to(quaternions.device)
    R[idxs, 0, 0] = 1 - yy - zz
    R[idxs, 0, 1] = xy - zw
    R[idxs, 0, 2] = xz + yw

    R[idxs, 1, 0] = xy + zw
    R[idxs, 1, 1] = 1 - xx - zz
    R[idxs, 1, 2] = yz - xw

    R[idxs, 2, 0] = xz - yw
    R[idxs, 2, 1] = yz + xw
    R[idxs, 2, 2] = 1 - xx - yy

    return R

def rotation_matrices_to_quaternions_torch(R):
    qr = 0.5 * torch.sqrt(1+torch.einsum('ijj->i', R)).unsqueeze(1)
    qi = 1/(4*qr) * (R[:, 2,1] - R[:, 1,2]).unsqueeze(1)
    qj = 1/(4*qr) * (R[:, 0,2] - R[:, 2,0]).unsqueeze(1)
    qk = 1/(4*qr) * (R[:, 1,0] - R[:, 0,1]).unsqueeze(1)

    return torch.cat([qr, qi, qj, qk], dim=1)