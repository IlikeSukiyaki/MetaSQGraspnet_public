import numpy as np
from copy import deepcopy

def get_SO3(SE3):
    return SE3[0:3, 0:3]

def get_p(SE3):
    return SE3[0:3, 3]

def define_SE3(R, p):
    SE3 = np.identity(4)
    SE3[0:3, 0:3] = R
    SE3[0:3, 3] = p
    return SE3

def change_SO3(SE3, R):
    SE3[0:3, 0:3] = R
    return SE3

def change_p(SE3, p):
    SE3[0:3, 3] = p
    return SE3

def inverse_SE3(SE3):
    R = np.transpose(get_SO3(SE3))
    p = - np.dot(R, get_p(SE3))
    inv_SE3 = define_SE3(R, p)
    return inv_SE3

def transform_point(SE3, p):
    p_h = np.ones(4)
    p_h[0:3] = p
    p_h = np.dot(SE3, p_h)
    return p_h[0:3]

# Matrix to vector
def ToVector(mat):
    if np.size(mat, 1) == 4:          
        vec = np.zeros((6,1))
        vec[0] = -mat[1,2]
        vec[1] = mat[0,2]
        vec[2] = -mat[0,1]
        vec[3] = mat[0,3]
        vec[4] = mat[1,3]
        vec[5] = mat[2,3]
    elif np.size(mat, 1) == 3:     
        vec = np.zeros((3,1))
        vec[0] = -mat(1,2)
        vec[1] = mat(0,2)
        vec[2] = -mat(0,1)
    else:
        raise ValueError('Dimension is not 3 by 3 or 4 by 4')
        
    return vec

# skew matrix
def skew(w):
    W = np.array([[0, -w[2], w[1]],
                  [w[2], 0, -w[0]],
                  [-w[1], w[0], 0]])
    
    return W

# SO3 exponential
def exp_so3(w):
    if len(w) != 3:
        raise ValueError('Dimension is not 3')
    eps = 1e-14
    wnorm = np.sqrt(sum(w*w))
    if wnorm < eps:
        R = np.eye(3)
    else:
        wnorm_inv = 1 / wnorm
        cw = np.cos(wnorm)
        sw = np.sin(wnorm)
        W = skew(w)
        R = np.eye(3) + sw * wnorm_inv * W + (1 - cw) * np.power(wnorm_inv,2) * W.dot(W)

    return R

# SE3 exponential
def exp_se3(S):
    if len(S) != 6:
        raise ValueError('Dimension is not 6')
    w = S[0:3]
    v = S[3:6]
    eps = 1e-14
    wnorm = np.sqrt(sum(w*w))
    if wnorm < eps:
        T = np.eye(4)
        T[0:3,3] = v.reshape(3)
    else:
        wnorm_inv = 1 / wnorm
        cw = np.cos(wnorm)
        sw = np.sin(wnorm)
        W = skew(w)
        P = np.eye(3) + (1 - cw) * np.power(wnorm_inv,2) * W + (wnorm - sw) * np.power(wnorm_inv,3) * W.dot(W)
        T = np.eye(4)
        T[0:3,0:3] = exp_so3(w)
        T[0:3,3] = P.dot(v).reshape(3)
    
    return T

# Logarithm of SO3
def log_SO3(R):
    w = np.zeros((3,1))
    cos_theta = (np.trace(R) - 1) / 2
    theta = np.arccos(cos_theta)
    if np.abs(theta) < 1e-6 :
        w = np.zeros((3,1))
    else :
        if abs(theta - np.pi) < 1e-6 : 
            for k in range(3) :
                if abs(1 + R[k,k]) > 1e-6 :
                    break
            w = deepcopy(R[:,k])
            w[k] = w[k] + 1
            w = w / np.sqrt(2 * (1 + R[k,k])) * theta
        else : 
            w_hat = (R - R.transpose()) / (2 * np.sin(theta)) * theta
            w[0] = w_hat[2,1]
            w[1] = w_hat[0,2]
            w[2] = w_hat[1,0]
            
    return w

# Logarithm of SE3
def log_SE3(T):

    R = T[0:3,0:3]
    p = T[0:3,3]
    logT = np.zeros((4,4))
    # if np.trace(R) < -1:
        # print(np.trace(R))
    cos_theta = (np.trace(R) - 1) / 2
    theta = np.arccos(cos_theta)
    if abs(theta) < 1e-6 :
        logT[0:3,3] = p
    else:
        w = log_SO3(R)
        W = skew(w)
        Pinv = np.eye(3) - 1 / 2 * theta * W + (1 - theta / 2 / np.tan(theta / 2)) * W.dot(W)
        logT[0:3,0:3] = W
        logT[0:3,3] = Pinv.dot(p)
  
    return logT

def quaternions_to_rotation_matrices(quaternions):
    
    # initialize
    K = quaternions.shape[0]
    R = np.zeros((K, 3, 3))

    # A unit quaternion is q = w + xi + yj + zk
    xx = quaternions[:, 1]**2
    yy = quaternions[:, 2]**2
    zz = quaternions[:, 3]**2
    ww = quaternions[:, 0]**2
    n = np.expand_dims((ww + xx + yy + zz), axis=-1)
    s = np.zeros((K, 1))
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

    idxs = range(K)
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