import numpy as np
from tqdm import tqdm
import fcl

from functions.primitives import DeformableSuperquadric
from functions.utils_numpy import get_SO3, inverse_SE3, define_SE3, exp_so3, transform_point

from grasping.gripper import Gripper

# ignore numerical warnings
import warnings
warnings.filterwarnings('ignore')

class Line:
	def __init__(self, point, direction):
		self.point = point
		self.direction = direction / np.linalg.norm(direction)

def generate_collision_free_grasp_pose(obj, num_sample_pnts=100):
	
	# sample antipodal_pnts
	total_antipodal_pnts, total_antipodal_scores = get_antipodal_points_superquadric(obj, num_sample_pnts)

	# collision check
	num_cands = len(total_antipodal_pnts)
	grasp_poses_coll_checked = []
	grasp_poses_coll_checked_score = []
	for step, antipodal_pnts_and_width in enumerate(total_antipodal_pnts):
		antipodal_pnts = antipodal_pnts_and_width[0]
		width = antipodal_pnts_and_width[1]
		grasp_pose_cands = generate_gripper_SE3_candidates(obj, antipodal_pnts, num_pose=12)
		
		for grasp_pose in grasp_pose_cands:

			gripper = Gripper(grasp_pose, width= min(width + 0.02, 0.08), collisionBox=True)
			num_coll = 0
			
			# # check collision with table
			# for grip_collisionBox in gripper.collisionBox:
			# 	request = fcl.CollisionRequest()
			# 	result = fcl.CollisionResult()
			# 	ret = fcl.collide(grip_collisionBox, fcl.CollisionObject(fcl.Halfspace(np.array([0., 0., 1.]), -0.01), fcl.Transform()), request, result)
			# 	num_coll += ret
			# if num_coll > 0:
			# 	continue

			# check collision with obj
			for grip_collisionBox in gripper.collisionBox:
				for obj_collisionBox in obj.collisionBox:
					request = fcl.CollisionRequest()
					result = fcl.CollisionResult()
					ret = fcl.collide(grip_collisionBox, obj_collisionBox, request, result)
					num_coll += ret
			if num_coll == 0 and width < 0.07:
				x_axis = grasp_pose[0:3, 0]
				if np.dot(x_axis, np.array([0, 0, 1])) < 0:
					if grasp_pose[2, 3] > 0.1:
						grasp_poses_coll_checked.append((grasp_pose, width))
						grasp_poses_coll_checked_score.append(total_antipodal_scores[step])
				else:
					if grasp_pose[2, 3] > 0.05:
						grasp_poses_coll_checked.append((grasp_pose, width))
						grasp_poses_coll_checked_score.append(total_antipodal_scores[step])

	return grasp_poses_coll_checked, grasp_poses_coll_checked_score

def generate_gripper_SE3_candidates(obj, antipodal_pnts, num_pose=12):
	obj_center = obj.mesh.get_center()
	antipodal_center = (antipodal_pnts[0] + antipodal_pnts[1]) / 2
	vec_to_obj_center = antipodal_center - obj_center
	antipodal_vec = antipodal_pnts[1] - antipodal_pnts[0]
	antipodal_vec = antipodal_vec / np.linalg.norm(antipodal_vec)
	if np.linalg.norm(vec_to_obj_center) < 1e-4:
		bbox = obj.mesh.get_oriented_bounding_box()
		R = bbox.R
		dot = abs(np.dot(np.transpose(R), antipodal_vec))
		vec_to_obj_center = - R[:, np.argmin(dot)]

	vec_to_obj_center = vec_to_obj_center / np.linalg.norm(vec_to_obj_center)
	gripper_z = vec_to_obj_center
	gripper_y = antipodal_vec
	gripper_x = np.cross(gripper_y, gripper_z)
	gripper_x = gripper_x / np.linalg.norm(gripper_x)
	gripper_z = np.cross(gripper_x, gripper_y)
	gripper_z = gripper_z / np.linalg.norm(gripper_z)

	gripper_SO3 = np.transpose(np.concatenate((gripper_x, gripper_y, gripper_z)).reshape(3,3))
	
	gripper_SE3s = []
	for pose in range(num_pose):
		rot = pose * 2 * np.pi / num_pose
		gripper_SE3 = define_SE3(np.dot(gripper_SO3, exp_so3(np.array([0, 1, 0]) * rot)), antipodal_center)
		gripper_SE3 = np.dot(gripper_SE3, define_SE3(np.identity(3), [0, 0 , -0.1]))
		gripper_SE3s.append(gripper_SE3)
	
	return gripper_SE3s

#########################################################################
################# Functions for Antipodal Sampling ######################
#########################################################################

def get_antipodal_points_superquadric(obj, num_pnts):
	
	# initialize
	total_pnts = []
	total_normals = []

	# sample points
	for p, primitive in enumerate(obj.primitives):
			
		# sampling
		sq_can_frame = DeformableSuperquadric(np.identity(4), primitive.parameters)
		pcd = sq_can_frame.mesh.sample_points_uniformly(int(num_pnts / len(obj.primitives)))

		# get points and their normals
		supquad_pnts = np.asarray(pcd.points)
		eta, omega = get_deformed_sq_polar_coordinate(supquad_pnts, primitive.parameters)
		supquad_normals = get_deformed_sq_normal(eta, omega, primitive.parameters)

		# transform points and their normals
		supquad_pnts = np.matmul(np.concatenate((supquad_pnts, np.ones((supquad_pnts.shape[0], 1))), axis=1), primitive.SE3.transpose())[:, 0:3]
		supquad_normals = np.matmul(supquad_normals, primitive.SE3[0:3, 0:3].transpose())
		
		# append
		total_pnts.append(supquad_pnts)
		total_normals.append(supquad_normals)

	grasp_pnt_cands = np.vstack(total_pnts)
	grasp_pnt_cands_norm = np.vstack(total_normals)

	# sample antipodal points
	total_antipodal_pnts = []
	total_antipodal_scores = []

	# get intersection points
	for cand in tqdm(range(grasp_pnt_cands.shape[0])):
		line = Line(grasp_pnt_cands[cand], grasp_pnt_cands_norm[cand])
		t_antipodal_pnts, antipodal_score = line_object_intersection(line, obj)
		if len(t_antipodal_pnts) == 2 and abs(t_antipodal_pnts[1] - t_antipodal_pnts[0]) > 0.003 and abs(t_antipodal_pnts[1] - t_antipodal_pnts[0]) < 0.074:
			antipodal_pnts = np.array([line.point + t_antipodal_pnts[0] * line.direction, line.point + t_antipodal_pnts[1] * line.direction])
			total_antipodal_pnts.append((antipodal_pnts, abs(t_antipodal_pnts[1] - t_antipodal_pnts[0])))
			total_antipodal_scores.append(antipodal_score)

	return total_antipodal_pnts, total_antipodal_scores

def line_object_intersection(line, obj):
	t = []
	for primitive in obj.primitives:
		intersection_result = line_deformed_superquadric_intersection(line, primitive)
		if type(intersection_result) is tuple:
			t_prim = intersection_result[0]
			antipodal_score = intersection_result[1]
		else:
			t_prim = intersection_result
		t = t + t_prim
	
	if len(t) > 1:
		return [min(t), max(t)], antipodal_score
	elif len(t) == 1:
		return t, None
	else:
		return [], None

def line_deformed_superquadric_intersection(line, superquadric):

	# transform line to superquadric coordinate
	line_point = transform_point(inverse_SE3(superquadric.SE3), line.point)
	line_direction = np.dot(get_SO3(inverse_SE3(superquadric.SE3)), line.direction)
	line_direction = line_direction / np.linalg.norm(line_direction)

	a1 = superquadric.parameters['a1']
	a2 = superquadric.parameters['a2']
	a3 = superquadric.parameters['a3']
	e1 = superquadric.parameters['e1']
	e2 = superquadric.parameters['e2']
	k = superquadric.parameters['k']
	b_prime = superquadric.parameters['b'] / np.maximum(a1, a2)
	alpha = np.arctan2(superquadric.parameters['sin_alpha'], superquadric.parameters['cos_alpha'])

	bounding_sphere_radius = max(a1, a2, a3)
	a = np.dot(line_direction, line_direction)
	b = 2 * np.dot(line_direction, line_point)
	c = np.dot(line_point, line_point) - bounding_sphere_radius**2
	
	D = b**2 - 4 * a * c 
	if D <= 0:
		return []
	else:
		t_b_sphere = [(-b + np.sqrt(D)) / (2 * a), (-b - np.sqrt(D)) / (2 * a)]
		if abs(t_b_sphere[0]) > abs(t_b_sphere[1]):
			t_init = t_b_sphere[0]
		else:
			t_init = t_b_sphere[1]
		
		t_iter = t_init
		for iter in range(1000):
			
			# point
			point_iter = line_direction * t_iter + line_point
			point_inv_b = np.squeeze(inv_bending(np.expand_dims(point_iter, axis=0), superquadric.parameters), axis=0)
			point_iter_sq = np.squeeze(inv_deformation(np.expand_dims(point_iter, axis=0), superquadric.parameters), axis=0)

			# function value
			F_iter = ((abs(point_iter_sq[0]) / a1)**(2 / e2) + (abs(point_iter_sq[1]) / a2)**(2 / e2))**(e2 / e1) + (abs(point_iter_sq[2]) / a3)**(2 / e1) - 1	

			# if find solution
			if abs(F_iter) < 0.1:
    			
				# pass only if antipodal pnt has similar normal
				antipodal_pnt = line_point + t_iter * line_direction
				eta, omega = get_deformed_sq_polar_coordinate(np.expand_dims(antipodal_pnt, axis=1).transpose(), superquadric.parameters)
				supquad_normals = get_deformed_sq_normal(eta, omega, superquadric.parameters)
				antipodal_score = np.linalg.norm(np.cross(supquad_normals[0], line_direction))
				if antipodal_score < np.sin(np.pi / 7):
					return list(np.sort([0, t_iter])), antipodal_score
				else:
					return []

			# derivative of function
			dFdx = (2 / (e1 * a1)) * ((abs(point_iter_sq[0]) / a1)**(2 / e2) + (abs(point_iter_sq[1]) / a2)**(2 / e2))**(e2 / e1 - 1) * (abs(point_iter_sq[0]) / a1) ** (2 / e2 - 1) * np.sign(point_iter_sq[0])
			dFdy = (2 / (e1 * a2)) * ((abs(point_iter_sq[0]) / a1)**(2 / e2) + (abs(point_iter_sq[1]) / a2)**(2 / e2))**(e2 / e1 - 1) * (abs(point_iter_sq[1]) / a2) ** (2 / e2 - 1) * np.sign(point_iter_sq[1])
			dFdz = (2 / (e1 * a3)) * (abs(point_iter_sq[2]) / a3) ** (2 / e1 - 1) * np.sign(point_iter_sq[2])

			# derivative of tapering
			t_k = superquadric.parameters['k'] * point_inv_b[2] / superquadric.parameters['a3'] + 1
			J_t  = np.array([[1 / t_k, 0, - superquadric.parameters['k'] * point_inv_b[0] / (t_k * (superquadric.parameters['k'] * point_inv_b[2] + superquadric.parameters['a3']))],
							[0, 1 / t_k, - superquadric.parameters['k'] * point_inv_b[1] / (t_k * (superquadric.parameters['k'] * point_inv_b[2] + superquadric.parameters['a3']))],
							[0, 0, 1]])

			# derivative of bending
			alpha_atan2 = alpha - np.arctan2(point_iter[1], point_iter[0])
			R = np.cos(alpha_atan2) * np.sqrt(point_iter[0] ** 2 + point_iter[1] ** 2)
			dRdX = (-point_iter[1] * np.sin(alpha_atan2) + point_iter[0] * np.cos(alpha_atan2)) / np.sqrt(point_iter[0]**2 + point_iter[1]**2)
			dRdY = (point_iter[0] * np.sin(alpha_atan2) + point_iter[1] * np.cos(alpha_atan2)) / np.sqrt(point_iter[0]**2 + point_iter[1]**2)
			drdX = (1/b_prime - R) / np.sqrt(point_iter[2] ** 2 + (1/b_prime - R) ** 2) * dRdX
			drdY = (1/b_prime - R) / np.sqrt(point_iter[2] ** 2 + (1/b_prime - R) ** 2) * dRdY
			drdZ = - point_iter[2] / np.sqrt(point_iter[2] ** 2 + (1/b_prime - R) ** 2)
			dgammadX = point_iter[2] / (point_iter[2] ** 2 + (1/b_prime - R) ** 2) * dRdX
			dgammadY = point_iter[2] / (point_iter[2] ** 2 + (1/b_prime - R) ** 2) * dRdY
			dgammadZ = (1/b_prime - R) / (point_iter[2] ** 2 + (1/b_prime - R) ** 2)
			J_b = np.array([[1 - np.cos(alpha) * (dRdX - drdX) , - np.cos(alpha) * (dRdY - drdY)   , np.cos(alpha) * drdZ        ],
							[- np.sin(alpha) * (dRdX - drdX)    , 1 - np.sin(alpha) * (dRdY - drdY) , np.sin(alpha) * drdZ        ],
							[1/b_prime * dgammadX               , 1/b_prime * dgammadY              , 1/b_prime * dgammadZ]])
			
			# derivative of function
			J = np.dot(J_t, J_b)
			dF_iter = np.dot(np.array([dFdx, dFdy, dFdz]), np.dot(J, line_direction))

			t_iter = t_iter - F_iter / dF_iter

		return []

#########################################################################
############### Functions for Deformable Superquadrics ##################
#########################################################################

def get_deformed_sq_polar_coordinate(deformed_supquad_pnts, deformed_supquad_parameters):
	sq_pnts = inv_deformation(deformed_supquad_pnts, deformed_supquad_parameters)
	return get_sq_polar_coordinate(sq_pnts, deformed_supquad_parameters)

def get_deformed_sq_normal(eta, omega, deformed_supquad_parameters):
		
	sq_pnts = get_sq_xyz_coordinate(eta, omega, deformed_supquad_parameters)
	sq_normals = get_sq_normal(eta, omega, deformed_supquad_parameters)
	t_k = deformed_supquad_parameters['k'] * sq_pnts[:, 2] / deformed_supquad_parameters['a3'] + 1
	
	J_t  = np.transpose(np.array([[t_k, np.zeros(t_k.shape), deformed_supquad_parameters['k'] * sq_pnts[:, 0] / deformed_supquad_parameters['a3']],
									[np.zeros(t_k.shape), t_k, deformed_supquad_parameters['k'] * sq_pnts[:, 1] / deformed_supquad_parameters['a3']],
									[np.zeros(t_k.shape), np.zeros(t_k.shape), np.ones(t_k.shape)]]), (2, 0, 1))
	
	b_prime = deformed_supquad_parameters['b'] / np.maximum(deformed_supquad_parameters['a1'], deformed_supquad_parameters['a2'])
	D_t = np.zeros(sq_pnts.shape)
	D_t[:, 0] = t_k * sq_pnts[:, 0]
	D_t[:, 1] = t_k * sq_pnts[:, 1]
	D_t[:, 2] = sq_pnts[:, 2]
	gamma = D_t[:, 2] * b_prime
	
	alpha = np.arctan2(deformed_supquad_parameters['sin_alpha'], deformed_supquad_parameters['cos_alpha'])
	
	alpha_atan2 = alpha - np.arctan2(D_t[:, 1], D_t[:, 0])
	r = np.cos(alpha_atan2) * np.sqrt(D_t[:, 0]**2 + D_t[:, 1]**2)
	drdx = (-D_t[:, 1] * np.sin(alpha_atan2) + D_t[:, 0] * np.cos(alpha_atan2)) / np.sqrt(D_t[:, 0]**2 + D_t[:, 1]**2)
	drdy = (D_t[:, 0] * np.sin(alpha_atan2) + D_t[:, 1] * np.cos(alpha_atan2)) / np.sqrt(D_t[:, 0]**2 + D_t[:, 1]**2)
	
	dRdx = np.cos(gamma) * drdx
	dRdy = np.cos(gamma) * drdy
	dRdz = b_prime * np.sin(gamma) * (1/b_prime - r)

	J_b = np.transpose(np.array([[1 + np.cos(alpha) * (dRdx - drdx), np.cos(alpha) * (dRdy - drdy)     , np.cos(alpha) * dRdz        ],
								 [np.sin(alpha) * (dRdx - drdx)    , 1 + np.sin(alpha) * (dRdy - drdy) , np.sin(alpha) * dRdz        ],
								 [-np.sin(gamma) * drdx            , -np.sin(gamma) * drdy             , b_prime * np.cos(gamma) * (1/b_prime - r)]]), (2, 0, 1))
	
	J = np.matmul(J_b ,J_t)

	supquad_normals = np.matmul(np.expand_dims(np.expand_dims(np.linalg.det(J), axis=1), axis=2) * np.transpose(np.linalg.inv(J), (0, 2, 1)), sq_normals[:, :, None]).squeeze(-1)
	supquad_normals = supquad_normals / np.expand_dims(np.linalg.norm(supquad_normals, axis=1), axis=1)

	return supquad_normals

#########################################################################
###################### Functions for Superquadrics ######################
#########################################################################

def get_sq_polar_coordinate(supquad_pnts, supquad_parameters):
	eta = np.arcsin(fexp(supquad_pnts[:, 2]/supquad_parameters['a3'], 1 / supquad_parameters['e1']))
	cosw = fexp(supquad_pnts[:, 0] / (supquad_parameters['a1'] * fexp(np.cos(eta), supquad_parameters['e1'])), 1 / supquad_parameters['e2'])
	sinw = fexp(supquad_pnts[:, 1] / (supquad_parameters['a1'] * fexp(np.cos(eta), supquad_parameters['e1'])), 1 / supquad_parameters['e2'])
	omega = np.arctan2(sinw, cosw)

	return eta, omega

def get_sq_xyz_coordinate(eta, omega, supquad_parameters):
	# make new vertices
	x = supquad_parameters['a1'] * fexp(np.cos(eta), supquad_parameters['e1']) * fexp(np.cos(omega), supquad_parameters['e2'])
	y = supquad_parameters['a2'] * fexp(np.cos(eta), supquad_parameters['e1']) * fexp(np.sin(omega), supquad_parameters['e2'])
	z = supquad_parameters['a3'] * fexp(np.sin(eta), supquad_parameters['e1'])

	return np.array([x, y, z]).transpose()

def get_sq_normal(eta, omega, supquad_parameters):
	n_x = fexp(np.cos(eta), 2 - supquad_parameters['e1']) * fexp(np.cos(omega), 2 - supquad_parameters['e2']) / supquad_parameters['a1']
	n_y = fexp(np.cos(eta), 2 - supquad_parameters['e1']) * fexp(np.sin(omega), 2 - supquad_parameters['e2']) / supquad_parameters['a2']
	n_z = fexp(np.sin(eta), 2 - supquad_parameters['e1']) / supquad_parameters['a3']

	supquad_normals = np.array([n_x, n_y, n_z]).transpose()
	supquad_normals = supquad_normals / np.expand_dims(np.linalg.norm(supquad_normals, axis=1), axis=1)

	return supquad_normals

#########################################################################
######################### Deformation Functions #########################
#########################################################################

def inv_deformation(dsq_pnts, dsq_parameters):
	return inv_taper(inv_bending(dsq_pnts, dsq_parameters), dsq_parameters)

def inv_bending(dsq_pnts, dsq_parameters):
	alpha = np.arctan2(dsq_parameters['sin_alpha'], dsq_parameters['cos_alpha'])
	b = dsq_parameters['b'] / np.maximum(dsq_parameters['a1'], dsq_parameters['a2'])
	R = np.cos(alpha - np.arctan2(dsq_pnts[:, 1], dsq_pnts[:, 0])) * np.sqrt(dsq_pnts[:, 0] ** 2 + dsq_pnts[:, 1] ** 2)
	r = 1/b - np.sqrt(dsq_pnts[:, 2] ** 2 + (1/b - R) ** 2)
	gamma = np.arctan2(dsq_pnts[:, 2], 1/b - R)

	x = dsq_pnts[:, 0] - np.cos(alpha) * (R - r)
	y = dsq_pnts[:, 1] - np.sin(alpha) * (R - r)
	z = 1 / b * gamma

	return np.array([x, y, z]).transpose()

def inv_taper(dsq_pnts, dsq_parameters):
	t_k = dsq_parameters['k'] * dsq_pnts[:, 2] / dsq_parameters['a3'] + 1
	x = dsq_pnts[:, 0] / t_k
	y = dsq_pnts[:, 1] / t_k
	z = dsq_pnts[:, 2] 

	return np.array([x, y, z]).transpose()

def fexp(x, p):
	return np.sign(x)*(np.abs(x)**p)