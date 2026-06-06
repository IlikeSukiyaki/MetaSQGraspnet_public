import numpy as np
import math
import open3d as o3d
from scipy.stats import special_ortho_group
from functions.utils_numpy import define_SE3, inverse_SE3
import copy

def add_obj_to_vis(obj, vis):
	obj.add_to_vis(vis)

def get_partial_pc_via_depth(obj, cam_poses, num_pcd_points=3000, get_color=True, plot=False, visualize_pc_with_mesh=False, render_mesh=False):
	
	cam_poses = np.asarray(cam_poses)
	cam_poses = cam_poses.reshape((-1,3))
	num_samples = cam_poses.shape[0]

	depth_im_width = 640 * 2
	depth_im_height = 1000
	FOV_H = 75 # degree
	FOV_V = 65 # degree
	fx = 498.83063258
	fy = 498.83063258
	cx = 0.5
	cy = 0.5

	camera = o3d.camera.PinholeCameraParameters()
	camera.intrinsic.set_intrinsics(depth_im_width, depth_im_height, fx, fy, depth_im_width * cx - 0.5, depth_im_height * cy - 0.5)
	diameter = 1.5 * np.linalg.norm(obj.mesh.get_oriented_bounding_box().extent)
	partial_pcds = []

	vis = o3d.visualization.Visualizer()
	vis.create_window(width=depth_im_width, height=depth_im_height, visible=render_mesh)
	add_obj_to_vis(obj, vis)
	vis.get_render_option().load_from_json("functions/RenderOption.json")

	voxel_size = []
	for sample in range(num_samples):
		cam_up = np.array([0, 0, 1])
		theta = np.arccos(np.dot(cam_up, cam_poses[sample, :]) / np.linalg.norm(cam_poses[sample, :]))
		if abs(theta) < 1e-4 or abs(math.pi - theta) < 1e-4:
			cam_up = np.array([0, 1, 0])
		cam_target = obj.mesh.get_center()
		cam_pos = obj.mesh.get_center() + diameter * cam_poses[sample, :] / np.linalg.norm(cam_poses[sample, :])
		
		cam_look = cam_pos - cam_target
		cam_look = cam_look / np.linalg.norm(cam_look)

		cam_right = np.cross(cam_up, cam_look)
		cam_right = cam_right / np.linalg.norm(cam_right)

		cam_up = np.cross(cam_look, cam_right)
		cam_up = cam_up / np.linalg.norm(cam_up)

		cam_R = np.array([cam_right, - cam_up, - cam_look])
		diameter_zoom = diameter
		cnt = 1
		while True:
			cam_t = - np.dot(cam_R, cam_pos)

			cam_extrinsic_matrix = define_SE3(cam_R, cam_t)

			camera.extrinsic = cam_extrinsic_matrix

			ctr = vis.get_view_control()
			ctr.convert_from_pinhole_camera_parameters(camera)
			
			vis.poll_events()
			vis.update_renderer()

			depth = vis.capture_depth_float_buffer()
			RGB =  vis.capture_screen_float_buffer()
			RGB = o3d.geometry.Image((np.asarray(RGB) * 255).astype(np.uint8))

			if get_color:
				RGBD = o3d.geometry.RGBDImage.create_from_color_and_depth(RGB, depth, convert_rgb_to_intensity=False, depth_scale=1, depth_trunc=3)
				partial_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(RGBD, camera.intrinsic, cam_extrinsic_matrix)
			else:	
				partial_pcd = o3d.geometry.PointCloud.create_from_depth_image(depth, camera.intrinsic, cam_extrinsic_matrix)
				partial_pcd.paint_uniform_color([0, 0, 0.9])

			num_orig_pcd_pnts = np.asarray(partial_pcd.points).shape[0]
			if num_orig_pcd_pnts > num_pcd_points:
				break

			cnt += 1
			print('zooming')
			diameter_zoom = (1.5 - cnt * 0.1) * np.linalg.norm(obj.mesh.get_oriented_bounding_box().extent)

			cam_pos = obj.mesh.get_center() + diameter_zoom * cam_poses[sample, :] / np.linalg.norm(cam_poses[sample, :])
		
		# downsample
		v_size = 0.5
		v_min = 0
		v_max = 1
		v_last_met_requirements = None
		cnt = 0
		for trial in range(100):
			num_tmp_pcd_pnts = np.asarray(partial_pcd.voxel_down_sample(v_size).points).shape[0]
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
		# print('cnt', cnt)
		partial_pcd = partial_pcd.voxel_down_sample(v_size)
		partial_pcd = partial_pcd.select_by_index([*range(num_pcd_points)])
		partial_pcds.append(partial_pcd)
		voxel_size.append(v_size)

	vis.destroy_window()

	if plot is True:
		for sample in range(num_samples):
			if visualize_pc_with_mesh:
				o3d.visualization.draw_geometries([obj.mesh, partial_pcds[sample]])
			else:
				o3d.visualization.draw_geometries([partial_pcds[sample]])

	return partial_pcds, voxel_size


def get_cam_pnts(samples=1, random=False, plot=False):

	cam_points = []
	phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians

	for i in range(samples):
		z = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
		radius = math.sqrt(1 - z * z)  # radius at y

		theta = phi * i  # golden angle increment

		x = math.cos(theta) * radius
		y = - math.sin(theta) * radius

		cam_points.append((x, y, z))

	cam_points = np.asarray(cam_points)
	
	if random is True:
		rand_rot = special_ortho_group.rvs(3)
		cam_points = np.transpose(np.matmul(rand_rot, np.transpose(cam_points)))
	
	if plot is True:
		cam_pcd = o3d.geometry.PointCloud(points = o3d.utility.Vector3dVector(cam_points))
		cam_pcd.paint_uniform_color([1, 0, 0])
		sphere = o3d.geometry.TriangleMesh.create_sphere(radius = 0.99, resolution = 50)
		o3d.visualization.draw_geometries([sphere, cam_pcd])

	return cam_points

def transform_to_pc_frame(pcd):
	try:
		bbox = o3d.geometry.OrientedBoundingBox.create_from_points(pcd.points)
	except:
		# print('----------2d convexhull error----------')
		pnts = np.asarray(pcd.points)
		pnts[0, 2] += 0.001
		pcd.points = o3d.utility.Vector3dVector(pnts)
		bbox = o3d.geometry.OrientedBoundingBox.create_from_points(pcd.points)
		return None, None

	pnts = np.asarray(pcd.points)
	_, plane_ind = pcd.segment_plane(0.003, 3, 1000)
	non_plane_ind = np.setdiff1d(np.arange(pnts.shape[0]), plane_ind)
	if non_plane_ind.shape[0] < 100:
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