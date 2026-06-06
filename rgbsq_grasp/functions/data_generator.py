import numpy as np
import open3d as o3d
import os
import functions.object_class as object_class
import json
import re
import copy
import csv
import random
import math

from functions.utils_datagen import add_obj_to_vis, get_cam_pnts, get_partial_pc_via_depth, transform_to_pc_frame
from functions.data_reader import create_object_from_data
from functions.utils_numpy import inverse_SE3

def generate_data(object_names, 
				  num_objects, 
				  num_cams, 
				  num_pnts=1500, 
				  append=True, 
				  save_full_pc=False, 
				  save_membership=True, 
				  plot_membership=False, 
				  visualize_object=False, 
				  visualize_cam_points=False, 
				  visualize_pc=False, 
				  visualize_pc_with_mesh=False, 
				  get_color=True, 
				  render_mesh=False, 
				  dir_name=None):

	for object_name in object_names:
		with open(f"object_params/{object_name}.json") as readfile:
			config = json.load(readfile)
		
		if os.path.exists(f'{dir_name}/{object_name}') and append:
			object_numbers = []
			for filename in os.listdir(f"./{dir_name}/{object_name}/"):
				object_numbers.append(int(re.search(r'\d+', filename).group(0)))
			obj_start_num = max(object_numbers)
			if object_numbers.count(obj_start_num) == num_cams:
				obj_start_num += 1
		else:   
			obj_start_num = 0

		for object_num in range(obj_start_num, num_objects):
			object_class_name = object_name.replace('truncated_', '')
			obj = object_class.load_object[f'{object_class_name}'](config)
			
			if visualize_object:
				vis = o3d.visualization.Visualizer()
				vis.create_window()

				add_obj_to_vis(obj, vis)

				vis.run()
				vis.destroy_window()

			# generate camera pnts
			cam_pnts = get_cam_pnts(num_cams, plot=visualize_cam_points)

			# get partial point clouds
			partial_pcds, voxel_size = get_partial_pc_via_depth(obj, cam_pnts, get_color=get_color, render_mesh=render_mesh, plot=visualize_pc, visualize_pc_with_mesh=visualize_pc_with_mesh, num_pcd_points=num_pnts)

			primitives = []
			for primitive_ind in range(obj.num_primitives):
				primitive = dict()
				primitive['type'] = obj.primitives[primitive_ind].type
				primitive['SE3'] = obj.primitives[primitive_ind].SE3
				primitive['parameters'] = obj.primitives[primitive_ind].parameters
				primitives.append(primitive)

			for view_point_num in range(num_cams):
				partial_pcd = partial_pcds[view_point_num]
				partial_pcd, object_frame = transform_to_pc_frame(partial_pcd)

				if partial_pcd is None:
					print(f"----- discard view point {view_point_num} since only a single plane is visible -----")
					continue
				
				data = dict()
				data['partial_pc'] = np.asarray(partial_pcd.points)
				data['primitives'] = copy.deepcopy(primitives)
				if data['partial_pc'].shape[0] != num_pnts:
					print(f"-----------error: {object_num:03}_viewpoint_{view_point_num:02} has {data['partial_pc'].shape[0]} points-----------")
					return
				for data_primitive in data['primitives']:
					data_primitive['SE3'] = np.matmul(inverse_SE3(object_frame), data_primitive['SE3'])
				
				if save_full_pc:
					obj_partial_pc_frame = create_object_from_data(data)
					full_obj_pcd = obj_partial_pc_frame.mesh.sample_points_poisson_disk(1500, use_triangle_normal = True)
					data['full_pc'] = np.concatenate((np.asarray(full_obj_pcd.points),  np.asarray(full_obj_pcd.normals)), axis=1)

				partial_pcd_colors = np.asarray(partial_pcd.colors)

				# voting to reduce noise
				partial_pcd_tree = o3d.geometry.KDTreeFlann(partial_pcd)
				for pnts in range(partial_pcd_colors.shape[0]):
					[_, idx, _] = partial_pcd_tree.search_radius_vector_3d(partial_pcd.points[pnts], 2 * voxel_size[view_point_num])
					partial_pcd_colors[pnts, :] = np.mean(partial_pcd_colors[np.asarray(idx), :], axis=0)

				clrs = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0], [1, 1, 0]]
				
				color_dist = np.zeros((partial_pcd_colors.shape[0], len(clrs)))
				for clr in range(len(clrs)):
					color_dist[:, clr] = np.linalg.norm(partial_pcd_colors - clrs[clr], axis=1)
				
				membership = np.argmin(color_dist, axis=1)

				if len(set(membership)) < len(data['primitives']):
					print(f"----- discard view point {view_point_num} since some primitives are not visible -----")
					continue

				if plot_membership:
					membership_color = np.zeros(partial_pcd_colors.shape)
					for pnt in range(partial_pcd_colors.shape[0]):
						membership_color[pnt, :] = clrs[membership[pnt]]
					
					partial_pcd.colors = o3d.utility.Vector3dVector(membership_color)
					o3d.visualization.draw_geometries([partial_pcd])
				
				if save_membership:
					data['membership'] = membership

				if not os.path.exists('datasets'):
					os.makedirs('datasets')
				
				if dir_name is None:
					if not os.path.exists('datasets/notspecified'):
						os.makedirs('datasets/notspecified')
					if not os.path.exists('datasets/notspecified/'+f"{object_name}"):
						os.makedirs('datasets/notspecified/'+f"{object_name}")
					np.save('datasets/notspecified' + f"/{object_name}/{object_name}_{object_num:04}_viewpoint_{view_point_num:02}.npy", data)
					print(f"saved {object_name}_{object_num:04}_viewpoint_{view_point_num:02}")					
				else:
					if not os.path.exists(dir_name):
						os.makedirs(dir_name)

					if not os.path.exists(dir_name + f'/{object_name}'):
						os.makedirs(dir_name + f'/{object_name}')
					
					np.save(dir_name + f"/{object_name}/{object_name}_{object_num:04}_viewpoint_{view_point_num:02}.npy", data)
					print(f"saved {object_name}_{object_num:04}_viewpoint_{view_point_num:02}")

		return True

def save_data(default_path = 'datasets/', shuffle = True, train_test_ratio = 0.6, train_val_ratio = 0.8):
    
    # load datalist
    obj_list = []
    obj_namelist = [name for name in os.listdir(default_path) if os.path.isdir(os.path.join(default_path, name)) and not name.endswith('.ipynb_checkpoints')]
    for obj_folder in obj_namelist:
        obj_path = default_path + '/' + obj_folder
        file_list = os.listdir(obj_path)
        file_list.sort()

        # object categoralize
        obj_category = []
        obj_index = 0
        while(True):
            prefix = obj_folder + '_' + str(obj_index)
            file_list_category = [file for file in file_list if file.startswith(prefix)]
            if not file_list_category:
                break

            else:
                if shuffle == True:
                    random.shuffle(file_list_category)
                obj_category.append(file_list_category)
                obj_index += 1

        obj_list.append(obj_category)

    # save training data
    csv_path = default_path + '/' + 'train_datalist.csv'
    with open(csv_path, 'w') as csvfile:
        writer = csv.writer(csvfile, 
                            delimiter=',',
                            quotechar='"', 
                            quoting=csv.QUOTE_MINIMAL)
        for category in obj_list:
            for obj in category:
                for obj_index in range(0, math.floor(len(obj) * train_test_ratio * train_val_ratio)):
                    writer.writerow([obj[obj_index]]) 

    # save validation data
    csv_path = default_path + '/' + 'validation_datalist.csv'
    with open(csv_path, 'w') as csvfile:
        writer = csv.writer(csvfile, 
                            delimiter=',',
                            quotechar='"', 
                            quoting=csv.QUOTE_MINIMAL)
        for category in obj_list:
            for obj in category:
                for obj_index in range(math.floor(len(obj) * train_test_ratio * train_val_ratio), math.floor(len(obj) * train_test_ratio)):
                    writer.writerow([obj[obj_index]]) 

    # save test data
    csv_path = default_path + '/' + 'test_datalist.csv'
    with open(csv_path, 'w') as csvfile:
        writer = csv.writer(csvfile, 
                            delimiter=',',
                            quotechar='"', 
                            quoting=csv.QUOTE_MINIMAL)
        for category in obj_list:
            for obj in category:
                for obj_index in range(math.floor(len(obj) * train_test_ratio), len(obj)):
                    writer.writerow([obj[obj_index]])   