import os
import argparse
from datetime import datetime
from functions.data_generator import generate_data, save_data

if __name__ == '__main__':
	
	# parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', type=str)
	parser.add_argument("--savedir", default="datasets/")
	parser.add_argument("--name", default=None)
	parser.add_argument("--num-objects", type=int, default=None)
	args = parser.parse_args()

	# configuration
	config = args.config
	if config not in ['primitive', 'object', 'extended_sq', 'superquadric']:
		raise ValueError('config is primitive, object, extended_sq, or superquadric.')
	savedir = args.savedir
	if args.name is None:
		dataset_name = datetime.now().strftime("%Y%m%d-%H%M")
	else:
		dataset_name = args.name

	# parameters
	if config == 'primitive':
		object_names = ['box', 
						'cylinder', 
						'ellipsoid', 
						'cone', 
						'truncated_cone', 
						'truncated_torus'
						]    
		num_partial_pnts = 1500
		save_full_pc = True
		save_membership = False
		num_objects = args.num_objects if args.num_objects is not None else 100	
	elif config == 'object':
		object_names = ['box', 
						'cylinder', 
						'ellipsoid', 
						'cone', 
						'truncated_cone', 
						'truncated_torus',
						'bottle_cone',
						'screw_driver',
						'cup_with_lid',
						'hammer_cylinder',
						'padlock',
						'dumbbell']
		num_partial_pnts = 3000
		save_full_pc = False
		save_membership = True
		num_objects = args.num_objects if args.num_objects is not None else 100	
	elif config == 'extended_sq':
		object_names = ['extended_sq']
		num_partial_pnts = 1500
		save_full_pc = True
		save_membership = False
		num_objects = args.num_objects if args.num_objects is not None else 1000
	elif config == 'superquadric':
		object_names = ['superquadric']
		num_partial_pnts = 1500
		save_full_pc = True
		save_membership = False
		num_objects = args.num_objects if args.num_objects is not None else 100

	# default parameters
	dir_name = os.path.join(savedir, dataset_name)
	append = True
	num_cams = 16
	print(f"Generating dataset under {dir_name} (one isolated worker process per object instance)")

	for trial in range(1000):
		success = generate_data(object_names, 
								num_objects, 
								num_cams, 
								num_pnts=num_partial_pnts, 
								append=append, 
								save_full_pc=save_full_pc, 
								save_membership=save_membership, 
								dir_name=dir_name)
		if success:
			object_names.pop(0)
		if not object_names:
			break

	# split
	save_data(
			default_path = dir_name,
			shuffle = False, 
			train_test_ratio = 0.9,
			train_val_ratio = 0.9
	)

