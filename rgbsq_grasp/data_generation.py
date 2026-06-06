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
	args = parser.parse_args()

	# configuration
	config = args.config
	if config not in ['primitive', 'object']:
		raise ValueError('config is either primitive or object.')
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
		num_objects = 100	
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
		num_objects = 100	

	# default parameters
	dir_name = os.path.join(savedir, dataset_name)
	append = True
	num_cams = 16	

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

