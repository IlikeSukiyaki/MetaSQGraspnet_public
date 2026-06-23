import numpy as np
import random
import open3d as o3d

from functions.utils_numpy import define_SE3, inverse_SE3, change_SO3, get_p, get_SO3, exp_so3
from functions.primitives import Cylinder, Box, Cone, Torus, Torus_handle, Superquadric, DeformableSuperquadric

class Object:
	def __init__(self, set_of_primitives, transform=True, collisionBox=False):
		self.initialzier(set_of_primitives, transform)
		if collisionBox:
			self.collisionBox = []
			for prim in set_of_primitives:
				self.collisionBox.append(prim.collisionBox)

	def initialzier(self, set_of_primitives, transform=True):
		self.num_primitives = len(set_of_primitives)
		self.primitives = set_of_primitives.copy()
		self.mesh = self.primitives[0].mesh
		for primitive in self.primitives[1:]:
			self.mesh = self.mesh + primitive.mesh
		if transform is not False:
			self.transform_to_object_frame(transform)

	def get_object_frame(self):
		num_pts = 10000
		obj_pcd = self.mesh.sample_points_uniformly(number_of_points=num_pts)
		mean, cov = obj_pcd.compute_mean_and_covariance()
		eig_values, eig_vectors = np.linalg.eig(cov)
		principle_axes = eig_vectors[np.argsort(-1 * eig_values), :]
		z = principle_axes[0, :] / np.linalg.norm(principle_axes[0, :])
		x = principle_axes[1, :] / np.linalg.norm(principle_axes[1, :])
		y = np.cross(z, x)
	
		T = np.identity(4)
		T[0:3, 0:3] = np.array([x, y, z]).transpose()
		T[0:3, 3] = mean
		
		return T

	def get_object_frame2(self):
		num_pts = 10000
		obj_pcd = self.mesh.sample_points_uniformly(number_of_points=num_pts)
		bbox = o3d.geometry.OrientedBoundingBox.create_from_points(obj_pcd.points)
		if np.abs(np.linalg.det(bbox.R) + 1) < 1e-4:
			object_frame = define_SE3(-bbox.R, bbox.center)
		else:
			object_frame = define_SE3(bbox.R, bbox.center)
		T_xyz_to_zxy = define_SE3(np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]), [0, 0, 0])
		object_frame = np.matmul(object_frame, T_xyz_to_zxy)
		
		return object_frame

	def transform_object(self, SE3):
		for primitive in range(self.num_primitives):
			self.primitives[primitive].mesh.transform(SE3)
			self.primitives[primitive].SE3 = np.matmul(SE3, self.primitives[primitive].SE3)
			if hasattr(self.primitives[primitive], 'collisionBox'):
				T = define_SE3(self.primitives[primitive].collisionBox.getRotation(), self.primitives[primitive].collisionBox.getTranslation())
				T = np.dot(SE3, T)
				self.primitives[primitive].collisionBox.setTransform(fcl.Transform(get_SO3(T), get_p(T)))
		self.mesh.transform(SE3)

	def transform_to_object_frame(self,transform):
		object_frame = self.get_object_frame2()
		if transform == 'center':
			object_frame = change_SO3(object_frame, np.identity(3))
		inv_obj_frame = inverse_SE3(object_frame)
		self.transform_object(inv_obj_frame) 

	def add_to_vis(self,vis):
		vis.add_geometry(self.mesh)


class Box_Object(Object):

	def __init__(self, config):
		if config['random'] is True:
			w = random.uniform(config['w']['min'], config['w']['max'])
			d = random.uniform(config['d']['min'], config['d']['max'])
			h = random.uniform(config['h']['min'], config['h']['max'])

		else: 
			w = config['w']
			d = config['d']
			h = config['h']

		lengths = [w, d, h]
		lengths.sort()

		w = lengths[0]
		d = lengths[1]
		h = lengths[2]

		P1_parameters = {"width": w, "depth": d, "height": h}

		P1_SE3 = np.identity(4)

		if 'color' in config.keys():
			P1_color = config['color']
			P1 = Box(SE3=P1_SE3, parameters=P1_parameters, color=P1_color)
		else:
			P1 = Box(SE3=P1_SE3, parameters=P1_parameters, color=None)

		self.initialzier([P1], transform=False)


class Cylinder_Object(Object):

	def __init__(self, config):
    		
		if config['random'] is True:
			r = random.uniform(config['r']['min'], config['r']['max'])
			h = random.uniform(config['h']['min'], config['h']['max'])
		else: 
			r = config['r']
			h = config['h']

		P1_SE3 = np.identity(4)
		P1_parameters = {"radius": r, "height": h}

		if 'color' in config.keys():
			P1_color = config['color']
			P1 = Cylinder(SE3=P1_SE3, parameters=P1_parameters, color=P1_color)
		else:
			P1 = Cylinder(SE3=P1_SE3, parameters=P1_parameters)

		self.initialzier([P1], transform=False)


class Cone_Object(Object):

	def __init__(self, config):
		if config['random'] is True:
			r_l = random.uniform(config['r_l']['min'], config['r_l']['max'])
			if 'r_u' in config.keys() and config['r_u'] is not None:
				r_u = random.uniform(config['r_u']['min'], min(config['r_u']['max'], 0.8 * r_l))
   
				if r_u > r_l:
					r_u, r_l = r_l, r_u
				elif r_u == r_l:
					r_u = r_u - config['r_u']['min']/2
   
			else:
				r_u = 0
			h = random.uniform(config['h']['min'], config['h']['max'])

		else: 
			r_l = config['r_l']
			r_u = config['r_u']
			h = config['h']
		
		P1_SE3 = np.identity(4)
		P1_parameters = {"lower_radius": r_l, "upper_radius": r_u, "height": h}

		if 'color' in config.keys():
			P1_color = config['color']
			P1 = Cone(SE3=P1_SE3, parameters=P1_parameters, color=P1_color)
		else:
			P1 = Cone(SE3=P1_SE3, parameters=P1_parameters)

		self.initialzier([P1], transform=False)


class Torus_Object(Object):

	def __init__(self, config):
		if config['random'] is True:
			r_torus = random.uniform(config['r_torus']['min'], config['r_torus']['max'])
			r_tube = random.uniform(max(0.01, config['r_tube']['min']), min(config['r_tube']['max'], 0.8 * r_torus))
			if 'c1' in config.keys() and config['c1'] is True:
				c1 = random.uniform(3.14/6, 3.14/2)
			else:
				c1 = r_torus + r_tube
		else: 
			r_torus = config['r_torus']
			r_tube = config['r_tube']
			c1 = config['c1']
	
		P1_SE3 = define_SE3(np.identity(3), [0, 0, 0])
		P1_parameters = {"torus_radius": r_torus, "tube_radius": r_tube, "c1": c1}

		if 'color' in config.keys():
			P1_color = config['color']
		else:
			P1_color = None

		P1 = Torus(SE3=P1_SE3, parameters=P1_parameters, color=P1_color)
	
		self.initialzier([P1], transform=False)


class Ellipsoid_Object(Object):
    
	def __init__(self, config):
		if config['random'] is True:
			a1 = random.uniform(config['a1']['min'], config['a1']['max'])
			a2 = random.uniform(config['a2']['min'], config['a2']['max'])
			a3 = random.uniform(config['a3']['min'], config['a3']['max'])
			
		else: 
			a1 = config['a1']
			a2 = config['a2']
			a3 = config['a3']
		
		if a1 > a2:
			tmp = a1
			a1 = a2
			a2 = tmp

		P1_SE3 = np.identity(4)
		P1_parameters = {"a1": a1, "a2": a2, "a3": a3, "e1": 1, "e2": 1}
		
		if 'color' in config.keys():
			P1_color = config['color']
		else:
			P1_color = None

		P1 = Superquadric(SE3=P1_SE3, parameters=P1_parameters, color=P1_color)

		self.initialzier([P1], transform=False)


class Superquadric_Object(Object):

	def __init__(self, config):
		if config['random'] is True:
			a_min = config['a1']['min']
			a_max = config['a1']['max']
			max_ratio = config.get('max_axis_ratio', 3.5)
			for _ in range(100):
				axes = sorted(random.uniform(a_min, a_max) for _ in range(3))
				if axes[2] / max(axes[0], 1e-6) <= max_ratio:
					a1, a2, a3 = axes[0], axes[1], axes[2]
					break
			else:
				axes = sorted(random.uniform(a_min, a_max) for _ in range(3))
				a1, a2, a3 = axes[0], axes[1], axes[2]

			e1 = random.uniform(config['e1']['min'], config['e1']['max'])
			e2 = random.uniform(config['e2']['min'], config['e2']['max'])
		else:
			a1 = config['a1']
			a2 = config['a2']
			a3 = config['a3']
			e1 = config['e1']
			e2 = config['e2']

		if a1 > a2:
			a1, a2 = a2, a1

		P1_SE3 = np.identity(4)
		P1_parameters = {"a1": a1, "a2": a2, "a3": a3, "e1": e1, "e2": e2}

		if 'color' in config.keys():
			P1_color = config['color']
		else:
			P1_color = None

		P1 = Superquadric(SE3=P1_SE3, parameters=P1_parameters, color=P1_color)

		self.initialzier([P1], transform=False)


def _sample_dsq_axes(profile, a_min, a_max, max_ratio):
	"""Sample (a1, a2, a3) with profile-specific axis-ratio constraints; a1 <= a2 <= a3."""
	for _ in range(100):
		if profile == "compact":
			ratio_cap = min(1.8, max_ratio)
			axes = sorted(random.uniform(a_min, a_max) for _ in range(3))
		elif profile == "elongated":
			short = random.uniform(a_min, a_max * 0.5)
			mid = random.uniform(short, min(a_max, short * 2.0))
			long = random.uniform(mid, min(a_max, short * max_ratio))
			axes = sorted([short, mid, long])
		elif profile == "flattened":
			short = random.uniform(a_min, a_max * 0.4)
			mid = random.uniform(max(short * 1.2, a_min), a_max)
			long = random.uniform(mid, min(a_max, short * max_ratio))
			axes = sorted([short, mid, long])
		else:
			axes = sorted(random.uniform(a_min, a_max) for _ in range(3))

		if axes[2] / max(axes[0], 1e-6) <= max_ratio:
			return axes[0], axes[1], axes[2]

	axes = sorted(random.uniform(a_min, a_max) for _ in range(3))
	return axes[0], axes[1], axes[2]


class ExtendedSQ_Object(Object):

	def __init__(self, config):
		if config["random"] is True:
			profiles = ["compact", "elongated", "flattened", "high_deformation"]
			weights = config.get("profile_weights", [0.4, 0.3, 0.2, 0.1])
			profile = random.choices(profiles, weights=weights, k=1)[0]

			a_min = config["a1"]["min"]
			a_max = config["a1"]["max"]
			max_ratio = config.get("max_axis_ratio", 3.5)
			a1, a2, a3 = _sample_dsq_axes(profile, a_min, a_max, max_ratio)

			if a1 > a2:
				a1, a2 = a2, a1

			e1 = random.uniform(config["e1"]["min"], config["e1"]["max"])
			e2 = random.uniform(config["e2"]["min"], config["e2"]["max"])

			if profile == "high_deformation":
				k = random.uniform(-0.65, -0.35) if random.random() < 0.5 else random.uniform(0.35, 0.65)
				b = random.uniform(0.35, 0.55)
			else:
				k = random.uniform(config["k"]["min"], config["k"]["max"])
				b = random.uniform(config["b"]["min"], config["b"]["max"])

			alpha = random.uniform(0, 2 * np.pi)
			cos_alpha = float(np.cos(alpha))
			sin_alpha = float(np.sin(alpha))
		else:
			profile = "fixed"
			a1 = config["a1"]
			a2 = config["a2"]
			a3 = config["a3"]
			e1 = config["e1"]
			e2 = config["e2"]
			k = config["k"]
			b = config["b"]
			cos_alpha = config["cos_alpha"]
			sin_alpha = config["sin_alpha"]
			alpha = float(np.arctan2(sin_alpha, cos_alpha))

		self.sampling_profile = profile
		self.alpha = alpha

		P1_SE3 = np.identity(4)
		P1_parameters = {
			"a1": a1,
			"a2": a2,
			"a3": a3,
			"e1": e1,
			"e2": e2,
			"k": k,
			"b": b,
			"cos_alpha": cos_alpha,
			"sin_alpha": sin_alpha,
		}

		if "color" in config.keys():
			P1_color = config["color"]
		else:
			P1_color = None

		P1 = DeformableSuperquadric(SE3=P1_SE3, parameters=P1_parameters, color=P1_color)
		P1.type = "extended_sq"

		self.initialzier([P1], transform=False)


DeformableSuperquadric_Object = ExtendedSQ_Object


class Bottle_Cone(Object):

	def __init__(self, config, collisionBox=False):
		if config['random'] is True:
			h1 = random.uniform(config['h1']['min'], config['h1']['max'])
			h2 = random.uniform(config['h2']['min'], config['h2']['max'])
			h3 = random.uniform(config['h3']['min'], config['h3']['max'])

			r1 = random.uniform(config['r1']['min'], config['r1']['max'])
			r_l2 = r1
			r_u2 = random.uniform(config['r_u2']['min'], min(config['r_u2']['max'], 2 * r_l2 / 3))
			r3 = random.uniform(r_u2, min(config['r3']['max'], r1-0.001))
		else: 
			h1 = config['h1']
			h2 = config['h2']
			h3 = config['h3']

			r1 = config['r1']
			r_l2 = config['r_l2']
			r_u2 = config['r_u2']
			r3 = config['r3']

		P1_SE3 = define_SE3(np.identity(3), [0, 0, 0])
		P2_SE3 = define_SE3(np.identity(3), [0, 0, h1 / 2 + h2 / 2])
		P3_SE3 = define_SE3(np.identity(3), [0, 0, h1 / 2 + h2 + h3 / 2])

		P1_parameters = {"radius": r1, "height": h1}
		P2_parameters = {"lower_radius": r_l2, "upper_radius": r_u2, "height": h2}
		P3_parameters = {"radius": r3, "height": h3}

		if 'color1' in config.keys():
			P1_color = config['color1']
			P1 = Cylinder(SE3=P1_SE3, parameters=P1_parameters, color=P1_color)
		else:
			P1 = Cylinder(SE3=P1_SE3, parameters=P1_parameters)
		if 'color2' in config.keys():
			P2_color = config['color2']
			P2 = Cone(SE3=P2_SE3, parameters=P2_parameters, color=P2_color)
		else:
			P2 = Cone(SE3=P2_SE3, parameters=P2_parameters)
		if 'color3' in config.keys():
			P3_color = config['color3']
			P3 = Cylinder(SE3=P3_SE3, parameters=P3_parameters, color=P3_color)
		else:
			P3 = Cylinder(SE3=P3_SE3, parameters=P3_parameters)
	
		self.initialzier([P1, P2, P3], transform='center')
		if collisionBox:
			self.collisionBox = []
			for primitive in self.primitives:
				self.collisionBox.append(primitive.collisionBox)

class Dumbbell(Object):

	def __init__(self, config):
		if config['random'] is True:
			h1 = random.uniform(config['h1']['min'], config['h1']['max'])
			h2 = random.uniform(config['h2']['min'], config['h2']['max'])

			r2 = random.uniform(config['r2']['min'], config['r2']['max'])
			r1 = random.uniform(max(config['r1']['min'], r2), config['r1']['max'])
		else: 
			h1 = config['h1']
			h2 = config['h2']

			r1 = config['r1']
			r2 = config['r2']
		
		r3 = r1
		h3 = h1

		P1_SE3 = define_SE3(np.identity(3), [0, 0, 0])
		P2_SE3 = define_SE3(np.identity(3), [0, 0, h1 / 2 + h2 / 2])
		P3_SE3 = define_SE3(np.identity(3), [0, 0, h1 / 2 + h2 + h3 / 2])

		P1_parameters = {"radius": r1, "height": h1}
		P2_parameters = {"radius": r2, "height": h2}
		P3_parameters = {"radius": r3, "height": h3}

		if 'color1' in config.keys():
			P1_color = config['color1']
		else:
			P1_color = None
		if 'color2' in config.keys():
			P2_color = config['color2']
		else:
			P2_color = None
		if 'color3' in config.keys():
			P3_color = config['color3']
		else:
			P3_color = None

		P1 = Cylinder(SE3=P1_SE3, parameters=P1_parameters, color=P1_color)
		P2 = Cylinder(SE3=P2_SE3, parameters=P2_parameters, color=P2_color)
		P3 = Cylinder(SE3=P3_SE3, parameters=P3_parameters, color=P3_color)
	
		self.initialzier([P1, P2, P3], transform='center')


class Padlock(Object):
    
	def __init__(self, config):
		if config['random'] is True:
			w = random.uniform(config['w']['min'], config['w']['max'])
			d = random.uniform(max(w, config['d']['min']), config['d']['max'])
			h = random.uniform(max(d, config['h']['min']), config['h']['max'])
			
			r_tube = random.uniform(config['r_tube']['min'], min(w, config['r_tube']['max']))
			r_torus = random.uniform(config['r_torus']['min'], min(config['r_torus']['max'], 0.9 * (h / 2 - r_tube)))
			c1 = random.uniform(0.2 * (-r_torus + r_tube), 0.2 * (r_torus - r_tube))
			 
		else: 
			r1 = config['r1']
			h1 = config['h1']
			t1 = config['t1']

			r_tube = config['r_tube']
			r_torus = config['r_torus']
			c1 = config['c1']
			z3 = config['z3']
		
		P1_SE3 = define_SE3(exp_so3(np.array([1, 0, 0]) * np.pi / 2), [0, 0, 0])
		P2_SE3 = define_SE3(exp_so3(np.array([0, 1, 0]) * np.pi / 2), [0, 0, d / 2 + c1])
		
		P1_parameters = {"width": w, "depth": d, "height": h}
		P2_parameters = {"torus_radius": r_torus, "tube_radius": r_tube, "c1": c1}
		
		if 'color1' in config.keys():
			P1_color = config['color1']
		else:
			P1_color = None
		if 'color2' in config.keys():
			P2_color = config['color2']
		else:
			P2_color = None


		P1 = Box(SE3=P1_SE3, parameters=P1_parameters, color=P1_color)
		P2 = Torus_handle(SE3=P2_SE3, parameters=P2_parameters, color=P2_color)

		self.initialzier([P1, P2], transform=False)	


class Screw_Driver(Object):
    	
	def __init__(self, config):
		if config['random'] is True:
			r1 = random.uniform(config['r1']['min'], config['r1']['max'])
			h1 = random.uniform(config['h1']['min'], config['h1']['max'])

			r2 = random.uniform(config['r2']['min'], config['r2']['max'])
			h2 = random.uniform(config['h2']['min'], config['h2']['max'])
		else: 
			r1 = config['r1']
			h1 = config['h1']

			r2 = config['r2']
			h2 = config['h2']

		P1_SE3 = define_SE3(np.identity(3), [0, 0, 0])
		P2_SE3 = define_SE3(np.identity(3), [0, 0, h1 / 2 + h2 / 2])

		P1_parameters = {"radius": r1, "height": h1}
		P2_parameters = {"radius": r2, "height": h2}
 
		if 'color1' in config.keys():
			P1_color = config['color1']
		else:
			P1_color = None
		if 'color2' in config.keys():
			P2_color = config['color2']
		else:
			P2_color = None

		P1 = Cylinder(SE3=P1_SE3, parameters=P1_parameters, color=P1_color)
		P2 = Cylinder(SE3=P2_SE3, parameters=P2_parameters, color=P2_color)

		self.initialzier([P1, P2], transform='center')

class Hammer_Cylinder(Object):

	def __init__(self, config):
		if config['random'] is True:
			r1 = random.uniform(config['r1']['min'], config['r1']['max'])
			r2 = random.uniform(max(config['r2']['min'], r1), config['r2']['max'])
		
			h2 = random.uniform(config['h2']['min'], config['h2']['max'])
			if config['z2_offset'] and random.random() < 0.5:
				z2_offset = random.uniform(r2, r2 + 0.010)
				h1 = random.uniform(config['h1']['min'], config['h1']['max']) + z2_offset + r2
			else:
				z2_offset = 0
				h1 = random.uniform(config['h1']['min'], config['h1']['max'])


		else: 
			h1 = config['h1']
			h2 = config['h2']

			r1 = config['r1']
			r2 = config['r2']
			z2_offset = config['z2_offset']
		
		P1_SE3 = define_SE3(np.identity(3), [0, 0, 0])
		P2_SE3 = define_SE3(exp_so3(np.array([1, 0 ,0]) * np.pi / 2), [0, 0, h1 / 2 - z2_offset])

		P1_parameters = {"radius": r1, "height": h1}
		P2_parameters = {"radius": r2, "height": h2}

		if 'color1' in config.keys():
			P1_color = config['color1']
		else:
			P1_color = None
		if 'color2' in config.keys():
			P2_color = config['color2']
		else:
			P2_color = None

		P1 = Cylinder(SE3=P1_SE3, parameters=P1_parameters, color=P1_color)
		P2 = Cylinder(SE3=P2_SE3, parameters=P2_parameters, color=P2_color)
	
		self.initialzier([P1, P2], transform='center')

class Cup_with_Lid(Object):
    
	def __init__(self, config):
		if config['random'] is True:
			r1 = random.uniform(config['r1']['min'], config['r1']['max'])
			h1 = random.uniform(config['h1']['min'], config['h1']['max'])

			r_tube = random.uniform(config['r_tube']['min'], config['r_tube']['max'])
			r_torus = random.uniform(config['r_torus']['min'], min(config['r_torus']['max'], h1 / 2 - r_tube))
			c1 = random.uniform(0.3 * (-r_torus + r_tube), 0.5 * (r_torus - r_tube))
			z3 = random.uniform(0, h1 / 2 - (r_tube + r_torus))
			if random.random() < 0.3:
				z3 = 0 
		else: 
			r1 = config['r1']
			h1 = config['h1']
			t1 = config['t1']

			r_tube = config['r_tube']
			r_torus = config['r_torus']
			c1 = config['c1']
			z3 = config['z3']
		
		P1_SE3 = define_SE3(np.identity(3), [0, 0, 0])
		P2_SE3 = define_SE3(np.dot(exp_so3(np.array([1, 0, 0]) * np.pi / 2), exp_so3(np.array([0, 0, 1]) * np.pi)), [np.sqrt(r1**2 - r_tube**2) + c1, 0, z3])
		
		P1_parameters = {"radius": r1, "height": h1}
		P2_parameters = {"torus_radius": r_torus, "tube_radius": r_tube, "c1": c1}
		
		if 'color1' in config.keys():
			P1_color = config['color1']
		else:
			P1_color = None
		if 'color2' in config.keys():
			P2_color = config['color2']
		else:
			P2_color = None

		P1 = Cylinder(SE3=P1_SE3, parameters=P1_parameters, color=P1_color)
		P2 = Torus_handle(SE3=P2_SE3, parameters=P2_parameters, color=P2_color)

		self.initialzier([P1, P2], transform=False)	


load_object = {
	# please add object_class here
	"box": Box_Object,
	"cone": Cone_Object,
	"cylinder": Cylinder_Object,
	"torus": Torus_Object,
	"ellipsoid": Ellipsoid_Object,
	"superquadric": Superquadric_Object,
	"extended_sq": ExtendedSQ_Object,
	"deformable_superquadric": ExtendedSQ_Object,
	"bottle_cone": Bottle_Cone,
	"dumbbell": Dumbbell,
	"hammer_cylinder": Hammer_Cylinder,
	"screw_driver": Screw_Driver,
	"padlock": Padlock,
	"cup_with_lid": Cup_with_Lid,
}