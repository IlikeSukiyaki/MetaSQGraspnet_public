import numpy as np
import open3d as o3d
import warnings


def _get_fcl():
    """Lazy import: python-fcl is only required for collision meshes (grasping)."""
    try:
        import fcl
    except ImportError as exc:
        raise ImportError(
            "python-fcl is required for collision meshes. Install with: pip install python-fcl"
        ) from exc
    return fcl

class Base_primitives:
	def transform_mesh(self):
		self.mesh.compute_vertex_normals()
		self.mesh.paint_uniform_color(self.color)
		self.mesh.transform(self.SE3)

class Cylinder(Base_primitives):
	def __init__(self, SE3, parameters, color=[0.8, 0.8, 0.8]):
		self.type = 'cylinder'
		self.SE3 = SE3
		self.parameters = parameters
		self.color = color
		self.mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=self.parameters['radius'], height=self.parameters['height'], resolution=100)
		self.transform_mesh()

class Box(Base_primitives):
	def __init__(self, SE3, parameters, color=[0.8, 0.8, 0.8]):
		self.type = 'box'
		self.SE3 = SE3
		self.parameters = parameters
		self.color = color
		self.mesh = o3d.geometry.TriangleMesh.create_box(width=self.parameters['width'], depth=self.parameters['height'], height=self.parameters['depth']) # change is intentional!
		self.mesh.translate([-self.parameters['width']/2, -self.parameters['depth']/2, -self.parameters['height']/2])
		self.transform_mesh()

class Cone(Base_primitives):
	def __init__(self, SE3, parameters, color=[0.8, 0.8, 0.8]):
		if parameters['upper_radius'] >= parameters['lower_radius']:
			warnings.warn("Set upper_radius smaller than lower_radius")
		self.type = 'cone'
		self.SE3 = SE3
		self.parameters = parameters
		self.color = color
		if parameters['upper_radius'] == 0:
			self.mesh = o3d.geometry.TriangleMesh.create_cone(radius=self.parameters['lower_radius'], height=self.parameters['height'], resolution = 100)
			self.mesh.translate(np.array([0, 0, - self.parameters['height'] / 2]))
		else:
			self.mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=self.parameters['lower_radius'], height=self.parameters['height'], resolution=100, split=1)
			vertices = np.asarray(self.mesh.vertices)
			upper_plane_ver = np.argwhere(vertices[:, 2] > self.parameters['height'] / 2 - 1e-10)
			upper_plane_ver = np.delete(upper_plane_ver, np.argwhere(upper_plane_ver == 0)) # remove center vertex

			vertices[upper_plane_ver, 0:2] = vertices[upper_plane_ver, 0:2] * self.parameters['upper_radius'] / self.parameters['lower_radius']
			self.mesh.vertices = o3d.utility.Vector3dVector(vertices)
		self.transform_mesh()

class Torus(Base_primitives):
	
	def __init__(self, SE3, parameters, color=[0.8, 0.8, 0.8], collisionBox=False):
		self.type = 'torus'
		self.SE3 = SE3
		self.parameters = parameters
		self.color = color
		self.mesh = o3d.geometry.TriangleMesh.create_torus(self.parameters['torus_radius'], self.parameters['tube_radius'], 100, 100)

		vertices_numpy = np.asarray(self.mesh.vertices)
		# eta = np.arcsin(vertices_numpy[:,2:3])
		normxy = np.sqrt(vertices_numpy[:,1:2]**2 + vertices_numpy[:,0:1]**2) - self.parameters['torus_radius']
		eta = np.arctan2(vertices_numpy[:,2:3], normxy)
		omega = np.arctan2(vertices_numpy[:,1:2], vertices_numpy[:,0:1])	

		outer_indices = (np.squeeze(np.logical_or(omega < -self.parameters['c1'], omega > self.parameters['c1']))).nonzero()
		self.mesh.remove_vertices_by_index(outer_indices[0])

		boundary_edges = np.asarray(self.mesh.get_non_manifold_edges(allow_boundary_edges = False))
		boundary_vertices = np.reshape(boundary_edges, -1)
		boundary_vertices = np.unique(boundary_vertices)

		vertices = np.asarray(self.mesh.vertices)
		for boundary_vertex in boundary_vertices:
			normxy_b = np.linalg.norm(vertices[boundary_vertex,0:2]) - self.parameters['torus_radius']
			eta_b = np.arctan2(vertices[boundary_vertex, 2], normxy_b)
			vertices[boundary_vertex, 0:2] = [(self.parameters['torus_radius'] + self.parameters['tube_radius'] * np.cos(eta_b)) * np.cos(self.parameters['c1']), np.sign(vertices[boundary_vertex, 1]) * (self.parameters['torus_radius'] + self.parameters['tube_radius'] * np.cos(eta_b)) * np.sin(abs(self.parameters['c1']))]
		
		centers = np.array([[self.parameters['torus_radius'] * np.cos(self.parameters['c1']), self.parameters['torus_radius'] * np.sin(self.parameters['c1']), 0], [self.parameters['torus_radius'] * np.cos(-self.parameters['c1']), self.parameters['torus_radius'] * np.sin(-self.parameters['c1']), 0]])
		vertices = np.concatenate((vertices, centers), axis=0)
		centers_ind = [vertices.shape[0] - 2, vertices.shape[0] - 1]

		triangles = np.asarray(self.mesh.triangles)
		positive_boundary_edges = boundary_edges[np.squeeze(np.argwhere(vertices[boundary_edges[:, 0], 1]>=0))]
		negative_boundary_edges = boundary_edges[np.squeeze(np.argwhere(vertices[boundary_edges[:, 0], 1]<0))]
		boundary_edges = [positive_boundary_edges, negative_boundary_edges]
		for section in range(2):
			plane_triangles = np.concatenate((boundary_edges[section], np.ones((boundary_edges[section].shape[0], 1)) * centers_ind[section]), axis = 1).astype(int)
			plane_normals = np.cross(vertices[plane_triangles[:, 1]] - vertices[plane_triangles[:, 0]], vertices[plane_triangles[:, 2]] - vertices[plane_triangles[:, 1]])

			opposite_normal_ind = np.squeeze(np.argwhere(plane_normals[:, 0] > 0))
			plane_triangles[opposite_normal_ind] = np.flip(plane_triangles[opposite_normal_ind], -1)

			triangles = np.append(triangles, plane_triangles, axis = 0)			
	
		self.mesh.vertices = o3d.utility.Vector3dVector(vertices)
		self.mesh.triangles = o3d.utility.Vector3iVector(triangles)
		
		self.transform_mesh()

class Torus_handle(Base_primitives):
	
	def __init__(self, SE3, parameters, color=[0.8, 0.8, 0.8], collisionBox=False):
		self.type = 'torus'
		self.SE3 = SE3
		self.parameters = parameters
		self.color = color
		self.mesh = o3d.geometry.TriangleMesh.create_torus(self.parameters['torus_radius'], self.parameters['tube_radius'], 100, 100)
		
		if parameters['c1'] > - (self.parameters['torus_radius'] + self.parameters['tube_radius']) and parameters['c1'] < (self.parameters['torus_radius'] + self.parameters['tube_radius']):
			bbox = o3d.geometry.AxisAlignedBoundingBox()
			bbox.min_bound = [- (self.parameters['torus_radius'] + self.parameters['tube_radius'])] * 3
			bbox.max_bound = [self.parameters['c1'], (self.parameters['torus_radius'] + self.parameters['tube_radius']), (self.parameters['torus_radius'] + self.parameters['tube_radius'])]
			self.mesh = self.mesh.crop(bbox)
			boundary_edges = np.asarray(self.mesh.get_non_manifold_edges(allow_boundary_edges = False))
			boundary_vertices = np.reshape(boundary_edges, -1)
			boundary_vertices = np.unique(boundary_vertices)

			for boundary_vertex in boundary_vertices.tolist():
				vertex_pnt = self.mesh.vertices[boundary_vertex]
				sinv = vertex_pnt[2] / self.parameters['tube_radius']
				cosv = (np.linalg.norm(vertex_pnt[0:2]) - self.parameters['torus_radius']) / self.parameters['tube_radius']
				cosu_proj = self.parameters['c1'] / (self.parameters['torus_radius'] + self.parameters['tube_radius'] * cosv)
				if cosu_proj > 1.0:
					u_proj = 0
				else:
					u_proj = np.arccos(cosu_proj)
				self.mesh.vertices[boundary_vertex] = [self.parameters['c1'], (self.parameters['torus_radius'] + self.parameters['tube_radius'] * cosv) * np.sin(np.sign(vertex_pnt[1]) * u_proj), self.parameters['tube_radius'] * sinv] 
			
			vertices = np.asarray(self.mesh.vertices)

			if self.parameters['c1'] > self.parameters['torus_radius'] - self.parameters['tube_radius'] or self.parameters['c1'] < -self.parameters['torus_radius'] + self.parameters['tube_radius']:
				vertices = np.concatenate((vertices, np.array([[self.parameters['c1'], 0, 0]])), axis=0)
				center_ver_ind = vertices.shape[0] - 1

				triangles = np.asarray(self.mesh.triangles)
				plane_triangles = np.concatenate((boundary_edges, np.ones((boundary_edges.shape[0], 1)) * center_ver_ind), axis = 1).astype(int)
				plane_normals = np.cross(vertices[plane_triangles[:, 1]] - vertices[plane_triangles[:, 0]], vertices[plane_triangles[:, 2]] - vertices[plane_triangles[:, 1]])
				plane_triangles = np.where(np.repeat(np.transpose([plane_normals[:, 2]]), 3, axis = 1) > 0, plane_triangles, np.transpose([plane_triangles[:, 1], plane_triangles[:, 0], plane_triangles[:, 2]]))

				opposite_normal_ind = np.squeeze(np.argwhere(plane_normals[:, 0] > 0))
				plane_triangles[opposite_normal_ind] = np.flip(plane_triangles[opposite_normal_ind], 1)

				triangles = np.append(triangles, plane_triangles, axis = 0)
			else:
				centers = np.array([[self.parameters['c1'], np.sqrt(self.parameters['torus_radius']**2 - self.parameters['c1']**2), 0], [self.parameters['c1'], -np.sqrt(self.parameters['torus_radius']**2 - self.parameters['c1']**2), 0]])
				vertices = np.concatenate((vertices, centers), axis=0)
				centers_ind = [vertices.shape[0] - 2, vertices.shape[0] - 1]

				triangles = np.asarray(self.mesh.triangles)
				positive_boundary_edges = boundary_edges[np.squeeze(np.argwhere(vertices[boundary_edges[:, 0], 1]>=0))]
				negative_boundary_edges = boundary_edges[np.squeeze(np.argwhere(vertices[boundary_edges[:, 0], 1]<0))]
				boundary_edges = [positive_boundary_edges, negative_boundary_edges]
				for section in range(2):
					plane_triangles = np.concatenate((boundary_edges[section], np.ones((boundary_edges[section].shape[0], 1)) * centers_ind[section]), axis = 1).astype(int)
					plane_normals = np.cross(vertices[plane_triangles[:, 1]] - vertices[plane_triangles[:, 0]], vertices[plane_triangles[:, 2]] - vertices[plane_triangles[:, 1]])
					plane_triangles = np.where(np.repeat(np.transpose([plane_normals[:, 2]]), 3, axis = 1) > 0, plane_triangles, np.transpose([plane_triangles[:, 1], plane_triangles[:, 0], plane_triangles[:, 2]]))

					opposite_normal_ind = np.squeeze(np.argwhere(plane_normals[:, 0] > 0))
					plane_triangles[opposite_normal_ind] = np.flip(plane_triangles[opposite_normal_ind], -1)

					triangles = np.append(triangles, plane_triangles, axis = 0)			
	
			self.mesh.vertices = o3d.utility.Vector3dVector(vertices)
			self.mesh.triangles = o3d.utility.Vector3iVector(triangles)
		
		
		self.transform_mesh()

class Superquadric(Base_primitives):
	def __init__(self, SE3, parameters, resolution=10, color=[0.8, 0.8, 0.8], collisionBox=False): 
		self.type = 'superquadric'
		self.SE3 = SE3
		self.parameters = parameters
		self.resolution = resolution
		self.color = color
		self.mesh = mesh_superquadric(self.parameters, self.SE3, resolution=self.resolution)

		if collisionBox:
			fcl = _get_fcl()
			self.collisionBox = fcl.BVHModel()
			self.collisionBox.beginModel(len(self.mesh.vertices), len(self.mesh.triangles))
			self.collisionBox.addSubModel(np.asarray(self.mesh.vertices), np.asarray(self.mesh.triangles))
			self.collisionBox.endModel()
			self.collisionBox = fcl.CollisionObject(self.collisionBox, fcl.Transform())

		self.transform_mesh()

class ExtendedSuperquadric(Base_primitives):
	def __init__(self, SE3, parameters, resolution=10, color=[0.8, 0.8, 0.8], collisionBox=False):
		self.type = 'extended_sq'
		self.SE3 = SE3
		self.parameters = parameters
		self.resolution = resolution
		self.color = color
		self.mesh = mesh_extended_superquadric(self.parameters, self.SE3, resolution=self.resolution)

		if collisionBox:
			fcl = _get_fcl()
			self.collisionBox = fcl.BVHModel()
			self.collisionBox.beginModel(len(self.mesh.vertices), len(self.mesh.triangles))
			self.collisionBox.addSubModel(np.asarray(self.mesh.vertices), np.asarray(self.mesh.triangles))
			self.collisionBox.endModel()
			self.collisionBox = fcl.CollisionObject(self.collisionBox, fcl.Transform())

		self.transform_mesh()

def mesh_superquadric(parameters, SE3, resolution=10):

	assert SE3.shape == (4, 4)

	# parameters
	a1 = parameters['a1']
	a2 = parameters['a2']
	a3 = parameters['a3']
	e1 = parameters['e1']
	e2 = parameters['e2']
	R = SE3[0:3, 0:3]
	t = SE3[0:3, 3:]

	# make grids
	mesh = o3d.geometry.TriangleMesh.create_sphere(radius = 1, resolution = resolution)
	vertices_numpy = np.asarray(mesh.vertices)
	eta = np.arcsin(vertices_numpy[:,2:3])
	omega = np.arctan2(vertices_numpy[:,1:2], vertices_numpy[:,0:1])

	# make new vertices
	x = a1 * fexp(np.cos(eta), e1) * fexp(np.cos(omega), e2)
	y = a2 * fexp(np.cos(eta), e1) * fexp(np.sin(omega), e2)
	z = a3 * fexp(np.sin(eta), e1)

	# reconstruct point matrix
	points = np.concatenate((x, y, z), axis=1)

	mesh.vertices = o3d.utility.Vector3dVector(points)

	return mesh

def mesh_extended_superquadric(parameters, SE3, resolution=10):
	
	assert SE3.shape == (4, 4)

	# parameters
	a1 = parameters['a1']
	a2 = parameters['a2']
	a3 = parameters['a3']
	e1 = parameters['e1']
	e2 = parameters['e2']
	if 'k' in parameters.keys():
		k = parameters['k']
	if 'b' in parameters.keys():
		b = parameters['b'] / np.maximum(a1, a2)
		cos_alpha = parameters['cos_alpha']
		sin_alpha = parameters['sin_alpha']
		alpha = np.arctan2(sin_alpha, cos_alpha)

	# make grids
	mesh = o3d.geometry.TriangleMesh.create_sphere(radius = 1, resolution = resolution)
	vertices_numpy = np.asarray(mesh.vertices)
	eta = np.arcsin(vertices_numpy[:,2:3])
	omega = np.arctan2(vertices_numpy[:,1:2], vertices_numpy[:,0:1])

	# make new vertices
	x = a1 * fexp(np.cos(eta), e1) * fexp(np.cos(omega), e2)
	y = a2 * fexp(np.cos(eta), e1) * fexp(np.sin(omega), e2)
	z = a3 * fexp(np.sin(eta), e1)

	points = np.concatenate((x, y, z), axis=1)
	mesh.vertices = o3d.utility.Vector3dVector(points)
	mesh.remove_duplicated_vertices()
	mesh.remove_degenerate_triangles()
	mesh = mesh.subdivide_midpoint(2)

	points = np.asarray(mesh.vertices)
	x = points[:, 0:1]
	y = points[:, 1:2]
	z = points[:, 2:3]

	# tampering
	if 'k' in parameters.keys():
		f_x = k / a3 * z + 1
		f_y = k / a3 * z + 1
		x = f_x * x
		y = f_y * y

	# bending
	if 'b' in parameters.keys():
		gamma = z * b
		r = np.cos(alpha - np.arctan2(y, x)) * np.sqrt(x ** 2 + y ** 2)
		R = 1 / b - np.cos(gamma) * (1 / b - r)
		x = x + np.cos(alpha) * (R - r)
		y = y + np.sin(alpha) * (R - r)
		z = np.sin(gamma) * (1 / b - r)

	# reconstruct point matrix
	points = np.concatenate((x, y, z), axis=1)

	mesh.vertices = o3d.utility.Vector3dVector(points)
	mesh.remove_duplicated_vertices()
	mesh.remove_degenerate_triangles()

	return mesh

def fexp(x, p):
	return np.sign(x)*(np.abs(x)**p)

gen_primitive = {
	"box": Box,
	"cone": Cone,
	"cylinder": Cylinder,
	"torus": Torus,
	"superquadric": Superquadric,
	"extended_sq": ExtendedSuperquadric,
	"deformable_superquadric": ExtendedSuperquadric,
}

gen_parameter = {
	"box": ['width', 'depth', 'height'],
	"cone": ['lower_radius', 'upper_radius', 'height'],
	"cylinder": ['radius', 'height'],
	"torus": ['torus_radius', 'tube_radius', 'c1'],
	"superquadric": ['a1', 'a2', 'a3', 'e1', 'e2'],
	"extended_sq": ['a1', 'a2', 'a3', 'e1', 'e2', 'k', 'b', 'cos_alpha', 'sin_alpha'],
	"deformable_superquadric": ['a1', 'a2', 'a3', 'e1', 'e2', 'k', 'b', 'cos_alpha', 'sin_alpha'],
}

# Legacy aliases retained for older imports and persisted dataset type strings.
DeformableSuperquadric = ExtendedSuperquadric
mesh_deformable_superquadric = mesh_extended_superquadric

