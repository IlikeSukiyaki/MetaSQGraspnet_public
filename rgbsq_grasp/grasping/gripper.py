import numpy as np
from functions.utils_numpy import define_SE3, exp_so3
import open3d as o3d
import fcl
from functions.utils_numpy import get_SO3, inverse_SE3, define_SE3, exp_so3, transform_point, get_p
from functions.primitives import Box

class Gripper:
	def __init__(self, SE3, width=0, collisionBox=False):
		self.hand_SE3 = SE3
		self.gripper_width = width
		if width < 0:
			print("gripper width exceeds minimum width. gripper width is set to 0")
			self.gripper_width = 0
		if width > 0.08:
			print("gripper width exceeds maximum width. gripper width is set to 0.08")
			self.gripper_width = 0.08

		self.hand = o3d.io.read_triangle_mesh("grasping/assets/hand.ply")
		self.hand.compute_vertex_normals()
		self.hand.paint_uniform_color([0.9, 0.9, 0.9])
		self.finger1 = o3d.io.read_triangle_mesh("grasping/assets/finger.ply")
		self.finger1.compute_vertex_normals()
		self.finger1.paint_uniform_color([0.7, 0.7, 0.7])
		self.finger2 = o3d.io.read_triangle_mesh("grasping/assets/finger.ply")
		self.finger2.compute_vertex_normals()
		self.finger2.paint_uniform_color([0.7, 0.7, 0.7])

		self.finger1_M = define_SE3(np.identity(3), np.array([0, self.gripper_width/2, 0.1654/3]))
		self.finger2_M = define_SE3(exp_so3(np.asarray([0, 0, 1]) * np.pi), np.array([0, -self.gripper_width/2, 0.1654/3]))

		self.finger1_SE3 = np.dot(self.hand_SE3, self.finger1_M)
		self.finger2_SE3 = np.dot(self.hand_SE3, self.finger2_M)
			
		self.hand.transform(self.hand_SE3)
		self.finger1.transform(self.finger1_SE3)
		self.finger2.transform(self.finger2_SE3)
		self.mesh = self.hand + self.finger1 + self.finger2

		if collisionBox:

			self.handBox_M = define_SE3(np.identity(3), np.array([0, 0, 0.0287]))

			self.finger1Box1_M = define_SE3(np.identity(3), np.array([0, 0.0185 + self.gripper_width / 2, 0.0667]))
			self.finger1Box2_M = define_SE3(exp_so3(np.array([1, 0, 0]) * 0.54), np.array([0, 0.0165 + self.gripper_width / 2, 0.0827]))
			self.finger1Box3_M = define_SE3(np.identity(3), np.array([0, 0.0072 + self.gripper_width / 2, 0.1001]))

			self.finger2Box1_M = define_SE3(np.identity(3), np.array([0, - (0.0185 + self.gripper_width / 2), 0.0667]))
			self.finger2Box2_M = define_SE3(exp_so3(np.array([1, 0, 0]) * -0.54), np.array([0, - (0.0165 + self.gripper_width / 2), 0.0827]))
			self.finger2Box3_M = define_SE3(np.identity(3), np.array([0, - (0.0072 + self.gripper_width / 2), 0.1001]))

			self.azure_M = define_SE3(np.identity(3), np.array([0.064, 0, 0.0287]))

			self.M = {0: self.handBox_M, 1: self.finger1Box1_M, 2: self.finger1Box2_M, 3: self.finger1Box3_M, 4: self.finger2Box1_M, 5: self.finger2Box2_M, 6: self.finger2Box3_M, 7: self.azure_M}

			handBox_SE3 = np.dot(self.hand_SE3, self.handBox_M)
		
			finger1Box1_SE3 = np.dot(self.hand_SE3, self.finger1Box1_M)
			finger1Box2_SE3 = np.dot(self.hand_SE3, self.finger1Box2_M)
			finger1Box3_SE3 = np.dot(self.hand_SE3, self.finger1Box3_M)

			finger2Box1_SE3 = np.dot(self.hand_SE3, self.finger2Box1_M)
			finger2Box2_SE3 = np.dot(self.hand_SE3, self.finger2Box2_M)
			finger2Box3_SE3 = np.dot(self.hand_SE3, self.finger2Box3_M)

			# azure_SE3 = np.dot(self.hand_SE3, self.azure_M)

			handBox = fcl.CollisionObject(fcl.Box(0.058, 0.202, 0.0746), fcl.Transform(get_SO3(handBox_SE3), get_p(handBox_SE3)))

			finger1Box1 = fcl.CollisionObject(fcl.Box(0.021, 0.0142, 0.015), fcl.Transform(get_SO3(finger1Box1_SE3), get_p(finger1Box1_SE3)))
			finger1Box2 = fcl.CollisionObject(fcl.Box(0.021, 0.0072, 0.024), fcl.Transform(get_SO3(finger1Box2_SE3), get_p(finger1Box2_SE3)))
			finger1Box3 = fcl.CollisionObject(fcl.Box(0.0181, 0.0146, 0.018), fcl.Transform(get_SO3(finger1Box3_SE3), get_p(finger1Box3_SE3)))

			finger2Box1 = fcl.CollisionObject(fcl.Box(0.021, 0.0142, 0.015), fcl.Transform(get_SO3(finger2Box1_SE3), get_p(finger2Box1_SE3)))
			finger2Box2 = fcl.CollisionObject(fcl.Box(0.021, 0.0072, 0.024), fcl.Transform(get_SO3(finger2Box2_SE3), get_p(finger2Box2_SE3)))
			finger2Box3 = fcl.CollisionObject(fcl.Box(0.0181, 0.0146, 0.018), fcl.Transform(get_SO3(finger2Box3_SE3), get_p(finger2Box3_SE3)))

			# azureBox = fcl.CollisionObject(fcl.Box(0.07, 0.202, 0.0746), fcl.Transform(get_SO3(azure_SE3), get_p(azure_SE3)))

			# self.collisionBox = [handBox, finger1Box1, finger1Box2, finger1Box3, finger2Box1, finger2Box2, finger2Box3, azureBox]
			self.collisionBox = [handBox, finger1Box1, finger1Box2, finger1Box3, finger2Box1, finger2Box2, finger2Box3]

			box0 = Box(SE3=handBox_SE3, parameters={'width': 0.058, 'depth': 0.202, 'height': 0.0746}, color=[0, 1, 0])
			box1 = Box(SE3=finger1Box1_SE3, parameters={'width': 0.021, 'depth': 0.0142, 'height': 0.015}, color=[0, 0, 1])
			box2 = Box(SE3=finger1Box2_SE3, parameters={'width': 0.021, 'depth': 0.0072, 'height': 0.024}, color=[0, 0, 1])
			box3 = Box(SE3=finger1Box3_SE3, parameters={'width': 0.0181, 'depth': 0.0146, 'height': 0.018}, color=[0, 0, 1])
			box4 = Box(SE3=finger2Box1_SE3, parameters={'width': 0.021, 'depth': 0.0142, 'height': 0.015}, color=[0, 0, 1])
			box5 = Box(SE3=finger2Box2_SE3, parameters={'width': 0.021, 'depth': 0.0072, 'height': 0.024}, color=[0, 0, 1])
			box6 = Box(SE3=finger2Box3_SE3, parameters={'width': 0.0181, 'depth': 0.0146, 'height': 0.018}, color=[0, 0, 1])

			self.gripper_obj = [box0, box1, box2, box3, box4, box5, box6]