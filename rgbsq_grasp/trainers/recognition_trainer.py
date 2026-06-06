import os
import time

import torch
import numpy as np
import open3d as o3d

from metrics import get_metric, averageMeter
from loss import get_loss
from functions.utils_numpy import define_SE3, quaternions_to_rotation_matrices
from functions.primitives import Superquadric, DeformableSuperquadric, gen_primitive, gen_parameter

class RecognitionTrainer:
    """Trainer for a conventional iterative training of model"""

    def __init__(self, training_cfg, device):
        self.cfg = training_cfg
        self.device = device
        self.time_meter = averageMeter()
        self.loss_meter = {"train": averageMeter(), "val": averageMeter()}

        self.loss = get_loss(training_cfg, device=device)
        self.show_metric = False
        dict_metric = training_cfg.get("metric", None)
        if dict_metric is not None:
            self.metric = get_metric(dict_metric)
            self.show_metric = True

    def train(self, model, opt, d_dataloaders, writer):
        cfg = self.cfg
        logdir = writer.log_dir
        best_val_loss = np.inf
        iter = 0

        train_loader, val_loader = (
            d_dataloaders['training'], 
            d_dataloaders['validation']
        )

        for epoch in range(cfg.n_epoch):
            for x, x_gt, l_gt, y in train_loader:
                # training
                iter += 1

                model.train()
                x = x.to(self.device)
                x_gt = x_gt.to(self.device)
                l_gt = l_gt.to(self.device)

                start_ts = time.time()
                train_step_result = model.train_step(
                    x, x_gt = x_gt, y=y, l_gt=l_gt, optimizer=opt, loss_function=self.loss
                )
                
                self.time_meter.update(time.time() - start_ts)
                self.loss_meter["train"].update(train_step_result["loss"])

                if iter % cfg.print_interval == 0:
                    self.record_results(writer, iter, "train", train_step_result)
                    print(
                        f"[Training] Iter [{iter:d}] Avg Loss: {self.loss_meter['train'].avg:.4f} Elapsed time: {self.time_meter.sum:.4f}"
                    )
                    self.time_meter.reset()

                # save model
                if iter % cfg.save_interval == 0:
                    self.save_model(model, logdir, best=False, i_iter=iter)

                if iter % cfg.val_interval == 0:
                    # validation
                    model.eval()
                    j = 0
                    for val_x, val_x_gt, val_l_gt, val_y in val_loader:
                        j += 1
                        val_x = val_x.to(self.device)
                        val_x_gt = val_x_gt.to(self.device)
                        val_l_gt = val_l_gt.to(self.device)

                        val_step_result = model.validation_step(
                            val_x, x_gt = val_x_gt, y=val_y, l_gt = val_l_gt, loss_function=self.loss
                        )
                        
                        self.loss_meter["val"].update(val_step_result["loss"])
                        if self.show_metric:
                            pass

                    # record
                    self.record_results(writer, iter, "val", train_step_result)
                    val_loss = self.loss_meter["val"].avg
                    print(f"[Validation] Iter [{iter:d}] Avg Loss: {val_loss:.4f}")

                    # save model
                    if val_loss < best_val_loss:
                        self.save_model(model, logdir, best=True, i_iter=iter)
                        print(
                            f"[Validation] Iter [{iter:d}] best model saved {val_loss} <= {best_val_loss}"
                        )
                        best_val_loss = val_loss

        return model, best_val_loss

    def record_results(self, writer, i, tag, results):
        # record loss
        writer.add_scalar(f"loss/{tag}_loss", self.loss_meter[tag].avg, i)

        # record segmentation result
        if i % self.cfg["visualize_interval"] == 0:
            pc = results["pc"][0 : self.cfg["visualize_number"]]
            pc_gt = results["pc_gt"][0 : self.cfg["visualize_number"]]
            shape_pred = results["mesh"][0 : self.cfg["visualize_number"]]
            shape_gt = {key: val[0 : self.cfg["visualize_number"]] for key, val in results["mesh_gt"].items() if key != "parameters"}
            shape_gt_parameters = {key: val[0 : self.cfg["visualize_number"]] for key, val in results["mesh_gt"]["parameters"].items()}
            shape_gt["parameters"] = shape_gt_parameters

            pc_coordinated, color_pc = pc_with_coordinates(pc)
            pc_gt_coordinated, color_pc_gt = pc_with_coordinates(pc_gt)
            mesh_pred = mesh_from_prediction(shape_pred)
            mesh_gt = mesh_from_groundtruth(shape_gt)
            total_vertices, total_faces, total_colors = meshs_to_numpy(mesh_pred, mesh_gt, coordinate = True)

            # write to Tensorboard
            writer.add_mesh(
                f"{tag} pc", 
                vertices=pc_coordinated, 
                colors=color_pc, 
                global_step=i
            )
            writer.add_mesh(
                f"{tag} pc_gt", 
                vertices=pc_gt_coordinated, 
                colors=color_pc_gt, 
                global_step=i
            )
            writer.add_mesh(
                f"{tag} mesh & mesh_gt", 
                vertices=total_vertices, 
                faces=total_faces, 
                colors = total_colors, 
                global_step=i
            )

        # record metrics
        if self.show_metric:
            pass

    def save_model(self, model, logdir, best=False, i_iter=None, epoch=None):
        if best:
            pkl_name = "model_best.pkl"
        else:
            if i_iter is not None:
                pkl_name = f"model_iter_{i_iter}.pkl"
            else:
                pkl_name = f"model_epoch_{epoch}.pkl"
        state = {"epoch": epoch, "model_state": model.state_dict(), "iter": i_iter}
        save_path = os.path.join(logdir, pkl_name)
        torch.save(state, save_path)
        print(f"Model saved: {pkl_name}")

def pc_with_coordinates(pc):
    # make coordinate frame
    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.2, origin=[0, 0, 0]
    )
    pc_coor_repeat = np.tile(
        np.asarray(coordinate.vertices), 
        (pc.shape[0], 1, 1)
    )
    color_coor_repeat = np.tile(
        np.asarray(coordinate.vertex_colors), 
        (pc.shape[0], 1, 1)
    )

    # concatenate
    pc_total = np.concatenate((pc, pc_coor_repeat), axis=1)
    color_total = np.concatenate((128 * np.ones(np.shape(pc)), 255 * color_coor_repeat), axis=1)
    
    return pc_total, color_total

def mesh_from_prediction(shape):
    
    shape_position = shape[:, :3]
    shape_orientation = quaternions_to_rotation_matrices(shape[:, 3:7])
    shape_parameters = shape[:, 7:]
    
    meshes = []
    for idx in range(len(shape)):
        SE3 = define_SE3(shape_orientation[idx, :, :], shape_position[idx, :])
        parameters = dict()
        parameters['a1'] = shape_parameters[idx, 0]
        parameters['a2'] = shape_parameters[idx, 1]
        parameters['a3'] = shape_parameters[idx, 2]
        parameters['e1'] = shape_parameters[idx, 3]
        parameters['e2'] = shape_parameters[idx, 4]
        if shape_parameters.shape[1] > 5: # deformable superquadric
            parameters['k'] = shape_parameters[idx, 5]
            parameters['b'] = shape_parameters[idx, 6]
            parameters['cos_alpha'] = shape_parameters[idx, 7]
            parameters['sin_alpha'] = shape_parameters[idx, 8]
            mesh = DeformableSuperquadric(SE3, parameters).mesh
        else: # superquadric
            mesh = Superquadric(SE3, parameters).mesh

        meshes.append(mesh)

    return meshes

def mesh_from_groundtruth(shape):

    shape_type = shape["type"]
    shape_pose = shape["SE3"].numpy()
    shape_parameters_dict = shape["parameters"]
    
    meshes = []
    for idx in range(len(shape_type)):
        SE3 = shape_pose[idx, :, :]
        parameters = dict()
        for p, name in enumerate(gen_parameter[f"{shape_type[idx]}"]):
            parameters[f"{name}"] = shape_parameters_dict[f"param{p+1}"][idx].numpy()

        mesh = gen_primitive[f"{shape_type[idx]}"](SE3, parameters).mesh
        meshes.append(mesh)
    
    return meshes

def meshs_to_numpy(mesh1, mesh2, color1 = [0, 1, 1], color2 = [0, 0, 1], coordinate = False):
    
    if len(mesh1) != len(mesh2):
        raise ValueError('mesh1 and mesh2 do not have same batch size')
    
    total_pointclouds = []
    total_faces = []
    total_colors = []

    max_num_pointclouds = 0
    max_num_faces = 0

    for batch in range(len(mesh1)):

        # color painting
        mesh1[batch].paint_uniform_color(color1)
        mesh2[batch].paint_uniform_color(color2)

        # make coordinate frame
        coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.4, origin=[0, 0, 0]
        )

        # combine meshes
        if coordinate:
            mesh = mesh1[batch] + mesh2[batch] + coordinate
        else:
            mesh = mesh1[batch] + mesh2[batch]

        pointclouds = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        colors = 255 * np.asarray(mesh.vertex_colors)

        # append for batch
        total_pointclouds.append(pointclouds)
        total_faces.append(faces)
        total_colors.append(colors)

        # find the maximum dimension
        if pointclouds.shape[0] > max_num_pointclouds:
            max_num_pointclouds = pointclouds.shape[0]
        if faces.shape[0] > max_num_faces:
            max_num_faces = faces.shape[0]

    # matching dimension between batches for tensorboard
    for batch in range(len(mesh1)):
        diff_num_pointclouds = max_num_pointclouds - total_pointclouds[batch].shape[0]
        diff_num_faces = max_num_faces - total_faces[batch].shape[0]
        total_pointclouds[batch] = np.concatenate((total_pointclouds[batch], np.zeros((diff_num_pointclouds, 3))), axis=0)
        total_faces[batch] = np.concatenate((total_faces[batch], np.zeros((diff_num_faces, 3))), axis=0)
        total_colors[batch] = np.concatenate((total_colors[batch], np.zeros((diff_num_pointclouds, 3))), axis=0)

    return np.asarray(total_pointclouds), np.asarray(total_faces), np.asarray(total_colors)