import numpy as np
import os
import argparse
import torch
import open3d as o3d
from datetime import datetime
from copy import deepcopy
from sklearn.preprocessing import normalize

from functions.pc_preprocess import process_pc, normalize_numpy_pc
from functions.utils_numpy import quaternions_to_rotation_matrices, define_SE3, inverse_SE3
from functions.primitives import Superquadric, DeformableSuperquadric
from functions.object_class import Object
from functions.data_reader import create_object_from_data
from functions.iou_calculator import iou
from models import load_pretrained

def pred_to_parameters(pred_fit, normalizer, pnts_frame):

    # SE3
    SE3 = define_SE3(np.squeeze(quaternions_to_rotation_matrices(np.expand_dims(pred_fit[3:7], axis=0))), normalizer * pred_fit[:3])
    SE3 = np.dot(pnts_frame, SE3)  
    
    # parameters
    parameters = dict()
    parameters['a1'] = pred_fit[7] * normalizer
    parameters['a2'] = pred_fit[8] * normalizer
    parameters['a3'] = pred_fit[9] * normalizer
    parameters['e1'] = pred_fit[10]
    parameters['e2'] = pred_fit[11]
    if len(pred_fit) > 12:
        parameters['k'] = pred_fit[12]
        parameters['b'] = pred_fit[13]
        parameters['cos_alpha'] = pred_fit[14]
        parameters['sin_alpha'] = pred_fit[15]

    return SE3, parameters


def show_geometries(geometries, title, footer=None):
    """Open3D window visualization (requires a display)."""
    geometries = [g for g in geometries if g is not None]
    if not geometries:
        return
    print(f"Opening viewer: {title}")
    if footer:
        print(footer)
    o3d.visualization.draw_geometries(geometries, window_name=title)


def format_segmentation_summary(pcd_seg, seg_pnts_list, source, gt_num_primitives=None, gt_types=None):
    labels, counts = np.unique(pcd_seg, return_counts=True)
    label_stats = ", ".join(
        f"class {int(label)}={int(count)} pts" for label, count in zip(labels, counts)
    )
    fitted = []
    dropped = []
    for label in range(int(np.max(pcd_seg)) + 1):
        count = int(np.sum(pcd_seg == label))
        if count > 100:
            fitted.append(f"class {label} ({count} pts)")
        elif count > 0:
            dropped.append(f"class {label} ({count} pts)")

    lines = [
        f"  [{source}] {len(labels)} class(es) in partial point cloud: {label_stats}",
        f"  [{source}] {len(seg_pnts_list)} segment(s) sent to SQNet after >100 pt filter: "
        + (", ".join(fitted) if fitted else "none"),
    ]
    if dropped:
        lines.append(f"  [{source}] dropped small segments: {', '.join(dropped)}")
    if gt_num_primitives is not None:
        gt_line = f"  [GT] object composed of {gt_num_primitives} primitive(s)"
        if gt_types:
            gt_line += f": {', '.join(gt_types)}"
        lines.append(gt_line)
    return "\n".join(lines)


def offset_mesh(mesh, translation):
    mesh_copy = deepcopy(mesh)
    mesh_copy.translate(np.asarray(translation, dtype=float), relative=True)
    return mesh_copy


def combined_extent(geometries):
    min_bound = np.min([g.get_min_bound() for g in geometries], axis=0)
    max_bound = np.max([g.get_max_bound() for g in geometries], axis=0)
    return max_bound - min_bound


def make_side_by_side(gt_mesh, pred_mesh, gap_factor=1.2):
    gt_copy = deepcopy(gt_mesh)
    pred_copy = deepcopy(pred_mesh)
    extent = combined_extent([gt_copy, pred_copy])
    offset = gap_factor * extent[0]
    gt_copy = offset_mesh(gt_copy, [-offset / 2, 0, 0])
    pred_copy = offset_mesh(pred_copy, [offset / 2, 0, 0])
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=max(0.1, 0.4 * extent[0]), origin=[0, 0, 0]
    )
    return [gt_copy, pred_copy, frame]


def make_wireframe_overlay(gt_mesh, pred_mesh):
    gt_lines = o3d.geometry.LineSet.create_from_triangle_mesh(deepcopy(gt_mesh))
    gt_lines.paint_uniform_color([0.2, 0.2, 0.2])
    return [gt_lines, deepcopy(pred_mesh)]


def show_prediction_comparisons(gt_mesh, pred_mesh, file_name, model_label, viz_mode):
    if viz_mode == "minimal":
        return

    if viz_mode in ("both", "side_by_side", "wireframe"):
        show_geometries([deepcopy(pred_mesh)], f"{file_name} | {model_label} prediction")

    if viz_mode in ("both", "side_by_side"):
        show_geometries(
            make_side_by_side(gt_mesh, pred_mesh),
            f"{file_name} | {model_label} vs GT (side by side)",
        )

    if viz_mode in ("both", "wireframe"):
        show_geometries(
            make_wireframe_overlay(gt_mesh, pred_mesh),
            f"{file_name} | {model_label} vs GT (wireframe overlay)",
        )

    if viz_mode == "legacy":
        show_geometries(
            [deepcopy(gt_mesh), deepcopy(pred_mesh)],
            f"{file_name} | {model_label} + ground truth",
        )


if __name__ == '__main__':

    #########################################################################
    ########################### Initial Settings ############################
    #########################################################################

    # argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--object', type=str)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--device', default=0)
    parser.add_argument('--run', default=None)
    parser.add_argument('--iou', action='store_true')
    parser.add_argument(
        '--use-gt-segmentation',
        action='store_true',
        help='Skip SegNet and split the point cloud using ground-truth membership labels.',
    )
    parser.add_argument(
        '--skip-dsqnet',
        action='store_true',
        help='Skip DSQNet (only run SQNet shape fitting).',
    )
    parser.add_argument(
        '--no-visualize',
        action='store_true',
        help='Disable Open3D viewer windows (useful on headless servers).',
    )
    parser.add_argument(
        '--viz-mode',
        type=str,
        default='both',
        choices=['both', 'side_by_side', 'wireframe', 'legacy', 'minimal'],
        help='Comparison visualization style for prediction vs ground truth.',
    )
    args = parser.parse_args()

    # configuration
    if args.device == 'cpu' or not torch.cuda.is_available():
        device = 'cpu'
    else:
        device = f'cuda:{args.device}'
    if args.run is not None:
        run_id = args.run
    else:
        run_id = datetime.now().strftime('%Y%m%d-%H%M')
    example_object = args.object
    example_idx = args.index
    visualize = not args.no_visualize

    # pre-trained networks
    seg_config = ('segnet_config/example', 'segnet_config.yml', 'model_best.pkl', {})
    sq_config = ('sqnet_config/example', 'sqnet_config.yml', 'model_best.pkl', {})
    dsq_config = ('dsqnet_config/example', 'dsqnet_config.yml', 'model_best.pkl', {})
    root = 'pretrained/'

    # evaluation dataset configuration
    data_path = 'datasets/evaluation_dataset'

    # parameters
    num_processed_pc_points = 1000
    num_fitting_pc_points = 300

    # color parameters
    prim_clr_dict = [
        [0, 98, 156], 
        [199, 202, 86], 
        [208, 45, 35], 
        [255, 106, 110],
        [177, 39, 99],
        [255, 173, 79]
    ]
    seg_clr_dict = [
        [129, 228, 189],
        [251, 208, 210],
        [242, 230, 81],
        [114, 47, 55],
        [101, 141, 198],
        [248, 212, 152]
    ]
    pnt_clr = [83, 86, 90]

    #########################################################################
    #################### Load Pre-trained Models ############################
    #########################################################################

    # load configuration 
    seg_identifier, seg_config_file, seg_ckpt_file, seg_kwargs = seg_config
    sq_identifier, sq_config_file, sq_ckpt_file, sq_kwargs = sq_config
    dsq_identifier, dsq_config_file, dsq_ckpt_file, dsq_kwargs = dsq_config

    # Load pretrained models (only what this run needs)
    seg_model = None
    if not args.use_gt_segmentation:
        seg_model, _ = load_pretrained(
            seg_identifier, seg_config_file, seg_ckpt_file, root=root, **seg_kwargs
        )
        seg_model.to(device)

    sqnet_model, _ = load_pretrained(
        sq_identifier, sq_config_file, sq_ckpt_file, root=root, **sq_kwargs
    )
    sqnet_model.to(device)

    dsqnet_model = None
    dsq_config_path = os.path.join(root, dsq_identifier, dsq_config_file)
    dsq_ckpt_path = os.path.join(root, dsq_identifier, dsq_ckpt_file)
    if not args.skip_dsqnet:
        if os.path.exists(dsq_config_path) and os.path.exists(dsq_ckpt_path):
            dsqnet_model, _ = load_pretrained(
                dsq_identifier, dsq_config_file, dsq_ckpt_file, root=root, **dsq_kwargs
            )
            dsqnet_model.to(device)
        else:
            print(
                "DSQNet checkpoint not found; continuing with SQNet only. "
                "Add pretrained/dsqnet_config/example/ or pass --skip-dsqnet."
            )
            args.skip_dsqnet = True

    #########################################################################
    ############################# Evaluation ################################
    #########################################################################

    # object number
    file_number = f"{example_idx + 90:04}"
    viewpoint = np.random.choice(14, size=14, replace = False)

    viewnum = 0
    breaknum = 0
    while True:
        
        # terminate
        if breaknum == 3:
            break

        # viewpoint
        file_viewpoint = f"{viewpoint[viewnum] + 1:02}"
        file_name = f'{example_object}_{file_number}_viewpoint_{file_viewpoint}'

        # load pointcloud
        data_name = os.path.join(data_path, example_object, f'{file_name}.npy')
        if os.path.exists(data_name):
            data = np.load(data_name, allow_pickle = True).item()
            viewnum = viewnum + 1
            breaknum = breaknum + 1
        else:
            viewnum = viewnum + 1
            continue
        pointcloud = data["partial_pc"]
        membership = data["membership"]
        gt_membership_type = []
        for primitive in data['primitives']:
            type_ = primitive['type']
            if type_ == 'cone' and primitive['parameters']['upper_radius'] > 0:
                type_ = 'truncated_cone'
            gt_membership_type.append(type_)

        print(f"Evaluating {file_name} (run_id={run_id})")

        # noise
        pointcloud = pointcloud.transpose()
        noise = np.random.uniform(-1, 1, size=pointcloud.shape)
        noise = normalize(noise, axis=0, norm='l2')
        noise_std = 0.001
        scale = np.random.normal(loc=0, scale=noise_std, size=(1, pointcloud.shape[1])).repeat(pointcloud.shape[0], axis=0)
        pointcloud = pointcloud + noise * scale
        pointcloud = pointcloud.transpose()

        # random idx
        perm = np.random.permutation(3000)
        permidx = perm[:1000]
        tempidx = np.zeros(3000, dtype=bool)
        tempidx[permidx] = True
        pointcloud = pointcloud[tempidx, :]
        membership = membership[tempidx]

        # ground truth mesh
        gt_mesh = create_object_from_data(data)
        gt_mesh_vis = deepcopy(gt_mesh.mesh)
        gt_mesh_vis.paint_uniform_color([0.7, 0.7, 0.7])

        # segment partially observed point cloud
        pcd = o3d.geometry.PointCloud(points = o3d.utility.Vector3dVector(pointcloud[:,0:3]))
        pcd, pcd_frame = process_pc(pcd, num_processed_pc_points=num_processed_pc_points)
        pcd_np_normalized, normalizer_seg = normalize_numpy_pc(np.asarray(pcd.points))
        pcd.points = o3d.utility.Vector3dVector(pcd_np_normalized)

        if args.use_gt_segmentation:
            pts_sub = pointcloud[:, 0:3]
            pcd_seg = membership.astype(int)
            seg_pnts_list = []
            for label in np.unique(pcd_seg):
                seg_pnts = pts_sub[np.squeeze(np.argwhere(pcd_seg == label)), :]
                if len(seg_pnts) > 100:
                    seg_pnts_list.append(seg_pnts)
        else:
            pnts_tensor = torch.tensor(
                np.expand_dims(np.array(pcd.points).transpose(), axis=0),
                dtype=torch.float,
            ).to(device)
            pred = seg_model(pnts_tensor)
            pcd_seg = np.argmax(np.squeeze(pred.detach().cpu().numpy()), axis=1)

            # seperate segmented pnts to seperate list
            pnts = np.asarray(pcd.points)
            pnts = pnts * normalizer_seg
            seg_pnts_list = []
            for p in range(max(pcd_seg)+1):
                seg_pnts = pnts[np.squeeze(np.argwhere(np.array(pcd_seg) == p)), :]
                if len(seg_pnts) > 100:
                    seg_pnts_list.append(seg_pnts)

        seg_pnts_list_prev = deepcopy(seg_pnts_list)
        for p, seg_pnts in enumerate(seg_pnts_list):
            seg_pnts_list_prev[p] = np.dot(np.concatenate((seg_pnts_list_prev[p], np.ones((seg_pnts_list_prev[p].shape[0], 1))), axis=1), inverse_SE3(pcd_frame).transpose())

        # segmentated pointcloud
        seg_color = np.zeros((len(pcd.points), 3))
        random_segclr_idx = np.random.choice(6, size=max(pcd_seg)+1, replace=False)
        for pnt in range(len(pointcloud)):
            seg_color[pnt, :] = seg_clr_dict[random_segclr_idx[pcd_seg[pnt]]]
        pcd.colors = o3d.utility.Vector3dVector(seg_color / 255)
        max_bound = pcd.get_max_bound()
        min_bound = pcd.get_min_bound()
        resolution = 20
        processed_pcd_ds = pcd.voxel_down_sample(np.max(max_bound-min_bound) / resolution)
        seg_pcd_vis = deepcopy(processed_pcd_ds)

        observed_pcd_vis = deepcopy(processed_pcd_ds)
        observed_pcd_vis.paint_uniform_color(np.array(pnt_clr) / 255)

        # process segmented pc (upsampling & frame transform)
        for p, seg_pnts in enumerate(seg_pnts_list):
            seg_pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(seg_pnts))
            seg_pcd, seg_pcd_frame = process_pc(seg_pcd, num_processed_pc_points=num_fitting_pc_points)
            seg_pnts = np.asarray(seg_pcd.points)
            seg_pnts_list[p] = (seg_pnts, seg_pcd_frame)
        
        # shape primitive recognition for sqnet and dsqnet
        sq_set = []
        dsq_set = []
        for p, seg_info in enumerate(seg_pnts_list):
            seg_pnts, normalizer = normalize_numpy_pc(seg_info[0])
            
            # sqnet
            pnts_tensor = torch.tensor(np.expand_dims(seg_pnts.transpose(), axis=0) , dtype=torch.float).to(device)
            pred_fit = sqnet_model(pnts_tensor)
            pred_fit = np.squeeze(pred_fit.detach().cpu().numpy())
            sq_SE3, sq_parameters = pred_to_parameters(pred_fit, normalizer, seg_info[1])
            sq_SE3 = np.dot(pcd_frame, sq_SE3)
            sq_fit = Superquadric(sq_SE3, sq_parameters)
            sq_set.append(sq_fit)

            if dsqnet_model is not None:
                pnts_tensor = torch.tensor(np.expand_dims(seg_pnts.transpose(), axis=0) , dtype=torch.float).to(device)
                pred_fit = dsqnet_model(pnts_tensor)
                pred_fit = np.squeeze(pred_fit.detach().cpu().numpy())
                dsq_SE3, dsq_parameters = pred_to_parameters(pred_fit, normalizer, seg_info[1])
                dsq_SE3 = np.dot(pcd_frame, dsq_SE3)
                dsq_fit = DeformableSuperquadric(dsq_SE3, dsq_parameters)
                dsq_set.append(dsq_fit)

        # predicted obj
        result_sq = Object(sq_set, transform=False)

        # fitted objects coloring
        segment_number = np.max(pcd_seg) + 1
        random_clr_idx = np.random.choice(6, size=segment_number, replace=False)
        if dsq_set:
            result_dsq = Object(dsq_set, transform=False)
            for p, primitive in enumerate(result_dsq.primitives):
                type_ = primitive.type
                primitive.mesh.paint_uniform_color(np.array(prim_clr_dict[random_clr_idx[p]]) / 255)
                if p == 0:
                    result_mesh_dsqnet_colored = primitive.mesh
                else:
                    result_mesh_dsqnet_colored = result_mesh_dsqnet_colored + primitive.mesh

        else:
            result_mesh_dsqnet_colored = None

        # fitted objects coloring
        for p, primitive in enumerate(result_sq.primitives):
            type_ = primitive.type
            primitive.mesh.paint_uniform_color(np.array(prim_clr_dict[random_clr_idx[p]]) / 255)
            if p == 0:
                result_mesh_sqnet_colored = primitive.mesh
            else:
                result_mesh_sqnet_colored = result_mesh_sqnet_colored + primitive.mesh

        if visualize:
            seg_source = "GT membership" if args.use_gt_segmentation else "SegNet"
            seg_summary = format_segmentation_summary(
                pcd_seg,
                seg_pnts_list,
                seg_source,
                gt_num_primitives=len(data["primitives"]),
                gt_types=gt_membership_type,
            )
            show_geometries([gt_mesh_vis], f"{file_name} | ground truth")
            show_geometries(
                [seg_pcd_vis],
                f"{file_name} | segmentation",
                footer=seg_summary,
            )
            show_geometries([observed_pcd_vis], f"{file_name} | observed point cloud")
            show_prediction_comparisons(
                gt_mesh_vis,
                result_mesh_sqnet_colored,
                file_name,
                "SQNet",
                args.viz_mode,
            )
            if result_mesh_dsqnet_colored is not None:
                show_prediction_comparisons(
                    gt_mesh_vis,
                    result_mesh_dsqnet_colored,
                    file_name,
                    "DSQNet",
                    args.viz_mode,
                )

        # calculate iou
        if args.iou:
            iou_sqnet = iou(gt_mesh.mesh, result_sq.mesh)
            print(f"[{file_name}] SQNet iou value: {iou_sqnet}")
            if dsq_set:
                iou_dsqnet = iou(gt_mesh.mesh, result_dsq.mesh)
                print(f"[{file_name}] DSQNet iou value: {iou_dsqnet}")

