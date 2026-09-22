import torch
import numpy as np
from glob import glob
import os
import argparse
import trimesh
from scipy.spatial import cKDTree

import models.local_model as model
import models.local_model_dropout as model_dropout

def voxelize_point_cloud(point_cloud, res=32):
    """Convert point cloud to voxel grid for IF-Net input."""
    bb_min, bb_max = -0.5, 0.5
    grid_points = np.mgrid[:res, :res, :res].T.reshape(-1, 3)
    grid_points = bb_min + (bb_max - bb_min) * grid_points / (res - 1)
    kdtree = cKDTree(grid_points)
    occupancies = np.zeros(len(grid_points), dtype=np.int8)
    _, idx = kdtree.query(point_cloud)
    occupancies[idx] = 1
    return np.packbits(occupancies), grid_points, bb_min, bb_max

def chamfer_distance(pred_points, gt_points):
    """Compute bidirectional Chamfer Distance."""
    tree1 = cKDTree(pred_points)
    tree2 = cKDTree(gt_points)
    dist1, _ = tree1.query(gt_points)
    dist2, _ = tree2.query(pred_points)
    return (np.mean(dist1**2) + np.mean(dist2**2)) / 2

def compute_iou(pred_occ, gt_occ):
    """Compute volumetric IoU."""
    intersection = np.sum(pred_occ & gt_occ)
    union = np.sum(pred_occ | gt_occ)
    return intersection / union if union > 0 else 0.0

def evaluate_model(net, split_file, data_path, num_points, use_fps, 
                   res=32, num_samples=100, device='cuda'):
    net.eval()
    split = np.load(split_file)
    test_models = [s.lstrip('/') for s in split['test'] if '03001627' in s]
    test_models = test_models[:num_samples]

    chamfer_scores = []
    iou_scores = []

    with torch.no_grad():
        for model_path in test_models:
            try:
                path = os.path.join(data_path, model_path)

                # Load point cloud
                if use_fps:
                    pc_file = os.path.join(path, 
                        'fps_point_cloud_{}points.npz'.format(num_points))
                else:
                    pc_file = os.path.join(path,
                        'voxelized_point_cloud_32res_{}points.npz'.format(num_points))

                if not os.path.exists(pc_file):
                    continue

                pc_data = np.load(pc_file)
                point_cloud = pc_data['point_cloud']

                # Voxelize for IF-Net input
                compressed_occ, grid_points, bb_min, bb_max = \
                    voxelize_point_cloud(point_cloud, res)
                occ = np.unpackbits(compressed_occ)[:res**3]
                inputs = torch.FloatTensor(
                    occ.reshape(res, res, res)).unsqueeze(0).to(device)

                # Load GT mesh
                off_path = os.path.join(path, 'isosurf_scaled.off')
                if not os.path.exists(off_path):
                    continue
                gt_mesh = trimesh.load(off_path)
                gt_points = np.array(gt_mesh.sample(10000))

                # Prepare query points
                query_points = grid_points.copy()
                query_points[:, 0], query_points[:, 2] = \
                    grid_points[:, 2].copy(), grid_points[:, 0].copy()
                query_points = 2 * query_points
                p = torch.FloatTensor(query_points).unsqueeze(0).to(device)

                # Forward pass
                logits = net(p, inputs)
                pred_occ = (logits.squeeze(0).cpu().numpy() > 0).astype(bool)
                gt_occ_grid = np.zeros(res**3, dtype=bool)

                # Sample GT occupancy on grid
                from data_processing.implicit_waterproofing import implicit_waterproofing
                gt_occ_vals = implicit_waterproofing(gt_mesh, 
                    grid_points[:res**3])[0].astype(bool)

                # Chamfer: get predicted surface points
                pred_surface_idx = np.where(pred_occ)[0]
                if len(pred_surface_idx) > 0:
                    pred_pts = grid_points[pred_surface_idx]
                    cd = chamfer_distance(pred_pts, gt_points)
                    chamfer_scores.append(cd)

                # IoU
                iou = compute_iou(pred_occ, gt_occ_vals)
                iou_scores.append(iou)

            except Exception as e:
                print('Error with {}: {}'.format(model_path, e))
                continue

    return {
        'chamfer_mean': np.mean(chamfer_scores) if chamfer_scores else 0,
        'chamfer_std': np.std(chamfer_scores) if chamfer_scores else 0,
        'iou_mean': np.mean(iou_scores) if iou_scores else 0,
        'iou_std': np.std(iou_scores) if iou_scores else 0,
        'n_samples': len(iou_scores)
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model_type', type=str, 
                        choices=['ShapeNetPoints', 'ShapeNetPointsDropout'],
                        required=True)
    parser.add_argument('-checkpoint', type=str, required=True)
    parser.add_argument('-num_points', type=int, default=3000)
    parser.add_argument('-use_fps', action='store_true')
    parser.add_argument('-num_samples', type=int, default=100)
    args = parser.parse_args()

    device = torch.device('cuda')

    # Load model
    if args.model_type == 'ShapeNetPoints':
        net = model.ShapeNetPoints().to(device)
    else:
        net = model_dropout.ShapeNetPointsDropout().to(device)

    checkpoint = torch.load(args.checkpoint)
    net.load_state_dict(checkpoint['model_state_dict'])
    print('Loaded checkpoint: epoch {}'.format(checkpoint['epoch']))

    results = evaluate_model(
        net=net,
        split_file='shapenet/split_chairs.npz',
        data_path='shapenet/data/',
        num_points=args.num_points,
        use_fps=args.use_fps,
        num_samples=args.num_samples,
        device=device
    )

    print('\n=== Results ===')
    print('Model: {}'.format(args.model_type))
    print('Input points: {}'.format(args.num_points))
    print('FPS: {}'.format(args.use_fps))
    print('Samples evaluated: {}'.format(results['n_samples']))
    print('IoU:     {:.4f} ± {:.4f}'.format(
        results['iou_mean'], results['iou_std']))
    print('Chamfer: {:.6f} ± {:.6f}'.format(
        results['chamfer_mean'], results['chamfer_std']))