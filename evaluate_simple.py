import torch
import numpy as np
from glob import glob
import os
import argparse
import trimesh
from scipy.spatial import cKDTree

import models.local_model as model
import models.local_model_dropout as model_dropout

def get_grid_points(res=32):
    bb_min, bb_max = -0.5, 0.5
    points = np.mgrid[:res, :res, :res].T.reshape(-1, 3).astype(np.float32)
    points = bb_min + (bb_max - bb_min) * points / (res - 1)
    return points

def evaluate_model(net, split_file, data_path, num_points, use_fps,
                   res=32, num_samples=200, device='cuda'):
    net.eval()
    split = np.load(split_file)
    test_models = [s.lstrip('/') for s in split['test'] if '03001627' in s]
    test_models = test_models[:num_samples]

    grid_points = get_grid_points(res)
    iou_scores = []
    chamfer_scores = []

    with torch.no_grad():
        for model_path in test_models:
            try:
                path = os.path.join(data_path, model_path)

                # Load point cloud input
                if use_fps:
                    pc_file = os.path.join(path,
                        'fps_point_cloud_{}points.npz'.format(num_points))
                else:
                    pc_file = os.path.join(path,
                        'voxelized_point_cloud_32res_{}points.npz'.format(num_points))

                if not os.path.exists(pc_file):
                    continue

                pc_data = np.load(pc_file)
                point_cloud = pc_data['point_cloud'].astype(np.float32)

                # Voxelize input point cloud
                kdtree = cKDTree(grid_points)
                occ_input = np.zeros(res**3, dtype=np.float32)
                _, idx = kdtree.query(point_cloud)
                occ_input[idx] = 1.0
                inputs = torch.FloatTensor(
                    occ_input.reshape(res, res, res)).unsqueeze(0).to(device)

                # Prepare query grid coords (IF-Net format)
                query = grid_points.copy()
                query[:, 0], query[:, 2] = grid_points[:, 2].copy(), grid_points[:, 0].copy()
                query = 2 * query
                p = torch.FloatTensor(query).unsqueeze(0).to(device)

                # Forward pass
                logits = net(p, inputs)
                pred_occ = (logits.squeeze(0).cpu().numpy() > 0)

                # Load GT occupancy from boundary samples
                gt_file = os.path.join(path, 'boundary_0.15_samples.npz')
                if not os.path.exists(gt_file):
                    continue

                # Use voxelized GT occupancy
                gt_vox_file = os.path.join(path,
                    'voxelized_point_cloud_32res_3000points.npz')
                if not os.path.exists(gt_vox_file):
                    continue
                gt_data = np.load(gt_vox_file)
                gt_occ = np.unpackbits(
                    gt_data['compressed_occupancies'])[:res**3].astype(bool)

                # IoU
                intersection = np.sum(pred_occ & gt_occ)
                union = np.sum(pred_occ | gt_occ)
                if union > 0:
                    iou_scores.append(intersection / union)

                # Chamfer: pred surface vs GT point cloud
                pred_pts = grid_points[pred_occ]
                gt_pts = grid_points[gt_occ]
                if len(pred_pts) > 0 and len(gt_pts) > 0:
                    t1 = cKDTree(pred_pts)
                    t2 = cKDTree(gt_pts)
                    d1, _ = t1.query(gt_pts[:1000])
                    d2, _ = t2.query(pred_pts[:1000])
                    cd = (np.mean(d1**2) + np.mean(d2**2)) / 2
                    chamfer_scores.append(cd)

            except Exception as e:
                continue

    return {
        'iou_mean': np.mean(iou_scores) if iou_scores else 0,
        'iou_std': np.std(iou_scores) if iou_scores else 0,
        'chamfer_mean': np.mean(chamfer_scores) if chamfer_scores else 0,
        'chamfer_std': np.std(chamfer_scores) if chamfer_scores else 0,
        'n_samples': len(iou_scores)
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model_type', type=str, required=True)
    parser.add_argument('-checkpoint', type=str, required=True)
    parser.add_argument('-num_points', type=int, default=3000)
    parser.add_argument('-use_fps', action='store_true')
    parser.add_argument('-num_samples', type=int, default=200)
    args = parser.parse_args()

    device = torch.device('cuda')

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
    print('Model:   {}'.format(args.model_type))
    print('Points:  {}'.format(args.num_points))
    print('FPS:     {}'.format(args.use_fps))
    print('N:       {}'.format(results['n_samples']))
    print('IoU:     {:.4f} ± {:.4f}'.format(results['iou_mean'], results['iou_std']))
    print('Chamfer: {:.6f} ± {:.6f}'.format(results['chamfer_mean'], results['chamfer_std']))
