import numpy as np
import trimesh
from glob import glob
import os
import multiprocessing as mp
from multiprocessing import Pool
import argparse
import traceback

ROOT = 'shapenet/data/'

def farthest_point_sampling(points, num_samples):
    """Pure numpy FPS implementation."""
    N = points.shape[0]
    selected = np.zeros(num_samples, dtype=int)
    distances = np.full(N, np.inf)
    # Start from a random point
    selected[0] = np.random.randint(0, N)
    for i in range(1, num_samples):
        last = points[selected[i-1]]
        dist = np.sum((points - last) ** 2, axis=1)
        distances = np.minimum(distances, dist)
        selected[i] = np.argmax(distances)
    return points[selected]

def fps_sampling(path):
    try:
        out_file = path + '/fps_point_cloud_{}points.npz'.format(args.num_points)
        if os.path.exists(out_file):
            return

        off_path = path + '/isosurf_scaled.off'
        mesh = trimesh.load(off_path)

        # Sample dense first, then FPS downsample
        dense_points = np.array(mesh.sample(10000))
        point_cloud = farthest_point_sampling(dense_points, args.num_points)

        np.savez(out_file, point_cloud=point_cloud)
        print('Finished {}'.format(path))
    except Exception:
        print('Error with {}: {}'.format(path, traceback.format_exc()))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-num_points', type=int, required=True)
    args = parser.parse_args()

    paths = [p for p in glob(ROOT + '/*/*/')
             if not any(s in p for s in ['/models/', '/images/'])]
    print('Found {} models'.format(len(paths)))
    p = Pool(8)
    p.map(fps_sampling, paths)
    print('Done.')
