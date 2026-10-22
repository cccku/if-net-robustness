import os
import glob
import trimesh
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool

INPUT_PATH = 'shapenet/data'

def convert_and_scale(path):
    out_file = path + '/isosurf_scaled.off'
    if os.path.exists(out_file):
        return

    try:
        mesh = trimesh.load(path + '/model.obj', process=False, force='mesh')
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(
                [trimesh.Trimesh(vertices=m.vertices, faces=m.faces)
                 for m in mesh.geometry.values()]
            )

        # Scale: largest bounding box edge = 1, centered at origin
        total_size = (mesh.bounds[1] - mesh.bounds[0]).max()
        centers = (mesh.bounds[1] + mesh.bounds[0]) / 2
        mesh.apply_translation(-centers)
        mesh.apply_scale(1 / total_size)

        mesh.export(out_file)
        print('Finished {}'.format(path))
    except Exception as e:
        print('Error with {}: {}'.format(path, e))

if __name__ == '__main__':
    paths = glob.glob(INPUT_PATH + '/*/*')
    paths = [p for p in paths if os.path.isdir(p)]
    print('Found {} models'.format(len(paths)))
    p = Pool(32)
    p.map(convert_and_scale, paths)
    print('Done.')
