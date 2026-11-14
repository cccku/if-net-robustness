"""Geometry in ONE world xyz coordinate system; no crop renormalization."""
from __future__ import annotations

import hashlib

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes

CONDITIONS = {
    "full": None,
    "x_positive": (0, 1), "x_negative": (0, -1),
    "z_positive": (2, 1), "z_negative": (2, -1),
}


def stable_seed(seed, *parts):
    key = "|".join([str(seed), *map(str, parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "little")


def load_mesh(path):
    mesh = trimesh.load(path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError(f"Not a nonempty triangle mesh: {path}")
    if not np.isfinite(mesh.vertices).all():
        raise ValueError("Mesh has nonfinite coordinates")
    return mesh


def normalize_mesh(mesh):
    mesh = mesh.copy()
    center = mesh.bounds.mean(axis=0)
    scale = float(np.ptp(mesh.bounds, axis=0).max())
    if scale <= 0:
        raise ValueError("Degenerate bounds")
    mesh.vertices = (mesh.vertices - center) / scale
    return mesh, {"center": center.tolist(), "scale": scale}


def sample_surface(mesh, n, seed):
    points, _ = trimesh.sample.sample_surface(mesh, n, seed=int(seed))
    return points.astype(np.float32)


def observed_mask(points, condition):
    rule = CONDITIONS[condition]
    if rule is None:
        return np.ones(len(points), dtype=bool)
    axis, sign = rule
    return sign * points[:, axis] >= 0


def fps(points, n, seed):
    points = np.asarray(points, dtype=np.float32)
    if len(points) < n or n < 1:
        raise ValueError(f"Cannot select {n} distinct FPS points from {len(points)}")
    selected = np.empty(n, dtype=np.int64)
    distances = np.full(len(points), np.inf)
    index = int(np.random.default_rng(seed).integers(len(points)))
    for i in range(n):
        selected[i] = index
        d2 = np.sum((points - points[index]) ** 2, axis=1)
        distances = np.minimum(distances, d2)
        distances[selected[:i + 1]] = -1
        index = int(np.argmax(distances))
        if i + 1 < n and distances[index] <= 0:
            raise ValueError("Not enough distinct positions for FPS")
    return points[selected].copy()


def voxelize(points, resolution=128):
    if resolution < 16 or len(points) == 0:
        raise ValueError("Input must be nonempty and resolution >=16")
    points = np.asarray(points)
    if not np.isfinite(points).all() or np.max(np.abs(points)) > 0.50001:
        raise ValueError("Expected normalized xyz points in [-0.5, 0.5]")
    # C-order tensor axes are world x,y,z, matching upstream preprocessing.
    indices = np.rint((points + .5) * (resolution - 1)).astype(np.int64)
    indices = np.clip(indices, 0, resolution - 1)
    volume = np.zeros((resolution,) * 3, dtype=np.float32)
    volume[tuple(indices.T)] = 1
    return volume


def query_lattice(resolution):
    axis = np.linspace(-.5, .5, resolution, dtype=np.float32)
    return np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)


def extract_surface(logits, resolution, probability=.5):
    if not 0 < probability < 1:
        raise ValueError("Occupancy threshold must be in (0,1)")
    level = float(np.log(probability / (1 - probability)))
    volume = np.asarray(logits, dtype=np.float32).reshape((resolution,) * 3)
    if not np.isfinite(volume).all():
        raise ValueError("Nonfinite occupancy logits")
    # Do not manufacture a closed box when no isosurface crosses the query domain.
    if volume.min() >= level or volume.max() <= level:
        raise ValueError("No isosurface crossing (all occupied or all empty)")
    vertices, faces, _, _ = marching_cubes(volume, level=level, allow_degenerate=False)
    vertices = vertices / (resolution - 1) - .5
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if not len(mesh.faces) or mesh.area <= 0:
        raise ValueError("Empty/degenerate extracted mesh")
    return mesh


def surface_metrics(pred, gt, missing_mask=None, threshold=.01):
    if len(pred) == 0 or len(gt) == 0:
        raise ValueError("Surface metrics require nonempty point sets")
    gt_to_pred = cKDTree(pred).query(gt, workers=1)[0]
    pred_to_gt = cKDTree(gt).query(pred, workers=1)[0]
    precision = float(np.mean(pred_to_gt <= threshold))
    recall = float(np.mean(gt_to_pred <= threshold))
    result = {
        "cd_l2": float((np.mean(gt_to_pred ** 2) + np.mean(pred_to_gt ** 2)) / 2),
        "cd_l1": float((np.mean(gt_to_pred) + np.mean(pred_to_gt)) / 2),
        "precision": precision, "recall": recall,
        "fscore": 2 * precision * recall / (precision + recall) if precision + recall else 0.,
        "missing_gt_to_pred_l2": None, "missing_recall": None,
        "missing_gt_count": 0,
    }
    if missing_mask is not None and np.any(missing_mask):
        distances = gt_to_pred[missing_mask]
        result.update(missing_gt_to_pred_l2=float(np.mean(distances ** 2)),
                      missing_recall=float(np.mean(distances <= threshold)),
                      missing_gt_count=int(np.sum(missing_mask)))
    return result

