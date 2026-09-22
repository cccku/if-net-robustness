import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os

MODEL_ID = '03001627/21a7166815b80f3f7fee47aa19e2e28c'
DATA_PATH = 'shapenet/data/'
BASELINE_GEN = 'experiments/iPC3000_dist-0.5_0.5_sigmas-0.15_0.015_v32_mShapeNetPoints/evaluation_299_@64/generation/'
DROPOUT_GEN = 'experiments/iPC3000_dist-0.5_0.5_sigmas-0.15_0.015_v32_mShapeNetPointsDropout/evaluation_299_@64/generation/'

def render_mesh(ax, mesh, title, color='lightblue', alpha=0.7):
    if mesh is None or len(mesh.faces) == 0:
        ax.text(0.5, 0.5, 0.5, 'No mesh', ha='center')
        ax.set_title(title, fontsize=10, fontweight='bold')
        return
    verts = mesh.vertices
    faces = mesh.faces
    poly = Poly3DCollection(verts[faces], alpha=alpha)
    poly.set_facecolor(color)
    poly.set_edgecolor('none')
    ax.add_collection3d(poly)
    ax.set_xlim(verts[:,0].min(), verts[:,0].max())
    ax.set_ylim(verts[:,1].min(), verts[:,1].max())
    ax.set_zlim(verts[:,2].min(), verts[:,2].max())
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_axis_off()
    ax.view_init(elev=20, azim=45)

def render_pointcloud(ax, points, title, color='steelblue', size=1):
    ax.scatter(points[:,0], points[:,1], points[:,2],
               s=size, c=color, alpha=0.6)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_axis_off()
    ax.view_init(elev=20, azim=45)

def load_mesh(path):
    try:
        mesh = trimesh.load(path, process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
        if isinstance(mesh, trimesh.Trimesh) and len(mesh.faces) > 0:
            mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
            return mesh
    except:
        pass
    return None

# ── Figure 1: Degradation study (input point clouds) ──────────────────────
fig = plt.figure(figsize=(15, 4))
fig.suptitle('Input Point Cloud Degradation — ShapeNet Chair', 
             fontsize=13, fontweight='bold', y=1.02)

pc_3000 = np.load(os.path.join(DATA_PATH, MODEL_ID, 
    'voxelized_point_cloud_32res_3000points.npz'))['point_cloud']
pc_1000 = np.load(os.path.join(DATA_PATH, MODEL_ID,
    'fps_point_cloud_1000points.npz'))['point_cloud']
pc_300  = np.load(os.path.join(DATA_PATH, MODEL_ID,
    'fps_point_cloud_300points.npz'))['point_cloud']
gt_mesh = load_mesh(os.path.join(DATA_PATH, MODEL_ID, 'isosurf_scaled.off'))

titles = ['GT Mesh', '3000 pts (dense)', '1000 pts (FPS)', '300 pts (FPS)']
data   = [None, pc_3000, pc_1000, pc_300]
colors = ['lightcoral', 'steelblue', 'darkorange', 'purple']

for i, (title, d, c) in enumerate(zip(titles, data, colors)):
    ax = fig.add_subplot(1, 4, i+1, projection='3d')
    if i == 0:
        render_mesh(ax, gt_mesh, title, color='lightcoral', alpha=0.8)
    else:
        render_pointcloud(ax, d, title, color=c, size=2 if i==1 else 4)

plt.tight_layout()
plt.savefig('figures/fig1_degradation_inputs.png', dpi=150, bbox_inches='tight')
print('Saved fig1_degradation_inputs.png')
plt.close()

# ── Figure 2: Reconstruction comparison ───────────────────────────────────
fig = plt.figure(figsize=(12, 4))
fig.suptitle('Reconstruction Quality — Baseline vs Scale Dropout',
             fontsize=13, fontweight='bold', y=1.02)

model_folder = MODEL_ID.split('/')[-1]
baseline_mesh = load_mesh(os.path.join(BASELINE_GEN, '03001627', 
    model_folder, 'surface_reconstruction.off'))
dropout_mesh  = load_mesh(os.path.join(DROPOUT_GEN, '03001627',
    model_folder, 'surface_reconstruction.off'))

meshes = [gt_mesh, baseline_mesh, dropout_mesh]
titles = ['Ground Truth', 'Baseline IF-Net\n(3000 pts input)', 
          'IF-Net + Scale Dropout\n(3000 pts input)']
colors = ['lightcoral', 'lightblue', 'lightgreen']

for i, (mesh, title, color) in enumerate(zip(meshes, titles, colors)):
    ax = fig.add_subplot(1, 3, i+1, projection='3d')
    render_mesh(ax, mesh, title, color=color, alpha=0.8)

plt.tight_layout()
plt.savefig('figures/fig2_reconstruction_comparison.png', dpi=150, bbox_inches='tight')
print('Saved fig2_reconstruction_comparison.png')
plt.close()

# ── Figure 3: Results bar chart ────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Quantitative Results — IoU and Chamfer Distance',
             fontsize=13, fontweight='bold')

points = ['3000', '1000\n(FPS)', '300\n(FPS)']
x = np.arange(len(points))
width = 0.35

# IoU
baseline_iou = [0.0289, 0.0313, 0.0418]
dropout_iou  = [0.0214, 0.0231, 0.0154]
baseline_iou_std = [0.0839, 0.0750, 0.0694]
dropout_iou_std  = [0.0461, 0.0384, 0.0266]

bars1 = ax1.bar(x - width/2, baseline_iou, width, label='Baseline IF-Net',
                color='steelblue', alpha=0.8,
                yerr=baseline_iou_std, capsize=4)
bars2 = ax1.bar(x + width/2, dropout_iou, width, label='IF-Net + Scale Dropout',
                color='darkorange', alpha=0.8,
                yerr=dropout_iou_std, capsize=4)
ax1.set_xlabel('Input Point Cloud Size', fontsize=11)
ax1.set_ylabel('IoU ↑', fontsize=11)
ax1.set_title('Volumetric IoU', fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels(points)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Chamfer
baseline_cd = [0.010845, 0.019050, 0.023128]
dropout_cd  = [0.083617, 0.073524, 0.083942]
baseline_cd_std = [0.023394, 0.033852, 0.035722]
dropout_cd_std  = [0.059023, 0.052817, 0.077343]

ax2.bar(x - width/2, baseline_cd, width, label='Baseline IF-Net',
        color='steelblue', alpha=0.8,
        yerr=baseline_cd_std, capsize=4)
ax2.bar(x + width/2, dropout_cd, width, label='IF-Net + Scale Dropout',
        color='darkorange', alpha=0.8,
        yerr=dropout_cd_std, capsize=4)
ax2.set_xlabel('Input Point Cloud Size', fontsize=11)
ax2.set_ylabel('Chamfer Distance ↓', fontsize=11)
ax2.set_title('Chamfer Distance', fontsize=12)
ax2.set_xticks(x)
ax2.set_xticklabels(points)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('figures/fig3_quantitative_results.png', dpi=150, bbox_inches='tight')
print('Saved fig3_quantitative_results.png')
plt.close()

print('All figures saved to figures/')
