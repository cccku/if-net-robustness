import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob, os

def load_tb_scalar(summary_dir, tag):
    steps, values = [], []
    for ef in sorted(glob.glob(os.path.join(summary_dir, 'events.out.tfevents.*'))):
        try:
            ea = EventAccumulator(ef); ea.Reload()
            if tag in ea.Tags().get('scalars', []):
                for e in ea.Scalars(tag):
                    steps.append(e.step); values.append(e.value)
        except: continue
    if steps:
        pairs = sorted(zip(steps, values))
        steps, values = zip(*pairs)
    return np.array(steps), np.array(values)

def smooth(values, window=15):
    return np.convolve(values, np.ones(window)/window, mode='valid')

baseline_dir = 'experiments/iPC3000_dist-0.5_0.5_sigmas-0.15_0.015_v32_mShapeNetPoints/summary/'
dropout_dir  = 'experiments/iPC3000_dist-0.5_0.5_sigmas-0.15_0.015_v32_mShapeNetPointsDropout/summary/'

b_tr_s, b_tr_v = load_tb_scalar(baseline_dir, 'training loss batch avg')
d_tr_s, d_tr_v = load_tb_scalar(dropout_dir,  'training loss batch avg')
b_vl_s, b_vl_v = load_tb_scalar(baseline_dir, 'val loss batch avg')
d_vl_s, d_vl_v = load_tb_scalar(dropout_dir,  'val loss batch avg')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Training vs Validation Loss: Baseline vs Scale Dropout',
             fontsize=13, fontweight='bold')

# Training loss
axes[0].plot(b_tr_s[14:], smooth(b_tr_v), color='steelblue',
             linewidth=2, label='Baseline IF-Net')
axes[0].plot(d_tr_s[14:], smooth(d_tr_v), color='darkorange',
             linewidth=2, label='IF-Net + Scale Dropout')
axes[0].set_xlabel('Epoch', fontsize=11)
axes[0].set_ylabel('Loss', fontsize=11)
axes[0].set_title('Training Loss (smoothed)', fontsize=11)
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3)
axes[0].set_ylim(0, 15000)

# Val loss
axes[1].plot(b_vl_s[14:], smooth(b_vl_v), color='steelblue',
             linewidth=2, label='Baseline IF-Net')
axes[1].plot(d_vl_s[14:], smooth(d_vl_v), color='darkorange',
             linewidth=2, label='IF-Net + Scale Dropout')
axes[1].set_xlabel('Epoch', fontsize=11)
axes[1].set_ylabel('Loss', fontsize=11)
axes[1].set_title('Validation Loss (smoothed)', fontsize=11)
axes[1].legend(fontsize=10)
axes[1].grid(alpha=0.3)
axes[1].set_ylim(0, 25000)

plt.tight_layout()
plt.savefig('figures/fig4_loss_curves.png', dpi=150, bbox_inches='tight')
print('Saved figures/fig4_loss_curves.png')
