# Controlled spatial-coverage experiments

This directory contains the experimental implementation for evaluating the inherited Baseline IF-Net (epoch 1450) and Scale Dropout (epoch 1499, training p=0.1) checkpoints under spatially incomplete inputs. The experiments use fixed weights without additional training. The original training and model implementation remains in the repository root.

## Experimental protocol

- Reference geometry: ShapeNet chair `isosurf_scaled.off` meshes from the supplied held-out test split; the main experiment uses the first 200 test chairs.
- Sample 20,000 candidate surface points per mesh. Construct full, `x >= 0`, and `x <= 0` observations before FPS, preserving coordinates after filtering.
- Apply FPS to obtain 300 or 1,000 points; the 300-point set is a prefix of the 1,000-point sequence. Supply identical inputs to both checkpoints.
- Voxelize inputs at 128 cubed and predict occupancy on a 128-cubed query lattice. Extract the surface at occupancy probability 0.5.
- Measure surface CD-L2, F-score at 0.01, missing-region mean squared distance, and missing-region recall. Record extraction failures with status and error information.
- Average single-model metrics over its successful extractions. Compare models on matched successful object IDs and compute 95% object-level bootstrap confidence intervals.

## Code map

| File | Purpose |
| --- | --- |
| `ifnet_eval/geometry.py` | Surface sampling, half-space filtering, FPS, voxelization, surface extraction, and geometry metrics |
| `ifnet_eval/model.py` | Checkpoint-compatible cached IF-Net inference and decoder-feature masking |
| `ifnet_eval/experiment.py` | Input preparation, paired evaluation, failure records, summaries, and bootstrap intervals |
| `ifnet_eval/runtime.py` | Internal sample selection and resumable evaluation support; imported by the experiment module |
| `scripts/build_manifest.py` | Select held-out mesh IDs from an explicit test split |
| `scripts/run_experiment.py` | Python entry point for input preparation and evaluation |
| `scripts/analyze_effects.py` | Within-object coverage and decoder-mask contrasts |
| `configs/pilot.json` | Main coverage protocol settings |
| `configs/ablation.json` | Additional decoder-feature masking settings |

The ablation configuration masks branches 0 and 1 for `high_off` and branches 4 and 5 for `coarse_off`. Ordinary main-experiment inference keeps all decoder branches active. Training-time Scale Dropout masks feature scales; half-space removal filters input geometry. These are distinct operations.

The JSON files define experimental settings. The manifest and optional index/limit arguments determine the evaluated objects; the configuration files do not impose the reported 200-object main or 20-object ablation sample counts themselves.

## Data and checkpoint interface

Paths in each configuration are relative to this directory. Provide `data/test_manifest.json`, the reference meshes it lists, and the two checkpoints at the configured paths, or update the configuration paths. A manifest contains a `samples` list with `id`, `mesh`, and `normalize` fields, plus split-verification metadata produced by `build_manifest.py`. Pre-scaled meshes use `normalize: false`.

This implementation uses the surface-extraction metric pipeline in `ifnet_eval`, independently of the historical evaluation scripts at the repository root. The code does not bundle meshes, checkpoints, result files, plotting tools, cluster submission scripts, or personal cluster configuration.

The code was used with Python 3.12 and PyTorch 2.5.1; choose a PyTorch build suitable for the target GPU environment. The inherited training code has its own environment definition at the repository root.
