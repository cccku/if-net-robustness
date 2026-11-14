"""Preparation and resumable paired evaluation. All paths are project-relative."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from .geometry import (CONDITIONS, fps, load_mesh, normalize_mesh, observed_mask,
                       sample_surface, stable_seed, voxelize, query_lattice,
                       extract_surface, surface_metrics)
from .model import file_sha256, load_model, query_world, synchronize, timed_encode
from .runtime import EvaluationPaused, RunBudget, select_samples

ROOT = Path(__file__).resolve().parents[1]


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load_config(path):
    config = read_json(resolve(path))
    for name in ("input_resolution", "query_resolution"):
        if config[name] < 16:
            raise ValueError(f"{name} must be >=16")
    if config["chunk_size"] < 1 or min(config["point_counts"]) < 1:
        raise ValueError("Invalid chunk size or point count")
    if not set(config["conditions"]) <= set(CONDITIONS):
        raise ValueError("Unknown coverage condition")
    if not config["ablations"] or any(
        i not in range(6) for mask in config["ablations"].values() for i in mask
    ):
        raise ValueError("Invalid scale ablation")
    if not 0 < config["threshold"] < 1 or config["fscore_threshold"] <= 0:
        raise ValueError("Invalid metric threshold")
    if len({m['name'] for m in config['models']}) != len(config['models']):
        raise ValueError("Duplicate model name")
    torch.set_num_threads(config.get("cpu_threads", 4))
    return config


def get_device(config):
    device = config["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but this PyTorch build cannot use CUDA")
    return device


def preflight(config):
    missing, checkpoints = [], []
    for spec in config["models"]:
        path = resolve(spec["checkpoint"])
        if not path.is_file():
            missing.append(str(path))
            continue
        _, info = load_model(path)
        info["name"] = spec["name"]
        if info["epoch"] != spec.get("expected_epoch", info["epoch"]):
            raise ValueError(f"Unexpected epoch for {spec['name']}")
        checkpoints.append(info)
    manifest_path = resolve(config["manifest"])
    sample_count = 0
    split_verified = False
    if not manifest_path.is_file():
        missing.append(str(manifest_path))
    else:
        manifest = read_json(manifest_path)
        split_verified = bool(manifest.get("split_verified"))
        samples = manifest["samples"]
        sample_count = len(samples)
        if not samples or len({s['id'] for s in samples}) != len(samples):
            raise ValueError("Manifest is empty or contains duplicate IDs")
        for sample in samples:
            if not resolve(sample["mesh"]).is_file():
                missing.append(str(resolve(sample["mesh"])))
    return {"ready": not missing and (split_verified or config.get("synthetic", False)),
            "missing": missing, "split_verified": split_verified,
            "synthetic": config.get("synthetic", False), "samples": sample_count,
            "device": get_device(config), "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "checkpoints": checkpoints,
            "epoch_mismatch": len({c["epoch"] for c in checkpoints}) > 1}


def prepare(config, start_index=0, stop_index=None, limit=None, budget=None):
    check = preflight(config)
    if not check["ready"]:
        raise RuntimeError("Data not ready: " + json.dumps(check, ensure_ascii=False))
    manifest = read_json(resolve(config["manifest"]))
    cache_root = ROOT / "data" / "prepared"
    descriptors = []
    samples = select_samples(manifest["samples"], start_index, stop_index, limit)
    for sample in samples:
        if budget is not None and budget.exhausted():
            raise EvaluationPaused("Preparation paused; completed cache entries are retained")
        path = resolve(sample["mesh"])
        descriptor = {"id": sample["id"], "mesh_sha256": file_sha256(path),
                      "normalize": bool(sample.get("normalize", False)),
                      "seed": config["seed"], "dense_points": config["dense_points"],
                      "surface_points": config["surface_points"],
                      "point_counts": config["point_counts"], "conditions": config["conditions"],
                      "preparation_version": 1}
        folder = cache_root / identity(descriptor)[:24]
        if not (folder / "ready.json").is_file():
            mesh = load_mesh(path)
            transform = None
            if descriptor["normalize"]:
                mesh, transform = normalize_mesh(mesh)
            if np.abs(mesh.vertices).max() > .50001:
                raise ValueError(f"{sample['id']}: bounds outside [-.5,.5]; set normalize explicitly for raw meshes")
            folder.mkdir(parents=True, exist_ok=True)
            mesh.export(folder / "gt_mesh.ply")
            dense = sample_surface(mesh, config["dense_points"], stable_seed(config["seed"], sample["id"], "dense"))
            gt = sample_surface(mesh, config["surface_points"], stable_seed(config["seed"], sample["id"], "gt"))
            np.savez_compressed(folder / "reference.npz", dense=dense, gt_surface=gt)
            for condition in config["conditions"]:
                observed = observed_mask(dense, condition)
                selected = fps(dense[observed], max(config["point_counts"]),
                               stable_seed(config["seed"], sample["id"], condition, "fps"))
                for n in config["point_counts"]:
                    np.savez_compressed(folder / f"{condition}_{n}.npz",
                                        points=selected[:n], missing_gt=~observed_mask(gt, condition),
                                        dense_retained_fraction=float(observed.mean()))
            write_json(folder / "ready.json", {**descriptor, "transform": transform,
                       "bounds": mesh.bounds.tolist(), "watertight": bool(mesh.is_watertight)})
        descriptors.append({"id": sample["id"], "folder": str(folder), **descriptor})
    write_json(resolve(config["output"]) / "prepared_index.json", {"samples": descriptors})
    return descriptors


def code_fingerprint():
    paths = sorted((ROOT / "ifnet_eval").glob("*.py"))
    return identity({p.name: file_sha256(p) for p in paths})


def evaluate(config, limit=None, retry_failures=False, start_index=0, stop_index=None, budget=None):
    budget = budget or RunBudget()
    prepared = prepare(config, start_index, stop_index, limit, budget)
    device = get_device(config)
    output = resolve(config["output"])
    checkpoint_hashes = {m["name"]: file_sha256(resolve(m["checkpoint"])) for m in config["models"]}
    protocol = {"config": config, "samples": prepared, "checkpoints": checkpoint_hashes,
                "implementation": code_fingerprint(), "actual_device": device,
                "torch": torch.__version__}
    fingerprint = identity(protocol)
    run_file = output / "run.json"
    if run_file.exists() and read_json(run_file)["fingerprint"] != fingerprint:
        raise RuntimeError("Output contains a different protocol/data/code version. Choose a new output directory.")
    if not run_file.exists():
        write_json(run_file, {"fingerprint": fingerprint, **protocol, "platform": platform.platform(),
                   "synthetic": config.get("synthetic", False),
                   "metric_protocol": "cd_l2=0.5*(mean GT->pred squared + mean pred->GT squared), normalized world xyz; surface sampled; missing metric is one-way GT->pred"})
    expected = len(prepared) * len(config["models"]) * len(config["point_counts"]) * len(config["conditions"]) * len(config["ablations"])

    def save_progress():
        saved = [read_json(p) for p in (output / "records").glob("*.json")]
        attempted = len(saved)
        if attempted > expected or any(r["fingerprint"] != fingerprint for r in saved):
            raise RuntimeError("Unexpected records in output directory")
        write_json(output / "progress.json", {"expected": expected, "attempted": attempted,
                   "pending": expected - attempted, "successful": sum(r["status"] == "ok" for r in saved),
                   "failed": sum(r["status"] != "ok" for r in saved),
                   "complete": attempted == expected, "pause_reason": budget.reason})
        return output

    save_progress()
    lattice = query_lattice(config["query_resolution"])
    for spec in config["models"]:
        if budget.exhausted():
            return save_progress()
        net, checkpoint = load_model(resolve(spec["checkpoint"]), device)
        for sample in prepared:
            folder = Path(sample["folder"])
            gt_surface = np.load(folder / "reference.npz")["gt_surface"]
            sample_token = identity(sample["id"])[:16]
            for condition in config["conditions"]:
                for n in config["point_counts"]:
                    case = {"sample": sample["id"], "model": spec["name"], "points": n,
                            "condition": condition, "epoch": checkpoint["epoch"],
                            "synthetic": config.get("synthetic", False)}
                    targets = []
                    for name, mask in config["ablations"].items():
                        record_path = output / "records" / f"{identity({**case, 'ablation': name})[:24]}.json"
                        if record_path.exists() and not (retry_failures and read_json(record_path)["status"] != "ok"):
                            continue
                        targets.append((name, mask, record_path))
                    if not targets:
                        continue
                    if budget.exhausted():
                        return save_progress()
                    asset = np.load(folder / f"{condition}_{n}.npz")
                    voxels = voxelize(asset["points"], config["input_resolution"])
                    features = None
                    encode_error = None
                    if str(device).startswith("cuda"):
                        torch.cuda.reset_peak_memory_stats(device)
                    try:
                        features, encode_seconds = timed_encode(net, voxels, device)
                    except Exception as exc:
                        encode_error = exc
                    for name, mask, record_path in targets:
                        if budget.exhausted():
                            return save_progress()
                        record = {**case, "ablation": name, "masked_scales": mask,
                                  "fingerprint": fingerprint,
                                  "input_sha256": file_sha256(folder / f"{condition}_{n}.npz"),
                                  "retained_fraction": float(asset["dense_retained_fraction"]),
                                  "status": "failed", "metrics": None}
                        try:
                            if encode_error is not None:
                                raise encode_error
                            synchronize(device)
                            start = time.perf_counter()
                            logits = query_world(net, features, lattice, config["chunk_size"], mask)
                            synchronize(device)
                            query_seconds = time.perf_counter() - start
                            start = time.perf_counter()
                            mesh = extract_surface(logits, config["query_resolution"], config["threshold"])
                            prediction_path = output / "meshes" / spec["name"] / sample_token / f"{condition}_{n}_{name}.ply"
                            prediction_path.parent.mkdir(parents=True, exist_ok=True)
                            mesh.export(prediction_path)
                            pred_surface = sample_surface(mesh, config["surface_points"],
                                stable_seed(config["seed"], sample["id"], condition, n, "pred_surface"))
                            metrics = surface_metrics(pred_surface, gt_surface, asset["missing_gt"], config["fscore_threshold"])
                            record.update(status="ok", metrics=metrics, mesh=str(prediction_path),
                                          encode_seconds=encode_seconds, query_seconds=query_seconds,
                                          surface_and_metric_seconds=time.perf_counter() - start,
                                          gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if str(device).startswith("cuda") else None)
                        except Exception as exc:
                            record.update(error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
                        write_json(record_path, record)
                        print(json.dumps({k: record[k] for k in ("sample", "model", "points", "condition", "ablation", "status")}), flush=True)
                    del features
        del net
    return save_progress()


def write_csv(path, rows):
    rows = list(rows)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(output):
    output = resolve(output)
    records = [read_json(p) for p in sorted((output / "records").glob("*.json"))]
    if not records:
        raise ValueError("No evaluation records")
    flattened = [{**{k: v for k, v in r.items() if k not in ("metrics", "traceback", "masked_scales")},
                  **(r["metrics"] or {})} for r in records]
    write_csv(output / "per_sample.csv", flattened)
    groups = {}
    for r in records:
        key = tuple(r[k] for k in ("model", "points", "condition", "ablation"))
        groups.setdefault(key, []).append(r)
    summary = []
    for key, group in sorted(groups.items()):
        ok = [r for r in group if r["status"] == "ok"]
        row = dict(zip(("model", "points", "condition", "ablation"), key))
        row.update(n_attempted=len(group), n_success=len(ok), n_failed=len(group) - len(ok),
                   failure_rate=(len(group) - len(ok)) / len(group))
        for metric in ("cd_l2", "cd_l1", "fscore", "missing_gt_to_pred_l2", "missing_recall"):
            values = [r["metrics"][metric] for r in ok if r["metrics"][metric] is not None]
            row[metric + "_mean_success_only"] = float(np.mean(values)) if values else None
            row[metric + "_std_success_only"] = float(np.std(values)) if values else None
        summary.append(row)
    write_csv(output / "summary.csv", summary)
    # Paired comparison uses the SAME successful object IDs, never unmatched means.
    pair_rows = []
    names = read_json(output / "run.json")["config"]["models"]
    if len(names) == 2:
        a, b = [m["name"] for m in names]
        settings = sorted({(r["points"], r["condition"], r["ablation"]) for r in records})
        for n, condition, ablation in settings:
            subsets = [{r["sample"]: r for r in records if r["model"] == name and
                        (r["points"], r["condition"], r["ablation"]) == (n, condition, ablation)} for name in (a, b)]
            common = sorted(set(subsets[0]) & set(subsets[1]))
            successful = [i for i in common if all(s[i]["status"] == "ok" for s in subsets)]
            row = {"model_a": a, "model_b": b, "points": n, "condition": condition,
                   "ablation": ablation, "n_common_attempted": len(common), "n_both_success": len(successful),
                   "n_a_only_success": sum(subsets[0][i]["status"] == "ok" and subsets[1][i]["status"] != "ok" for i in common),
                   "n_b_only_success": sum(subsets[1][i]["status"] == "ok" and subsets[0][i]["status"] != "ok" for i in common)}
            for metric in ("cd_l2", "missing_gt_to_pred_l2"):
                deltas = np.array([subsets[1][i]["metrics"][metric] - subsets[0][i]["metrics"][metric]
                                   for i in successful if subsets[0][i]["metrics"][metric] is not None])
                row[metric + "_b_minus_a"] = float(deltas.mean()) if len(deltas) else None
                if len(deltas):
                    rng = np.random.default_rng(20261023)
                    bootstrap = np.array([rng.choice(deltas, len(deltas), replace=True).mean() for _ in range(2000)])
                    row[metric + "_paired_bootstrap_95_low"] = float(np.quantile(bootstrap, .025))
                    row[metric + "_paired_bootstrap_95_high"] = float(np.quantile(bootstrap, .975))
            pair_rows.append(row)
        write_csv(output / "paired_comparison.csv", pair_rows)
    write_json(output / "summary.json", {"groups": summary, "paired": pair_rows,
               "warning": "Conditional on successful reconstructions. Read failure rates alongside distances. Bootstrap resamples objects within each direction, not training runs."})
    return summary

