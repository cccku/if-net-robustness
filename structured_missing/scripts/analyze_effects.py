"""Within-object coverage penalties and scale-mask sensitivity, with failures."""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ifnet_eval.experiment import read_json, resolve, write_csv


def compare(reference, changed, metric):
    ids = sorted(set(reference) & set(changed))
    both = [i for i in ids if reference[i]["status"] == changed[i]["status"] == "ok"]
    deltas = [changed[i]["metrics"][metric] - reference[i]["metrics"][metric] for i in both
              if reference[i]["metrics"][metric] is not None and changed[i]["metrics"][metric] is not None]
    return {"n_common": len(ids), "n_both_success": len(both),
            "n_reference_ok_changed_failed": sum(reference[i]["status"] == "ok" and changed[i]["status"] != "ok" for i in ids),
            "n_reference_failed_changed_ok": sum(reference[i]["status"] != "ok" and changed[i]["status"] == "ok" for i in ids),
            "metric": metric, "delta_changed_minus_reference": float(np.mean(deltas)) if deltas else None,
            "n_metric_pairs": len(deltas)}


def analyze(output):
    output = resolve(output)
    records = [read_json(p) for p in (output / "records").glob("*.json")]
    groups = {}
    for r in records:
        groups.setdefault((r["model"], r["points"], r["condition"], r["ablation"]), {})[r["sample"]] = r
    coverage, sensitivity = [], []
    for (model, n, condition, ablation), changed in sorted(groups.items()):
        if condition != "full" and (model, n, "full", ablation) in groups:
            coverage.append({"model": model, "points": n, "condition": condition, "ablation": ablation,
                **compare(groups[(model, n, "full", ablation)], changed, "cd_l2")})
        if ablation != "all" and (model, n, condition, "all") in groups:
            for metric in ("cd_l2", "missing_gt_to_pred_l2"):
                sensitivity.append({"model": model, "points": n, "condition": condition, "ablation": ablation,
                    **compare(groups[(model, n, condition, "all")], changed, metric)})
    write_csv(output / "coverage_penalty.csv", coverage)
    write_csv(output / "scale_sensitivity.csv", sensitivity)
    return coverage, sensitivity


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/pilot")
    args = parser.parse_args()
    coverage, sensitivity = analyze(args.output)
    print(f"Saved {len(coverage)} coverage contrasts and {len(sensitivity)} scale contrasts")

