"""Select held-out chair meshes from an explicit split; never scan train silently."""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ifnet_eval.experiment import ROOT, resolve, write_json
from ifnet_eval.model import file_sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", required=True, help="NPZ containing 'test' IDs or TXT with one test ID per line")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--normalize-raw", action="store_true", help="Allow model.obj fallback; normalize full mesh once")
    parser.add_argument("--output", default="data/test_manifest.json")
    args = parser.parse_args()
    data_root, split = resolve(args.data_root), resolve(args.split)
    if args.limit < 1:
        parser.error("--limit must be positive")
    if split.suffix.lower() == ".npz":
        with np.load(split, allow_pickle=False) as data:
            ids = data["test"].tolist()
    else:
        ids = split.read_text(encoding="utf-8-sig").splitlines()
    samples, missing, seen = [], [], set()
    for raw in ids:
        raw = raw.decode() if isinstance(raw, bytes) else str(raw)
        parts = raw.strip().replace("\\", "/").strip("/").split("/")
        if "03001627" not in parts:
            continue
        start = parts.index("03001627")
        if len(parts) < start + 2 or parts[start + 1] in (".", "..", ""):
            raise ValueError(f"Invalid shape ID: {raw}")
        object_id = "/".join(parts[start:start + 2])
        if object_id in seen:
            continue
        seen.add(object_id)
        folder = data_root / object_id
        mesh, normalize = folder / "isosurf_scaled.off", False
        if not mesh.is_file() and args.normalize_raw:
            mesh, normalize = folder / "model.obj", True
        if not mesh.is_file():
            missing.append(object_id)
        else:
            try:
                mesh_path = mesh.resolve().relative_to(ROOT).as_posix()
            except ValueError:
                mesh_path = str(mesh.resolve())
            samples.append({"id": object_id, "mesh": mesh_path, "normalize": normalize})
        if len(samples) == args.limit:
            break
    if not samples:
        raise RuntimeError("No held-out chair meshes found. Obtain test data first.")
    output = resolve(args.output)
    if output.exists():
        raise FileExistsError(f"Manifest already exists: {output}; choose another --output")
    write_json(output, {"split_verified": True, "split_source": str(split),
               "split_sha256": file_sha256(split), "selection": "first available chair IDs in supplied test split",
               "missing_ids_before_limit": missing, "samples": samples})
    print(f"Wrote {len(samples)} samples; skipped {len(missing)} missing IDs. Requested {args.limit}.")


if __name__ == "__main__":
    main()
