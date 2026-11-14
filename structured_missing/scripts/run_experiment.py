"""CLI entrypoint; safe to run from any working directory."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ifnet_eval.experiment import (evaluate, load_config, preflight, prepare,
                                   read_json, resolve, summarize, write_json)
from ifnet_eval.runtime import EvaluationPaused, RunBudget


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("check", "prepare", "evaluate", "summarize"))
    parser.add_argument("--config", default="configs/pilot.json")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--stop-index", type=int, help="Exclusive end of fixed manifest range")
    parser.add_argument("--max-seconds", type=float, help="Budget does not alter experiment identity")
    parser.add_argument("--output", help="Use a separate directory for each pilot/protocol")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--retry-failures", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.output:
        config["output"] = args.output
    if args.device:
        config["device"] = args.device
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.max_seconds is not None and args.max_seconds <= 0:
        parser.error("--max-seconds must be positive")
    if args.command == "check":
        result = preflight(config)
        write_json(resolve(config["output"]) / "preflight.json", result)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["ready"] else 2
    if args.command == "prepare":
        with RunBudget(args.max_seconds).signals() as budget:
            print(json.dumps(prepare(config, args.start_index, args.stop_index, args.limit, budget), indent=2, ensure_ascii=False))
    elif args.command == "evaluate":
        with RunBudget(args.max_seconds).signals() as budget:
            output = evaluate(config, args.limit, args.retry_failures,
                              args.start_index, args.stop_index, budget)
        if any((output / "records").glob("*.json")):
            summarize(output)
        print(f"Saved results to {output}")
        progress = read_json(output / "progress.json")
        print(json.dumps(progress, ensure_ascii=False))
        return 0 if progress["complete"] else 75
    else:
        print(json.dumps(summarize(config["output"]), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except EvaluationPaused as exc:
        print(str(exc), flush=True)
        raise SystemExit(75)

