import argparse
import json
import os
import csv
from datetime import datetime

import yaml

from .dataset import load_subset
from .model_interface import build_model
from .attack import run_attack
from .evaluate import compute_asr


def save_run_summary(cfg, success, total, asr):
    os.makedirs("results", exist_ok=True)

    engine = cfg["engine"]
    methods = cfg["attack"]["encryption"]
    methods_str = "+".join(methods)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join("results", "runs_summary.csv")

    file_exists = os.path.exists(summary_path)

    with open(summary_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "engine", "methods", "success", "total", "ASR"])
        writer.writerow([timestamp, engine, methods_str, success, total, asr])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--limit", type=int, default=10)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))

    rows = load_subset(cfg["paths"]["data_csv"])[: args.limit]
    model = build_model(cfg["engine"], **cfg.get(cfg["engine"], {}))

    results = [run_attack(r, cfg, model) for r in rows]
    succ, total, asr = compute_asr(
        results,
        cfg["eval"]["refusal_keywords"],
        cfg["eval"]["judge_rule_strict"],
    )

    # print to stdout for quick viewing
    print(json.dumps({"success": succ, "total": total, "ASR": asr, "engine": cfg["engine"]}, indent=2))

    # log to CSV for report
    save_run_summary(cfg, succ, total, asr)


if __name__ == "__main__":
    main()
