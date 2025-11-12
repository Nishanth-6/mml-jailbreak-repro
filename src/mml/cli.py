import argparse, yaml, json
from .dataset import load_subset
from .model_interface import build_model
from .attack import run_attack
from .evaluate import compute_asr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--limit", type=int, default=10)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    rows = load_subset(cfg["paths"]["data_csv"])[:args.limit]
    model = build_model(cfg["engine"], **cfg.get(cfg["engine"], {}))
    results = [run_attack(r, cfg, model) for r in rows]
    succ, total, asr = compute_asr(results, cfg["eval"]["refusal_keywords"], cfg["eval"]["judge_rule_strict"])
    print(json.dumps({"success":succ,"total":total,"ASR":asr,"engine":cfg["engine"]}, indent=2))

if __name__=="__main__":
    main()
