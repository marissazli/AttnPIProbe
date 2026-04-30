"""
eval_attn_tracker_expanded.py

Evaluates AttentionTracker on the full expanded dataset:
  - 1,575 malicious variants
  - 1,575 paired benign variants
  - Loaded from variations/variants_master.json

Reports TPR, FPR, and FNR across 3 input formats and 5 attack types.
Also breaks down by urgency level and reference style (new variation axes).

Usage:
    python eval_attn_tracker_expanded.py --model_name qwen2-attn --gpu_id 1
    python eval_attn_tracker_expanded.py --model_name qwen2-attn --gpu_id 1 --sample 200
"""

import argparse
import json
import os
import random
import torch
import numpy as np
from tqdm import tqdm

from utils import open_config, create_model
from detector.attn import AttentionDetector


ATTACK_TYPE_NAMES = {
    1: "Direct Execution",
    2: "Parameterized Execution",
    3: "Conditional Execution",
    4: "Functional Execution",
    5: "Transfer Execution",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_expanded_dataset(master_path, sample=None, seed=0):
    """
    Load malicious and benign variants from variants_master.json.
    Normalises field names to match the format builders.
    Optionally samples N pairs for fast testing.
    """
    with open(master_path) as f:
        master = json.load(f)

    malicious = master["malicious"]
    benign = master["benign"]

    if sample is not None:
        rng = random.Random(seed)
        indices = rng.sample(range(len(malicious)), min(sample, len(malicious)))
        malicious = [malicious[i] for i in indices]
        benign = [benign[i] for i in indices]

    # Normalise: malicious scenarios use attack_payload, benign use benign_instruction
    # Both need attack_observation_idx and steps
    def normalise_mal(s):
        return {
            "id": s["id"],
            "seed_id": s.get("seed_id", ""),
            "domain": s.get("domain", ""),
            "attack_type": s.get("attack_type") or s.get("attack", 0),
            "title": s.get("title", s.get("id", "")),
            "user_prompt": s["user_prompt"],
            "steps": s["steps"],
            "attack_observation_idx": s["attack_observation_idx"],
            "attack_payload": s.get("attack_payload", ""),
            "urgency": s.get("urgency", ""),
            "reference_style": s.get("reference_style", ""),
            "label": 1,
        }

    def normalise_ben(s):
        return {
            "id": s["id"],
            "seed_id": s.get("seed_id", ""),
            "domain": s.get("domain", ""),
            "attack_type": s.get("attack_type") or s.get("attack", 0),
            "title": s.get("title", s.get("id", "")),
            "user_prompt": s["user_prompt"],
            "steps": s["steps"],
            "attack_observation_idx": s["attack_observation_idx"],
            "attack_payload": s.get("benign_instruction", ""),  # reuse field name
            "urgency": s.get("urgency", ""),
            "reference_style": s.get("reference_style", ""),
            "label": 0,
        }

    return [normalise_mal(s) for s in malicious], [normalise_ben(s) for s in benign]


# ---------------------------------------------------------------------------
# Input format builders (same as seed eval)
# ---------------------------------------------------------------------------

def build_payload_only(scenario):
    return scenario.get("attack_payload", "")


def build_full_observation(scenario):
    idx = scenario["attack_observation_idx"]
    return scenario["steps"][idx]["observation"]


def build_full_context(scenario):
    parts = ["USER TASK: {}\n".format(scenario["user_prompt"])]
    for step in scenario["steps"]:
        parts.append("[Step {} | Tool: {}]".format(step["step"], step["tool"]))
        parts.append("Observation: {}".format(step["observation"]))
    return "\n".join(parts)


FORMAT_BUILDERS = {
    "payload_only":    build_payload_only,
    "full_observation": build_full_observation,
    "full_context":    build_full_context,
}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_evaluation(detector, scenarios, fmt_name, fmt_builder):
    results = []
    for s in tqdm(scenarios, desc=fmt_name, leave=False):
        input_text = fmt_builder(s)
        detection_result = detector.detect(input_text)
        detected = detection_result[0]
        focus_score = detection_result[1].get("focus_score", None)
        results.append({
            "id": s["id"],
            "seed_id": s.get("seed_id", ""),
            "domain": s.get("domain", ""),
            "attack_type": s["attack_type"],
            "label": s["label"],
            "format": fmt_name,
            "detected": detected,
            "focus_score": focus_score,
            "urgency": s.get("urgency", ""),
            "reference_style": s.get("reference_style", ""),
        })
    return results


def compute_tpr(results):
    """TPR on malicious (label=1): fraction correctly detected."""
    mal = [r for r in results if r["label"] == 1]
    if not mal:
        return None, 0, 0
    detected = sum(1 for r in mal if r["detected"])
    return round(detected / len(mal), 4), detected, len(mal)


def compute_fpr(results):
    """FPR on benign (label=0): fraction incorrectly flagged."""
    ben = [r for r in results if r["label"] == 0]
    if not ben:
        return None, 0, 0
    fp = sum(1 for r in ben if r["detected"])
    return round(fp / len(ben), 4), fp, len(ben)


def print_summary(all_results, n_mal, n_ben):
    formats = ["payload_only", "full_observation", "full_context"]

    print("\n" + "=" * 70)
    print("AttentionTracker — Expanded Dataset ({} mal + {} ben)".format(n_mal, n_ben))
    print("=" * 70)

    # Overall TPR / FPR by format
    print("\n── Overall TPR / FPR by Input Format ──")
    print("{:<22} {:>10} {:>10} {:>10} {:>10}".format(
        "Format", "TPR", "FPR", "Det(mal)", "FP(ben)"))
    print("-" * 65)
    for fmt in formats:
        res = [r for r in all_results if r["format"] == fmt]
        tpr, det, n_m = compute_tpr(res)
        fpr, fp, n_b = compute_fpr(res)
        tpr_str = "{:.1%}".format(tpr) if tpr is not None else "N/A"
        fpr_str = "{:.1%}".format(fpr) if fpr is not None else "N/A"
        print("{:<22} {:>10} {:>10} {:>10} {:>10}".format(
            fmt, tpr_str, fpr_str,
            "{}/{}".format(det, n_m),
            "{}/{}".format(fp, n_b)))

    # TPR by attack type x format
    print("\n── TPR by Attack Type × Format ──")
    header = "{:<28}".format("Attack Type") + "".join("{:>16}".format(f[:14]) for f in formats)
    print(header)
    print("-" * (28 + 16 * len(formats)))
    for at in [1, 2, 3, 4, 5]:
        row = "{:<28}".format(ATTACK_TYPE_NAMES[at])
        for fmt in formats:
            res = [r for r in all_results if r["format"] == fmt and r["attack_type"] == at]
            tpr, det, n_m = compute_tpr(res)
            row += "{:>16}".format("{:.1%}".format(tpr) if tpr is not None else "N/A")
        print(row)

    # FPR by attack type x format
    print("\n── FPR by Attack Type × Format ──")
    print(header)
    print("-" * (28 + 16 * len(formats)))
    for at in [1, 2, 3, 4, 5]:
        row = "{:<28}".format(ATTACK_TYPE_NAMES[at])
        for fmt in formats:
            res = [r for r in all_results if r["format"] == fmt and r["attack_type"] == at]
            fpr, fp, n_b = compute_fpr(res)
            row += "{:>16}".format("{:.1%}".format(fpr) if fpr is not None else "N/A")
        print(row)

    # TPR / FPR by urgency level (full_context only)
    fc = [r for r in all_results if r["format"] == "full_context"]
    urgency_levels = sorted(set(r["urgency"] for r in fc if r["urgency"]))
    if urgency_levels:
        print("\n── TPR / FPR by Urgency Level (full_context) ──")
        print("{:<20} {:>10} {:>10}".format("Urgency", "TPR", "FPR"))
        print("-" * 42)
        for urg in urgency_levels:
            res = [r for r in fc if r["urgency"] == urg]
            tpr, _, _ = compute_tpr(res)
            fpr, _, _ = compute_fpr(res)
            print("{:<20} {:>10} {:>10}".format(
                urg,
                "{:.1%}".format(tpr) if tpr is not None else "N/A",
                "{:.1%}".format(fpr) if fpr is not None else "N/A"))

    # TPR / FPR by reference style (full_context only)
    ref_styles = sorted(set(r["reference_style"] for r in fc if r["reference_style"]))
    if ref_styles:
        print("\n── TPR / FPR by Reference Style (full_context) ──")
        print("{:<30} {:>10} {:>10}".format("Reference Style", "TPR", "FPR"))
        print("-" * 52)
        for rs in ref_styles:
            res = [r for r in fc if r["reference_style"] == rs]
            tpr, _, _ = compute_tpr(res)
            fpr, _, _ = compute_fpr(res)
            print("{:<30} {:>10} {:>10}".format(
                rs,
                "{:.1%}".format(tpr) if tpr is not None else "N/A",
                "{:.1%}".format(fpr) if fpr is not None else "N/A"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    set_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    # Load model + detector
    model_config = open_config(config_path="./configs/model_configs/{}_config.json".format(args.model_name))
    model = create_model(config=model_config)
    model.print_model_info()
    detector = AttentionDetector(model)
    print("Detector: {}".format(detector.name))

    # Load expanded dataset
    master_path = args.master_path
    print("Loading expanded dataset from {}...".format(master_path))
    mal_scenarios, ben_scenarios = load_expanded_dataset(
        master_path, sample=args.sample, seed=args.seed)
    print("Loaded {} malicious + {} benign variants".format(
        len(mal_scenarios), len(ben_scenarios)))

    # Run evaluation across all formats
    all_results = []

    print("\nEvaluating malicious variants...")
    for fmt_name, fmt_builder in FORMAT_BUILDERS.items():
        results = run_evaluation(detector, mal_scenarios, fmt_name, fmt_builder)
        all_results.extend(results)

    print("Evaluating benign variants...")
    for fmt_name, fmt_builder in FORMAT_BUILDERS.items():
        results = run_evaluation(detector, ben_scenarios, fmt_name, fmt_builder)
        all_results.extend(results)

    print_summary(all_results, len(mal_scenarios), len(ben_scenarios))

    # Save results
    tag = "sample{}".format(args.sample) if args.sample else "full"
    out_path = "./result/agentic_pi_eval/{}-expanded-{}-seed{}.json".format(
        args.model_name, tag, args.seed)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nResults saved to: {}".format(out_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate AttentionTracker on the full expanded dataset")
    parser.add_argument("--model_name", type=str, default="qwen2-attn")
    parser.add_argument("--gpu_id", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--master_path", type=str,
                        default="./variations/variants_master.json")
    parser.add_argument("--sample", type=int, default=None,
                        help="Evaluate on a random sample of N pairs instead of all 1575 (for testing)")
    args = parser.parse_args()
    main(args)