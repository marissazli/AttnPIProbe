"""
eval_attntrace_expanded.py

Evaluates AttnTrace on the full expanded dataset:
  - 1,680 malicious variants (105 seeds + 1,575 expanded)
  - Loaded from variations/variants_master.json

AttnTrace is a context traceback method: given (question, contexts, answer),
it identifies which context segment caused the answer.

Ground truth: attack_observation_idx — the step containing the payload.

For expanded variants, attack_payload is used as the answer proxy
(the injected instruction text the model would follow if susceptible).
For seed scenarios, target_answer is used where available.

Metrics:
  - Top-1/2/3 accuracy
  - MRR
  - Score gap (attack obs score minus max non-attack score)

Breakdown by:
  - Context format (observations_only, labeled_segments)
  - Attack type
  - Urgency level
  - Reference style
  - Attack observation position (position bias analysis)
  - Scenario step count

Usage:
    python eval_attntrace_expanded.py --model_name qwen2.5-7b --gpu_id 2
    python eval_attntrace_expanded.py --model_name qwen2.5-7b --gpu_id 2 --sample 200
"""

import argparse
import json
import os
import random
import types
import time
import torch
import numpy as np
from tqdm import tqdm

from src.models import create_model
from src.attribution import create_attr
from src.prompts import wrap_prompt


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
    Load malicious variants from variants_master.json.
    Normalises to the field names AttnTrace expects.
    Only malicious scenarios are evaluated (AttnTrace is a localization method,
    not a binary detector — we measure whether it correctly identifies the
    attack observation).
    """
    with open(master_path) as f:
        master = json.load(f)

    malicious = master["malicious"]

    if sample is not None:
        rng = random.Random(seed)
        indices = rng.sample(range(len(malicious)), min(sample, len(malicious)))
        malicious = [malicious[i] for i in indices]

    normalised = []
    for s in malicious:
        attack_type = s.get("attack_type") or s.get("attack", 0)
        # Use target_answer if present (seed scenarios), else attack_payload
        answer = s.get("target_answer") or s.get("attack_payload", "")
        normalised.append({
            "id": s["id"],
            "seed_id": s.get("seed_id", s["id"]),
            "domain": s.get("domain", ""),
            "attack_type": attack_type,
            "title": s.get("title", s["id"]),
            "user_prompt": s["user_prompt"],
            "steps": s["steps"],
            "attack_observation_idx": s["attack_observation_idx"],
            "target_answer": answer,
            "urgency": s.get("urgency", "seed"),
            "reference_style": s.get("reference_style", "seed"),
            "n_steps": len(s["steps"]),
        })

    return normalised


# ---------------------------------------------------------------------------
# Context format builders
# ---------------------------------------------------------------------------

def build_observations_only(scenario):
    return [step["observation"] for step in scenario["steps"]]


def build_labeled_segments(scenario):
    segments = []
    for step in scenario["steps"]:
        prefix = "[Step {} | Tool: {}]\n".format(step["step"], step["tool"])
        segments.append(prefix + step["observation"])
    return segments


FORMAT_BUILDERS = {
    "observations_only": build_observations_only,
    "labeled_segments":  build_labeled_segments,
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_rank(importance_scores, attack_idx):
    ranked = sorted(range(len(importance_scores)),
                    key=lambda i: importance_scores[i], reverse=True)
    return ranked.index(attack_idx) + 1


def compute_score_gap(importance_scores, attack_idx):
    attack_score = importance_scores[attack_idx]
    others = [s for i, s in enumerate(importance_scores) if i != attack_idx]
    return attack_score - max(others) if others else 0.0


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def run_attntrace_eval(attr, scenarios, fmt_name, fmt_builder, verbose=False):
    results = []
    for s in tqdm(scenarios, desc=fmt_name, leave=False):
        question = s["user_prompt"]
        contexts = fmt_builder(s)
        answer = s["target_answer"]
        attack_idx = s["attack_observation_idx"]

        try:
            texts, important_ids, importance_scores, elapsed, _ = attr.attribute(
                question, contexts, answer
            )
        except Exception as e:
            print("  ERROR on {}: {}".format(s["id"], e))
            results.append({
                "id": s["id"], "seed_id": s["seed_id"],
                "domain": s["domain"], "attack_type": s["attack_type"],
                "format": fmt_name, "urgency": s["urgency"],
                "reference_style": s["reference_style"],
                "n_steps": s["n_steps"], "attack_idx": attack_idx,
                "rank": None, "top1": False, "top2": False, "top3": False,
                "rr": 0.0, "score_gap": None, "error": str(e),
            })
            continue

        n_segments = len(contexts)
        seg_scores = [0.0] * n_segments
        if len(importance_scores) == n_segments:
            seg_scores = list(importance_scores)
        else:
            for seg_i in range(n_segments):
                seg_text = contexts[seg_i]
                for j, text_unit in enumerate(texts):
                    if text_unit.strip() in seg_text or seg_text.startswith(text_unit[:30]):
                        if j < len(importance_scores):
                            seg_scores[seg_i] = max(seg_scores[seg_i], importance_scores[j])

        rank = compute_rank(seg_scores, attack_idx)
        gap = compute_score_gap(seg_scores, attack_idx)
        rr = 1.0 / rank

        if verbose:
            print("  {} | rank={} | gap={:.4f} | scores={}".format(
                s["id"], rank, gap, [round(x, 4) for x in seg_scores]))

        results.append({
            "id": s["id"], "seed_id": s["seed_id"],
            "domain": s["domain"], "attack_type": s["attack_type"],
            "format": fmt_name, "urgency": s["urgency"],
            "reference_style": s["reference_style"],
            "n_steps": s["n_steps"], "attack_idx": attack_idx,
            "rank": rank, "top1": rank == 1, "top2": rank <= 2, "top3": rank <= 3,
            "rr": rr, "score_gap": round(gap, 6),
            "importance_scores": [round(x, 6) for x in seg_scores],
            "error": None,
        })
    return results


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def agg(results):
    """Compute Top-1/2/3, MRR, AvgGap for a list of results."""
    valid = [r for r in results if r["rank"] is not None]
    if not valid:
        return None
    n = len(valid)
    return {
        "n": n,
        "top1": sum(r["top1"] for r in valid) / n,
        "top2": sum(r["top2"] for r in valid) / n,
        "top3": sum(r["top3"] for r in valid) / n,
        "mrr":  sum(r["rr"] for r in valid) / n,
        "gap":  sum(r["score_gap"] for r in valid if r["score_gap"] is not None) / n,
    }


def print_summary(all_results, n_total):
    formats = list(FORMAT_BUILDERS.keys())

    print("\n" + "=" * 70)
    print("AttnTrace — Expanded Dataset ({} malicious scenarios)".format(n_total))
    print("=" * 70)

    # Overall by format
    print("\n── Overall by Context Format ──")
    print("{:<22} {:>8} {:>8} {:>8} {:>8} {:>10} {:>6}".format(
        "Format", "Top-1", "Top-2", "Top-3", "MRR", "AvgGap", "N"))
    print("-" * 72)
    for fmt in formats:
        res = [r for r in all_results if r["format"] == fmt]
        m = agg(res)
        if m:
            print("{:<22} {:>8.1%} {:>8.1%} {:>8.1%} {:>8.3f} {:>10.4f} {:>6}".format(
                fmt, m["top1"], m["top2"], m["top3"], m["mrr"], m["gap"], m["n"]))

    # Top-1 by attack type x format
    print("\n── Top-1 by Attack Type × Format ──")
    header = "{:<28}".format("Attack Type") + "".join("{:>18}".format(f[:16]) for f in formats)
    print(header)
    print("-" * (28 + 18 * len(formats)))
    for at in [1, 2, 3, 4, 5]:
        row = "{:<28}".format(ATTACK_TYPE_NAMES[at])
        for fmt in formats:
            res = [r for r in all_results if r["format"] == fmt and r["attack_type"] == at]
            m = agg(res)
            row += "{:>18}".format("{:.1%} (n={})".format(m["top1"], m["n"]) if m else "N/A")
        print(row)

    # Position bias: Top-1 by attack_observation_idx (labeled_segments only)
    ls = [r for r in all_results if r["format"] == "labeled_segments"]
    positions = sorted(set(r["attack_idx"] for r in ls))
    if len(positions) > 1:
        print("\n── Position Bias: Top-1 by Attack Observation Index (labeled_segments) ──")
        print("{:<12} {:>10} {:>10} {:>8}".format("Atk obs idx", "Top-1", "MRR", "N"))
        print("-" * 42)
        for pos in positions:
            res = [r for r in ls if r["attack_idx"] == pos]
            m = agg(res)
            if m:
                print("{:<12} {:>10.1%} {:>10.3f} {:>8}".format(
                    pos, m["top1"], m["mrr"], m["n"]))

    # Step count breakdown (labeled_segments)
    step_counts = sorted(set(r["n_steps"] for r in ls))
    if len(step_counts) > 1:
        print("\n── Top-1 by Step Count (labeled_segments) ──")
        print("{:<12} {:>10} {:>10} {:>8}".format("N steps", "Top-1", "MRR", "N"))
        print("-" * 42)
        for n in step_counts:
            res = [r for r in ls if r["n_steps"] == n]
            m = agg(res)
            if m:
                print("{:<12} {:>10.1%} {:>10.3f} {:>8}".format(
                    n, m["top1"], m["mrr"], m["n"]))

    # Urgency breakdown (labeled_segments)
    urgency_levels = sorted(set(r["urgency"] for r in ls if r["urgency"]))
    if urgency_levels:
        print("\n── Top-1 / MRR by Urgency Level (labeled_segments) ──")
        print("{:<20} {:>10} {:>10} {:>8}".format("Urgency", "Top-1", "MRR", "N"))
        print("-" * 50)
        for urg in urgency_levels:
            res = [r for r in ls if r["urgency"] == urg]
            m = agg(res)
            if m:
                print("{:<20} {:>10.1%} {:>10.3f} {:>8}".format(
                    urg, m["top1"], m["mrr"], m["n"]))

    # Reference style breakdown (labeled_segments)
    ref_styles = sorted(set(r["reference_style"] for r in ls if r["reference_style"]))
    if ref_styles:
        print("\n── Top-1 / MRR by Reference Style (labeled_segments) ──")
        print("{:<30} {:>10} {:>10} {:>8}".format("Reference Style", "Top-1", "MRR", "N"))
        print("-" * 60)
        for rs in ref_styles:
            res = [r for r in ls if r["reference_style"] == rs]
            m = agg(res)
            if m:
                print("{:<30} {:>10.1%} {:>10.3f} {:>8}".format(
                    rs, m["top1"], m["mrr"], m["n"]))

    # MRR interpretation
    overall = agg(ls)
    if overall:
        print("\n── Interpretation (labeled_segments) ──")
        print("MRR: {:.3f}".format(overall["mrr"]))
        print("  MRR=1.0 -> always ranks attack observation #1 (perfect)")
        print("  MRR=0.5 -> attack observation ranked #2 on average")
        print("  MRR=0.33 -> attack observation ranked #3 on average")
        if overall["mrr"] < 0.5:
            print("  -> AttnTrace struggles to localise the poisoned observation")
        else:
            print("  -> AttnTrace successfully traces back to the poisoned observation")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def setup_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    setup_seeds(args.seed)

    print("Loading model: {} on GPU {}".format(args.model_name, args.gpu_id))
    llm = create_model(
        config_path="model_configs/{}_config.json".format(args.model_name),
        device="cuda:{}".format(args.gpu_id)
    )

    attr_args = types.SimpleNamespace(
        attr_type="attntrace",
        explanation_level=args.explanation_level,
        K=args.K,
        avg_k=args.avg_k,
        q=args.q,
        B=args.B,
        verbose=0,
    )
    print("Initializing AttnTrace (explanation_level={}, K={}, q={}, B={})".format(
        args.explanation_level, args.K, args.q, args.B))
    attr = create_attr(attr_args, llm=llm)

    print("Loading expanded dataset from {}...".format(args.master_path))
    scenarios = load_expanded_dataset(
        args.master_path, sample=args.sample, seed=args.seed)
    print("Loaded {} malicious scenarios".format(len(scenarios)))

    tag = "sample{}".format(args.sample) if args.sample else "full"
    output_path = "./result/agentic_pi_attntrace/{}-expanded-{}-seed{}.json".format(
        args.model_name, tag, args.seed)
    checkpoint_path = output_path.replace(".json", "_checkpoint.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load checkpoint if resuming
    all_results = []
    completed_formats = set()
    if os.path.exists(checkpoint_path) and not args.overwrite:
        with open(checkpoint_path) as f:
            all_results = json.load(f)
        completed_formats = set(r["format"] for r in all_results if r.get("rank") is not None or r.get("error"))
        print("Resuming from checkpoint: {} results, formats done: {}".format(
            len(all_results), completed_formats))

    for fmt_name, fmt_builder in FORMAT_BUILDERS.items():
        if fmt_name in completed_formats:
            print("\nSkipping format {} (already in checkpoint)".format(fmt_name))
            continue

        print("\nRunning format: {}".format(fmt_name))
        fmt_results = []
        for i, s in enumerate(tqdm(scenarios, desc=fmt_name, leave=False)):
            question = s["user_prompt"]
            contexts = fmt_builder(s)
            answer = s["target_answer"]
            attack_idx = s["attack_observation_idx"]
            try:
                texts, important_ids, importance_scores, elapsed, _ = attr.attribute(
                    question, contexts, answer)
                n_segments = len(contexts)
                seg_scores = [0.0] * n_segments
                if len(importance_scores) == n_segments:
                    seg_scores = list(importance_scores)
                else:
                    for seg_i in range(n_segments):
                        seg_text = contexts[seg_i]
                        for j, text_unit in enumerate(texts):
                            if text_unit.strip() in seg_text or seg_text.startswith(text_unit[:30]):
                                if j < len(importance_scores):
                                    seg_scores[seg_i] = max(seg_scores[seg_i], importance_scores[j])
                rank = compute_rank(seg_scores, attack_idx)
                gap = compute_score_gap(seg_scores, attack_idx)
                fmt_results.append({
                    "id": s["id"], "seed_id": s["seed_id"],
                    "domain": s["domain"], "attack_type": s["attack_type"],
                    "format": fmt_name, "urgency": s["urgency"],
                    "reference_style": s["reference_style"],
                    "n_steps": s["n_steps"], "attack_idx": attack_idx,
                    "rank": rank, "top1": rank == 1, "top2": rank <= 2, "top3": rank <= 3,
                    "rr": 1.0 / rank, "score_gap": round(gap, 6),
                    "importance_scores": [round(x, 6) for x in seg_scores],
                    "error": None,
                })
            except Exception as e:
                print("  ERROR on {}: {}".format(s["id"], e))
                fmt_results.append({
                    "id": s["id"], "seed_id": s["seed_id"],
                    "domain": s["domain"], "attack_type": s["attack_type"],
                    "format": fmt_name, "urgency": s["urgency"],
                    "reference_style": s["reference_style"],
                    "n_steps": s["n_steps"], "attack_idx": attack_idx,
                    "rank": None, "top1": False, "top2": False, "top3": False,
                    "rr": 0.0, "score_gap": None, "error": str(e),
                })

            # Checkpoint every 100 scenarios
            if (i + 1) % 100 == 0:
                with open(checkpoint_path, "w") as f:
                    json.dump(all_results + fmt_results, f)
                print("  Checkpoint saved at {}/{} ({})".format(i + 1, len(scenarios), fmt_name))

        all_results.extend(fmt_results)
        # Save checkpoint after each format completes
        with open(checkpoint_path, "w") as f:
            json.dump(all_results, f)
        print("  Format {} complete, checkpoint saved".format(fmt_name))

    print_summary(all_results, len(scenarios))
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nResults saved to: {}".format(output_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate AttnTrace on the full expanded dataset")
    parser.add_argument("--model_name", type=str, default="qwen2.5-7b",
                        choices=["llama3.1-8b", "llama3.1-70b", "qwen2-7b", "qwen2.5-7b",
                                 "gpt4o-mini", "gpt4.1-mini", "gemini-2.0-flash",
                                 "claude-haiku-3.5", "deepseek-v3"])
    parser.add_argument("--gpu_id", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--master_path", type=str,
                        default="./variations/variants_master.json")
    parser.add_argument("--sample", type=int, default=None,
                        help="Evaluate on N randomly sampled scenarios (for testing)")
    parser.add_argument("--explanation_level", type=str, default="segment",
                        choices=["sentence", "paragraph", "segment"])
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--avg_k", type=int, default=5)
    parser.add_argument("--q", type=float, default=0.4)
    parser.add_argument("--B", type=int, default=30)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--overwrite", action="store_true",
                        help="Ignore checkpoint and start fresh")
    args = parser.parse_args()
    main(args)