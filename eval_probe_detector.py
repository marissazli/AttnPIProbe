"""
eval_probe_detector.py

Trains a logistic regression probe on Qwen2.5-7B hidden states
and evaluates it as a prompt injection detector.

PRIMARY EVALUATION (--split seed_to_variant):
  Train: 105 seed hidden states (mal) + 105 seed hidden states (ben)
  Test:  1,575 variant hidden states (mal) + 1,575 variant hidden states (ben)
  -> Zero leakage: probe trained on seeds, tested on unseen surface variations

SECONDARY EVALUATION (--split grouped):
  70/30 seed-grouped split — all variants of a seed go to same partition

Both compare directly against:
  - AttentionTracker: 100% TPR / ~100% FPR (full_context format)
  - AttnTrace:        79.0% Top-1 (position bias artefact)

Usage:
    python eval_probe_detector.py --all_layers
    python eval_probe_detector.py --layer 4 --split seed_to_variant
    python eval_probe_detector.py --layer 4 --split grouped
"""

import argparse
import json
import os
import random
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix
)


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

def infer_attack_type(s):
    at = s.get("attack_type") or s.get("attack")
    if at:
        return int(at)
    sid = s.get("seed_id") or s.get("id", "")
    parts = sid.split("-")
    if len(parts) >= 2:
        try:
            return int(parts[1])
        except ValueError:
            pass
    return 0


def load_data(hidden_path, master_path):
    print("Loading hidden states from {}...".format(hidden_path))
    cached = np.load(hidden_path, allow_pickle=True)
    all_hidden = cached["all_hidden"]   # (n_obs, n_layers, hidden_dim)
    layer_indices = list(cached["layer_indices"])
    print("Hidden states: {} | layers: {}".format(all_hidden.shape, layer_indices))

    print("Loading scenario metadata from {}...".format(master_path))
    with open(master_path) as f:
        master = json.load(f)

    malicious = master["malicious"]
    benign = master["benign"]

    metadata = []
    for mal, ben in zip(malicious, benign):
        attack_type = infer_attack_type(mal)
        if attack_type == 0:
            continue
        mal_id = mal["id"]
        is_seed = "-V" not in mal_id

        metadata.append({
            "id": mal_id,
            "seed_id": mal.get("seed_id", mal_id),
            "is_seed": is_seed,
            "domain": mal.get("domain", ""),
            "attack_type": attack_type,
            "label": 1,
            "urgency": mal.get("urgency", "seed"),
            "reference_style": mal.get("reference_style", "seed"),
        })
        metadata.append({
            "id": ben["id"],
            "seed_id": ben.get("seed_id", ben["id"]),
            "is_seed": is_seed,
            "domain": ben.get("domain", mal.get("domain", "")),
            "attack_type": attack_type,
            "label": 0,
            "urgency": ben.get("urgency", "seed"),
            "reference_style": ben.get("reference_style", "seed"),
        })

    assert len(metadata) == all_hidden.shape[0], \
        "Metadata {} != hidden states {}".format(len(metadata), all_hidden.shape[0])

    labels = np.array([m["label"] for m in metadata])
    return all_hidden, layer_indices, labels, metadata


def get_split_seed_to_variant(metadata):
    """
    Train: all seed observations (is_seed=True)
    Test:  all variant observations (is_seed=False)
    Zero leakage — probe never sees any variant of any test scenario.
    """
    train_idx = [i for i, m in enumerate(metadata) if m["is_seed"]]
    test_idx  = [i for i, m in enumerate(metadata) if not m["is_seed"]]
    return train_idx, test_idx


def get_split_grouped(metadata, test_size=0.3, seed=0):
    """
    Seed-grouped 70/30 split — all variants of a seed go to same partition.
    """
    rng = random.Random(seed)
    seed_groups = defaultdict(list)
    for i, m in enumerate(metadata):
        base = m["seed_id"].split("-V")[0] if "-V" in m["seed_id"] else m["seed_id"]
        seed_groups[base].append(i)

    all_seeds = list(seed_groups.keys())
    rng.shuffle(all_seeds)
    n_test = int(len(all_seeds) * test_size)
    test_seeds = set(all_seeds[:n_test])

    train_idx = [i for i, m in enumerate(metadata)
                 if (m["seed_id"].split("-V")[0] if "-V" in m["seed_id"] else m["seed_id"]) not in test_seeds]
    test_idx  = [i for i, m in enumerate(metadata)
                 if (m["seed_id"].split("-V")[0] if "-V" in m["seed_id"] else m["seed_id"]) in test_seeds]
    return train_idx, test_idx


# ---------------------------------------------------------------------------
# Probe training and evaluation
# ---------------------------------------------------------------------------

def train_probe(X_train, y_train):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs"))
    ])
    pipe.fit(X_train, y_train)
    return pipe


def evaluate_probe(pipe, X_test, y_test, metadata_test):
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    results = {
        "overall": {
            "tpr": round(tpr, 4), "fpr": round(fpr, 4),
            "accuracy": round(acc, 4), "auc": round(auc, 4),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "n_test": len(y_test),
        },
        "by_attack_type": {},
        "by_urgency": {},
        "by_reference_style": {},
    }

    def breakdown(key, values):
        out = {}
        for val in sorted(set(values)):
            idx = [i for i, m in enumerate(metadata_test) if m[key] == val]
            if len(idx) < 4:
                continue
            y_t = y_test[idx]
            y_p = y_pred[idx]
            y_pb = y_prob[idx]
            if len(set(y_t)) < 2:
                continue
            tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_t, y_p).ravel()
            tpr_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0.0
            fpr_t = fp_t / (fp_t + tn_t) if (fp_t + tn_t) > 0 else 0.0
            out[str(val)] = {
                "tpr": round(tpr_t, 4), "fpr": round(fpr_t, 4),
                "auc": round(roc_auc_score(y_t, y_pb), 4),
                "n": len(idx),
            }
        return out

    results["by_attack_type"] = breakdown(
        "attack_type", [m["attack_type"] for m in metadata_test])
    results["by_urgency"] = breakdown(
        "urgency", [m["urgency"] for m in metadata_test])
    results["by_reference_style"] = breakdown(
        "reference_style", [m["reference_style"] for m in metadata_test])

    return results


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_results(layer, eval_results, n_train, n_test, split_name):
    o = eval_results["overall"]
    print("\n" + "=" * 65)
    print("Probe Detector | Layer {} | Split: {} | train={} test={}".format(
        layer, split_name, n_train, n_test))
    print("=" * 65)
    print("\n── Overall ──")
    print("  TPR:      {:.1%}  ({}/{})".format(o["tpr"], o["tp"], o["tp"] + o["fn"]))
    print("  FPR:      {:.1%}  ({}/{})".format(o["fpr"], o["fp"], o["fp"] + o["tn"]))
    print("  Accuracy: {:.1%}".format(o["accuracy"]))
    print("  ROC AUC:  {:.4f}".format(o["auc"]))
    print("\n  [Baseline] AttentionTracker full_context: TPR=100.0% FPR=100.0%")
    print("  [Baseline] AttnTrace Top-1 (labeled_segments): 79.0% (position bias)")

    print("\n── TPR / FPR by Attack Type ──")
    print("{:<28} {:>8} {:>8} {:>8} {:>6}".format(
        "Attack Type", "TPR", "FPR", "AUC", "N"))
    print("-" * 62)
    for at in [1, 2, 3, 4, 5]:
        r = eval_results["by_attack_type"].get(str(at))
        if r:
            print("{:<28} {:>8.1%} {:>8.1%} {:>8.4f} {:>6}".format(
                ATTACK_TYPE_NAMES[at], r["tpr"], r["fpr"], r["auc"], r["n"]))

    if eval_results["by_urgency"]:
        print("\n── TPR / FPR by Urgency ──")
        print("{:<20} {:>8} {:>8} {:>6}".format("Urgency", "TPR", "FPR", "N"))
        print("-" * 44)
        for urg, r in sorted(eval_results["by_urgency"].items()):
            print("{:<20} {:>8.1%} {:>8.1%} {:>6}".format(
                urg, r["tpr"], r["fpr"], r["n"]))

    if eval_results["by_reference_style"]:
        print("\n── TPR / FPR by Reference Style ──")
        print("{:<30} {:>8} {:>8} {:>6}".format("Reference Style", "TPR", "FPR", "N"))
        print("-" * 54)
        for rs, r in sorted(eval_results["by_reference_style"].items()):
            print("{:<30} {:>8.1%} {:>8.1%} {:>6}".format(
                rs, r["tpr"], r["fpr"], r["n"]))


def print_layer_comparison(all_layer_results, split_name):
    print("\n" + "=" * 65)
    print("Probe Detector — TPR / FPR by Layer | Split: {}".format(split_name))
    print("Baseline: AttentionTracker = 100% TPR / ~100% FPR")
    print("=" * 65)
    print("{:<8} {:>10} {:>10} {:>10}".format("Layer", "TPR", "FPR", "AUC"))
    print("-" * 42)
    for layer in sorted(all_layer_results.keys()):
        o = all_layer_results[layer]["overall"]
        print("{:<8} {:>10.1%} {:>10.1%} {:>10.4f}".format(
            layer, o["tpr"], o["fpr"], o["auc"]))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    np.random.seed(args.seed)
    random.seed(args.seed)

    all_hidden, layer_indices, labels, metadata = load_data(
        args.hidden_path, args.master_path)

    n_seeds = sum(1 for m in metadata if m["is_seed"])
    n_variants = sum(1 for m in metadata if not m["is_seed"])
    print("Seeds: {} | Variants: {} | Total: {}".format(
        n_seeds, n_variants, len(metadata)))

    # Get split
    if args.split == "seed_to_variant":
        train_idx, test_idx = get_split_seed_to_variant(metadata)
        split_name = "seed->variant (train=seeds, test=variants)"
    else:
        train_idx, test_idx = get_split_grouped(metadata, seed=args.seed)
        split_name = "seed-grouped 70/30"

    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)
    y_train = labels[train_idx]
    y_test = labels[test_idx]
    metadata_test = [metadata[i] for i in test_idx]

    print("Train: {} | Test: {}".format(len(train_idx), len(test_idx)))
    print("Train label balance: {:.1%} positive".format(y_train.mean()))
    print("Test label balance:  {:.1%} positive".format(y_test.mean()))

    # Layers to evaluate
    if args.all_layers:
        eval_layer_indices = list(range(len(layer_indices)))
    else:
        if args.layer in layer_indices:
            eval_layer_indices = [layer_indices.index(args.layer)]
        else:
            print("Layer {} not found, using layer {}".format(
                args.layer, layer_indices[1]))
            eval_layer_indices = [1]

    all_layer_results = {}
    for li in eval_layer_indices:
        layer = layer_indices[li]
        print("\nTraining probe at layer {}...".format(layer))
        X_train = all_hidden[train_idx, li, :]
        X_test = all_hidden[test_idx, li, :]
        pipe = train_probe(X_train, y_train)
        eval_results = evaluate_probe(pipe, X_test, y_test, metadata_test)
        all_layer_results[layer] = eval_results
        print_results(layer, eval_results, len(train_idx), len(test_idx), split_name)

    if args.all_layers:
        print_layer_comparison(all_layer_results, split_name)

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    layers_tag = "all_layers" if args.all_layers else "layer{}".format(args.layer)
    out_path = os.path.join(
        args.output_dir,
        "probe_detector_{}_{}_{}.json".format(
            args.split, layers_tag, args.seed))
    def clean(obj):
        if isinstance(obj, dict): return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list): return [clean(v) for v in obj]
        if hasattr(obj, "item"): return obj.item()
        return obj

    with open(out_path, "w") as f:
        json.dump(clean({
            "split": args.split,
            "train_size": len(train_idx),
            "test_size": len(test_idx),
            "layer_indices": layer_indices,
            "results_by_layer": {str(k): v for k, v in all_layer_results.items()},
        }), f, indent=2)
    print("\nResults saved to: {}".format(out_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Probe detector: train on seeds, test on variants")
    parser.add_argument("--hidden_path", type=str,
                        default="./phase2_results_expanded/qwen2.5-7b-expanded-n3360-hidden.npz")
    parser.add_argument("--master_path", type=str,
                        default="./variations/variants_master.json")
    parser.add_argument("--output_dir", type=str,
                        default="./result/probe_detector")
    parser.add_argument("--layer", type=int, default=4)
    parser.add_argument("--all_layers", action="store_true",
                        help="Evaluate all extracted layers")
    parser.add_argument("--split", type=str, default="seed_to_variant",
                        choices=["seed_to_variant", "grouped"],
                        help="seed_to_variant: train=seeds test=variants (default); "
                             "grouped: 70/30 seed-grouped split")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args)