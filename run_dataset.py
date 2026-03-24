import argparse
import os
import json
import random
import math
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from utils import open_config, create_model, get_retrieval_config
from detector.attn import AttentionDetector
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from retrieval.data_loader import load_local_queries, load_local_corpus
from retrieval.retriever import LexicalRetriever


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_benchmark_eval(args, detector):
    output_logs = f"./result/{args.dataset_name}/{args.model_name}-{args.seed}.json"
    output_result = f"./result/{args.dataset_name}/result.jsonl"

    dataset = load_dataset(args.dataset_name)
    test_data = dataset["test"]

    labels, predictions, scores = [], [], []
    logs = []

    for data in tqdm(test_data):
        result = detector.detect(data["text"])
        detect = result[0]
        score = result[1]["focus_score"]

        labels.append(data["label"])
        predictions.append(detect)
        scores.append(1 - score)

        logs.append(
            {
                "text": data["text"],
                "label": data["label"],
                "result": result,
            }
        )

    auc_score = roc_auc_score(labels, scores)
    auprc_score = average_precision_score(labels, scores)

    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    auc_score = round(auc_score, 3)
    auprc_score = round(auprc_score, 3)
    fnr = round(fnr, 3)
    fpr = round(fpr, 3)

    print(f"AUC Score: {auc_score}; AUPRC Score: {auprc_score}; FNR: {fnr}; FPR: {fpr}")

    os.makedirs(os.path.dirname(output_logs), exist_ok=True)
    with open(output_logs, "w", encoding="utf-8") as f_out:
        f_out.write(json.dumps({"result": logs}, indent=4))

    os.makedirs(os.path.dirname(output_result), exist_ok=True)
    with open(output_result, "a", encoding="utf-8") as f_out:
        f_out.write(
            json.dumps(
                {
                    "model": args.model_name,
                    "seed": args.seed,
                    "auc": auc_score,
                    "auprc": auprc_score,
                    "fnr": fnr,
                    "fpr": fpr,
                }
            )
            + "\n"
        )


def _build_adversarial_prompt(normal_prompt, adversarial_prompt, suffix):
    if adversarial_prompt:
        return adversarial_prompt
    return normal_prompt + " " + suffix.strip()


def _build_external_context(retriever, prompt, top_k):
    retrieved = retriever.retrieve(prompt, top_k=top_k)
    context_chunks = [doc["text"] for doc in retrieved]
    return "\n\n".join(context_chunks), retrieved


def _calc_focus_score(model, detector, prompt, external_context=None):
    _, _, attention_maps, _, input_range, _ = model.inference(
        detector.instruction,
        prompt,
        external_context=external_context,
        max_output_tokens=1,
    )
    return detector.attn2score(attention_maps, input_range)


def _safe_mean(values):
    return float(np.mean(values)) if values else 0.0


def _safe_std(values):
    return float(np.std(values)) if values else 0.0


def _compute_condition_stats(values):
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": round(float(np.mean(values)), 6),
        "std": round(float(np.std(values)), 6),
        "min": round(float(np.min(values)), 6),
        "max": round(float(np.max(values)), 6),
    }


def _build_threshold_grid(min_v, max_v, step_v):
    if step_v <= 0:
        raise ValueError("--threshold_sweep_step must be > 0")
    if max_v < min_v:
        raise ValueError("--threshold_sweep_max must be >= --threshold_sweep_min")

    n_steps = int(math.floor((max_v - min_v) / step_v)) + 1
    grid = [round(min_v + i * step_v, 6) for i in range(n_steps)]
    if grid[-1] < round(max_v, 6):
        grid.append(round(max_v, 6))
    return grid


def _compute_threshold_sweep(benign_scores, adversarial_scores, thresholds):
    rows = []
    best = {
        "threshold": None,
        "youden_j": float("-inf"),
        "balanced_accuracy": float("-inf"),
        "tpr": 0.0,
        "fpr": 0.0,
    }
    benign_n = max(1, len(benign_scores))
    adv_n = max(1, len(adversarial_scores))

    for t in thresholds:
        fp = sum(1 for s in benign_scores if s <= t)
        tp = sum(1 for s in adversarial_scores if s <= t)
        fpr = fp / benign_n
        tpr = tp / adv_n
        youden_j = tpr - fpr
        bal_acc = 0.5 * (tpr + (1.0 - fpr))
        row = {
            "threshold": round(float(t), 6),
            "tpr": round(float(tpr), 6),
            "fpr": round(float(fpr), 6),
            "youden_j": round(float(youden_j), 6),
            "balanced_accuracy": round(float(bal_acc), 6),
        }
        rows.append(row)
        if (youden_j > best["youden_j"]) or (
            youden_j == best["youden_j"] and bal_acc > best["balanced_accuracy"]
        ):
            best = {
                "threshold": row["threshold"],
                "youden_j": youden_j,
                "balanced_accuracy": bal_acc,
                "tpr": tpr,
                "fpr": fpr,
            }

    return {
        "rows": rows,
        "best_by_youden": {
            "threshold": round(float(best["threshold"]), 6)
            if best["threshold"] is not None
            else None,
            "youden_j": round(float(best["youden_j"]), 6)
            if best["threshold"] is not None
            else 0.0,
            "balanced_accuracy": round(float(best["balanced_accuracy"]), 6)
            if best["threshold"] is not None
            else 0.0,
            "tpr": round(float(best["tpr"]), 6),
            "fpr": round(float(best["fpr"]), 6),
        },
    }


def _aggregate_group_stats(logs, group_key):
    groups = {}
    for row in logs:
        key = row.get(group_key, "") or "unknown"
        if key not in groups:
            groups[key] = {
                "normal_noext": [],
                "adversarial_noext": [],
                "normal_ext": [],
                "adversarial_ext": [],
            }
        fs = row["focus_scores"]
        for cond in groups[key]:
            groups[key][cond].append(fs[cond])

    out = {}
    for key, vals in groups.items():
        out[key] = {
            "count": len(vals["normal_noext"]),
            "mean_focus_scores": {
                cond: round(_safe_mean(cond_vals), 6) for cond, cond_vals in vals.items()
            },
            "separation_noext_adv_minus_normal": round(
                _safe_mean(vals["adversarial_noext"]) - _safe_mean(vals["normal_noext"]), 6
            ),
            "separation_ext_adv_minus_normal": round(
                _safe_mean(vals["adversarial_ext"]) - _safe_mean(vals["normal_ext"]), 6
            ),
        }
    return out


def run_retrieval_compare_eval(args, model, detector):
    if not args.local_query_path or not args.local_corpus_path:
        raise ValueError(
            "--local_query_path and --local_corpus_path are required when --eval_mode retrieval_compare"
        )

    queries = load_local_queries(args.local_query_path)
    corpus = load_local_corpus(args.local_corpus_path)
    retriever = LexicalRetriever(corpus)

    if args.max_samples > 0:
        queries = queries[: args.max_samples]

    cond_scores = {
        "normal_noext": [],
        "adversarial_noext": [],
        "normal_ext": [],
        "adversarial_ext": [],
    }
    cond_detects = {
        "normal_noext": [],
        "adversarial_noext": [],
        "normal_ext": [],
        "adversarial_ext": [],
    }

    logs = []
    for sample in tqdm(queries):
        normal_prompt = sample["text"]
        adversarial_prompt = _build_adversarial_prompt(
            normal_prompt=normal_prompt,
            adversarial_prompt=sample.get("adversarial_text", ""),
            suffix=args.default_adversarial_suffix,
        )

        normal_context, normal_docs = _build_external_context(
            retriever, normal_prompt, args.retrieval_top_k
        )
        adv_context, adv_docs = _build_external_context(
            retriever, adversarial_prompt, args.retrieval_top_k
        )

        normal_noext = _calc_focus_score(model, detector, normal_prompt)
        adversarial_noext = _calc_focus_score(model, detector, adversarial_prompt)
        normal_ext = _calc_focus_score(
            model, detector, normal_prompt, external_context=normal_context
        )
        adversarial_ext = _calc_focus_score(
            model, detector, adversarial_prompt, external_context=adv_context
        )

        sample_scores = {
            "normal_noext": normal_noext,
            "adversarial_noext": adversarial_noext,
            "normal_ext": normal_ext,
            "adversarial_ext": adversarial_ext,
        }

        for key, score in sample_scores.items():
            cond_scores[key].append(score)
            cond_detects[key].append(bool(score <= detector.threshold))

        logs.append(
            {
                "id": sample["id"],
                "label": sample["label"],
                "source_prompt_id": sample.get("source_prompt_id", ""),
                "variant_index": sample.get("variant_index", 0),
                "attack_type": sample.get("attack_type", ""),
                "motivation": sample.get("motivation", ""),
                "normal_prompt": normal_prompt,
                "adversarial_prompt": adversarial_prompt,
                "retrieval": {
                    "normal_docs": normal_docs,
                    "adversarial_docs": adv_docs,
                },
                "focus_scores": sample_scores,
            }
        )

    summary = {
        "model": args.model_name,
        "seed": args.seed,
        "retrieval_mode": args.retrieval_mode,
        "retrieval_top_k": args.retrieval_top_k,
        "num_samples": len(logs),
        "threshold": detector.threshold,
        "mean_focus_scores": {k: round(_safe_mean(v), 6) for k, v in cond_scores.items()},
        "detection_rates": {k: round(_safe_mean(v), 6) for k, v in cond_detects.items()},
        "condition_stats": {
            k: _compute_condition_stats(v) for k, v in cond_scores.items()
        },
    }
    summary["deltas"] = {
        "normal_ext_minus_normal_noext": round(
            summary["mean_focus_scores"]["normal_ext"]
            - summary["mean_focus_scores"]["normal_noext"],
            6,
        ),
        "adversarial_ext_minus_adversarial_noext": round(
            summary["mean_focus_scores"]["adversarial_ext"]
            - summary["mean_focus_scores"]["adversarial_noext"],
            6,
        ),
        "separation_noext_adv_minus_normal": round(
            summary["mean_focus_scores"]["adversarial_noext"]
            - summary["mean_focus_scores"]["normal_noext"],
            6,
        ),
        "separation_ext_adv_minus_normal": round(
            summary["mean_focus_scores"]["adversarial_ext"]
            - summary["mean_focus_scores"]["normal_ext"],
            6,
        ),
    }
    thresholds = _build_threshold_grid(
        min_v=args.threshold_sweep_min,
        max_v=args.threshold_sweep_max,
        step_v=args.threshold_sweep_step,
    )
    summary["threshold_sweep"] = {
        "config": {
            "min": round(float(args.threshold_sweep_min), 6),
            "max": round(float(args.threshold_sweep_max), 6),
            "step": round(float(args.threshold_sweep_step), 6),
            "num_thresholds": len(thresholds),
        },
        "noext": _compute_threshold_sweep(
            benign_scores=cond_scores["normal_noext"],
            adversarial_scores=cond_scores["adversarial_noext"],
            thresholds=thresholds,
        ),
        "ext": _compute_threshold_sweep(
            benign_scores=cond_scores["normal_ext"],
            adversarial_scores=cond_scores["adversarial_ext"],
            thresholds=thresholds,
        ),
    }
    summary["grouped_analysis"] = {
        "by_attack_type": _aggregate_group_stats(logs, "attack_type"),
        "by_source_prompt_id": _aggregate_group_stats(logs, "source_prompt_id"),
    }

    output_dir = os.path.join("result", args.output_subdir)
    os.makedirs(output_dir, exist_ok=True)
    output_logs = os.path.join(
        output_dir, f"{args.model_name}-{args.seed}-retrieval_compare.json"
    )
    output_summary = os.path.join(output_dir, "retrieval_compare_summary.jsonl")

    with open(output_logs, "w", encoding="utf-8") as f_out:
        f_out.write(json.dumps({"summary": summary, "result": logs}, indent=4))

    with open(output_summary, "a", encoding="utf-8") as f_out:
        f_out.write(json.dumps(summary) + "\n")

    print("===================")
    print("4-condition retrieval comparison summary")
    print(json.dumps(summary, indent=2))


def main(args):
    set_seed(args.seed)

    model_config_path = f"./configs/model_configs/{args.model_name}_config.json"
    model_config = open_config(config_path=model_config_path)
    retrieval_defaults = get_retrieval_config(model_config)
    if args.retrieval_mode is None:
        args.retrieval_mode = retrieval_defaults["retrieval_mode"]
    if args.retrieval_top_k is None:
        args.retrieval_top_k = int(retrieval_defaults["retrieval_top_k"])
    if args.default_adversarial_suffix is None:
        args.default_adversarial_suffix = retrieval_defaults["default_adversarial_suffix"]
    if args.output_subdir is None:
        args.output_subdir = retrieval_defaults["output_subdir"]

    model = create_model(config=model_config)
    model.print_model_info()

    detector = AttentionDetector(model)
    print("===================")
    print(f"Using detector: {detector.name}")

    if args.eval_mode == "benchmark":
        run_benchmark_eval(args, detector)
    elif args.eval_mode == "retrieval_compare":
        run_retrieval_compare_eval(args, model, detector)
    else:
        raise ValueError(f"Unsupported eval mode: {args.eval_mode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prompt Injection Detection Script")

    parser.add_argument(
        "--model_name", type=str, default="qwen2-attn",
        help="Path to the model configuration file."
    )
    parser.add_argument(
        "--dataset_name", type=str, default="deepset/prompt-injections",
        help="Path to the dataset."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--eval_mode",
        type=str,
        default="benchmark",
        choices=["benchmark", "retrieval_compare"],
        help="benchmark keeps original behavior; retrieval_compare runs 4-condition external-context evaluation."
    )
    parser.add_argument(
        "--local_query_path",
        type=str,
        default="",
        help="Path to local query JSON/JSONL with fields: id, text, label, optional adversarial_text.",
    )
    parser.add_argument(
        "--local_corpus_path",
        type=str,
        default="",
        help="Path to local corpus JSON/JSONL with fields: doc_id, text, optional title.",
    )
    parser.add_argument("--retrieval_mode", type=str, default=None, choices=["lexical"])
    parser.add_argument("--retrieval_top_k", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument(
        "--default_adversarial_suffix",
        type=str,
        default=None,
        help="Used when local query record does not include adversarial_text.",
    )
    parser.add_argument(
        "--output_subdir",
        type=str,
        default=None,
        help="Subdirectory under ./result/ for retrieval_compare outputs.",
    )
    parser.add_argument(
        "--threshold_sweep_min",
        type=float,
        default=0.0,
        help="Minimum threshold value for retrieval_compare threshold sweep.",
    )
    parser.add_argument(
        "--threshold_sweep_max",
        type=float,
        default=1.0,
        help="Maximum threshold value for retrieval_compare threshold sweep.",
    )
    parser.add_argument(
        "--threshold_sweep_step",
        type=float,
        default=0.05,
        help="Threshold step size for retrieval_compare threshold sweep.",
    )

    args = parser.parse_args()
    main(args)