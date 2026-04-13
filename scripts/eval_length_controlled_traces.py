import argparse
import json
import math
import os
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from utils import open_config, create_model
from detector.attn import AttentionDetector


FormatName = Literal["payload_only", "full_observation", "full_context"]


def build_payload_only(scenario: Dict[str, Any]) -> str:
    return scenario.get("attack_payload") or scenario.get("benign_instruction", "")


def build_full_observation(scenario: Dict[str, Any]) -> str:
    idx = scenario["attack_observation_idx"]
    return scenario["steps"][idx]["observation"]


def build_full_context(scenario: Dict[str, Any]) -> str:
    parts = [f"USER TASK: {scenario['user_prompt']}\n"]
    for step in scenario["steps"]:
        parts.append(f"[Step {step['step']} | Tool: {step['tool']}]")
        parts.append(f"Observation: {step['observation']}")
    return "\n".join(parts)


FORMAT_BUILDERS = {
    "payload_only": build_payload_only,
    "full_observation": build_full_observation,
    "full_context": build_full_context,
}


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def compute_basic_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    return {
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
        "tpr": float(tpr),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "accuracy": float((tp + tn) / max(1, (tp + tn + fp + fn))),
    }


def pick_threshold_for_target_fpr(
    y_true: np.ndarray,
    scores: np.ndarray,
    target_fpr: float,
) -> float:
    """
    Choose a threshold on the repo's 'focus_score' such that detected = (focus_score <= threshold)
    yields FPR <= target_fpr, using the ROC curve.

    We define a "positive" as label=1 (malicious). Lower focus_score indicates more likely positive.
    For ROC, we use decision_score = -focus_score (higher -> more positive).
    """
    decision = -scores
    fpr, tpr, thresh = roc_curve(y_true, decision, pos_label=1)
    # `thresh` is on decision; convert to focus_score threshold: decision >= t  <=> -focus_score >= t <=> focus_score <= -t
    feasible = np.where(fpr <= target_fpr)[0]
    if feasible.size == 0:
        # No threshold meets target; return strictest (smallest focus_score threshold).
        return float(-thresh.max())
    # pick the one with highest TPR among feasible
    best_idx = feasible[np.argmax(tpr[feasible])]
    return float(-thresh[best_idx])


def evaluate_format(
    detector: AttentionDetector,
    rows: List[Dict[str, Any]],
    fmt: FormatName,
    *,
    override_threshold: Optional[float],
) -> Dict[str, Any]:
    y_true: List[int] = []
    focus_scores: List[float] = []
    per_example: List[Dict[str, Any]] = []

    if override_threshold is not None:
        detector.threshold = float(override_threshold)

    for r in rows:
        text = FORMAT_BUILDERS[fmt](r)
        detected, meta = detector.detect(text)
        fs = float(meta.get("focus_score", math.nan))
        label = int(r["label"])
        y_true.append(label)
        focus_scores.append(fs)
        per_example.append(
            {
                "id": r.get("id"),
                "variant": r.get("variant"),
                "label": label,
                "format": fmt,
                "detected": bool(detected),
                "focus_score": fs,
            }
        )

    y_pred = [1 if ex["detected"] else 0 for ex in per_example]  # detected => malicious

    y_true_np = np.array(y_true, dtype=int)
    scores_np = np.array(focus_scores, dtype=float)
    decision_np = -scores_np

    out: Dict[str, Any] = {
        "format": fmt,
        "n": len(rows),
        "threshold": float(detector.threshold),
        "basic": compute_basic_metrics(y_true, y_pred),
        "score_summary": {
            "focus_score_mean": float(np.nanmean(scores_np)),
            "focus_score_std": float(np.nanstd(scores_np)),
            "focus_score_min": float(np.nanmin(scores_np)),
            "focus_score_max": float(np.nanmax(scores_np)),
        },
        "per_example": per_example,
    }

    # AUC/AUPRC require both classes present
    if len(set(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true_np, decision_np))
        out["auprc"] = float(average_precision_score(y_true_np, decision_np))
    else:
        out["roc_auc"] = None
        out["auprc"] = None

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AttentionDetector on length-controlled trace datasets")
    parser.add_argument("--model_name", type=str, default="qwen2-attn",
                        help="Model config name (configs/model_configs/<name>_config.json)")
    parser.add_argument("--in_path", type=str, default="./datasets/agentic_pi_length_controlled/traces.jsonl")
    parser.add_argument("--out_path", type=str, default="./result/agentic_pi_length_controlled/eval.json")
    parser.add_argument("--formats", type=str, default="payload_only,full_observation,full_context")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override detector threshold. If unset, uses detector default (0.5).")
    parser.add_argument("--calibrate_target_fpr", type=float, default=None,
                        help="If set, calibrate a threshold to achieve FPR<=target on the chosen format.")
    parser.add_argument("--calibrate_format", type=str, default="full_context",
                        choices=["payload_only", "full_observation", "full_context"],
                        help="Which format to use for threshold calibration if calibrate_target_fpr is set.")
    args = parser.parse_args()

    rows = read_jsonl(args.in_path)

    model_config_path = f"./configs/model_configs/{args.model_name}_config.json"
    model_config = open_config(model_config_path)
    model = create_model(config=model_config)
    model.print_model_info()
    detector = AttentionDetector(model)

    formats: List[FormatName] = [f.strip() for f in args.formats.split(",") if f.strip()]  # type: ignore[assignment]

    calibrated_threshold: Optional[float] = None
    calibration_report: Optional[Dict[str, Any]] = None

    if args.calibrate_target_fpr is not None:
        # Calibrate using focus_scores on the selected format with the *detector's* scoring.
        fmt: FormatName = args.calibrate_format  # type: ignore[assignment]
        y_true: List[int] = []
        focus_scores: List[float] = []
        for r in rows:
            text = FORMAT_BUILDERS[fmt](r)
            _, meta = detector.detect(text)
            focus_scores.append(float(meta.get("focus_score", math.nan)))
            y_true.append(int(r["label"]))

        y_true_np = np.array(y_true, dtype=int)
        scores_np = np.array(focus_scores, dtype=float)
        calibrated_threshold = pick_threshold_for_target_fpr(
            y_true_np, scores_np, float(args.calibrate_target_fpr)
        )
        calibration_report = {
            "calibrate_format": fmt,
            "target_fpr": float(args.calibrate_target_fpr),
            "calibrated_threshold": float(calibrated_threshold),
        }

    threshold_override = args.threshold if args.threshold is not None else calibrated_threshold

    eval_out: Dict[str, Any] = {
        "input": {
            "model_name": args.model_name,
            "in_path": args.in_path,
            "formats": formats,
            "threshold_override": threshold_override,
        },
        "calibration": calibration_report,
        "by_format": {},
    }

    for fmt in formats:
        res = evaluate_format(detector, rows, fmt, override_threshold=threshold_override)
        eval_out["by_format"][fmt] = res

    write_json(args.out_path, eval_out)
    print(f"Wrote evaluation to {args.out_path}")


if __name__ == "__main__":
    main()

