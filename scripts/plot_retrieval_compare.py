import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CONDITIONS = ["normal_noext", "adversarial_noext", "normal_ext", "adversarial_ext"]


def _load_result(path):
    with open(path, "r", encoding="utf-8") as f_in:
        data = json.load(f_in)
    if "result" not in data:
        raise ValueError("Expected retrieval compare JSON with top-level 'result' key.")
    return data


def _safe_label(log):
    source_id = log.get("source_prompt_id") or log.get("id", "")
    attack_type = log.get("attack_type", "")
    if attack_type:
        return f"{source_id}:{attack_type}"
    return str(source_id)


def _plot_paired_bar(logs, output_dir, max_items):
    logs = logs[:max_items] if max_items > 0 else logs
    labels = [_safe_label(log) for log in logs]
    benign_vals = [log["focus_scores"]["normal_noext"] for log in logs]
    adv_vals = [log["focus_scores"]["adversarial_noext"] for log in logs]

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 0.45), 6))
    ax.bar(x - width / 2, benign_vals, width, label="benign (noext)")
    ax.bar(x + width / 2, adv_vals, width, label="adversarial (noext)")
    ax.set_title("Per-sample benign vs adversarial focus score")
    ax.set_xlabel("sample")
    ax.set_ylabel("focus score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=8)
    ax.legend()
    fig.tight_layout()
    out = output_dir / "paired_bar_benign_vs_adversarial_noext.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)


def _plot_distribution(logs, output_dir):
    grouped = {cond: [] for cond in CONDITIONS}
    for log in logs:
        scores = log["focus_scores"]
        for cond in CONDITIONS:
            grouped[cond].append(scores[cond])

    data = [grouped[c] for c in CONDITIONS]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(data, labels=CONDITIONS, showfliers=False)
    ax.set_title("Focus-score distribution by condition")
    ax.set_ylabel("focus score")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    out = output_dir / "distribution_by_condition_boxplot.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)


def _plot_attack_heatmap(logs, output_dir):
    attack_types = sorted({log.get("attack_type", "") for log in logs if log.get("attack_type", "")})
    if not attack_types:
        return

    matrix = np.zeros((len(attack_types), len(CONDITIONS)), dtype=float)
    for i, attack_type in enumerate(attack_types):
        subset = [log for log in logs if log.get("attack_type", "") == attack_type]
        if not subset:
            continue
        for j, cond in enumerate(CONDITIONS):
            vals = [row["focus_scores"][cond] for row in subset]
            matrix[i, j] = float(np.mean(vals)) if vals else 0.0

    fig, ax = plt.subplots(figsize=(10, max(4, len(attack_types) * 0.7)))
    im = ax.imshow(matrix, aspect="auto")
    ax.set_title("Mean focus score by attack type and condition")
    ax.set_xticks(np.arange(len(CONDITIONS)))
    ax.set_xticklabels(CONDITIONS, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(attack_types)))
    ax.set_yticklabels(attack_types)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("mean focus score")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=8)

    fig.tight_layout()
    out = output_dir / "heatmap_attack_type_by_condition.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate PNG plots from retrieval compare output.")
    parser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help="Path to run_dataset retrieval compare JSON (contains summary/result).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Directory to save plots. Defaults to <input_dir>/plots",
    )
    parser.add_argument(
        "--max_bar_items",
        type=int,
        default=60,
        help="Maximum number of samples to show in paired bar chart (0 for all).",
    )
    args = parser.parse_args()

    data = _load_result(args.input_json)
    logs = data["result"]

    in_path = Path(args.input_json)
    output_dir = Path(args.output_dir) if args.output_dir else in_path.parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    _plot_paired_bar(logs, output_dir, args.max_bar_items)
    _plot_distribution(logs, output_dir)
    _plot_attack_heatmap(logs, output_dir)
    print(f"Saved plots to: {output_dir}")


if __name__ == "__main__":
    main()
