"""
phase2_representations_expanded.py

Phase 2 on the full expanded dataset (1,680 mal + 1,680 benign).

Key additions over the seed-only version:
  - Loads observations from variations/variants_master.json
  - Stratified k-fold cross-validation for probe accuracy (replaces in-distribution training acc)
  - Permutation test for statistical significance of linear separability
  - Sample flag for memory management (full 3,360 hidden states may not fit)
  - All seed plotting functions reused unchanged

Usage:
    python phase2_representations_expanded.py --model_name qwen2.5-7b --gpu_id 0
    python phase2_representations_expanded.py --model_name qwen2.5-7b --gpu_id 0 --sample 400
"""

import argparse
import json
import os
import random
import numpy as np
import torch
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM

ATTACK_TYPE_NAMES = {
    1: "Direct\nExecution",
    2: "Parameterized\nExecution",
    3: "Conditional\nExecution",
    4: "Functional\nExecution",
    5: "Transfer\nExecution",
}

ATTACK_COLORS = {0: "#AAAAAA", 1: "#E74C3C", 2: "#E67E22", 3: "#3498DB", 4: "#27AE60", 5: "#9B59B6"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_observations_from_master(master_path, sample=None, seed=0):
    """
    Load malicious and benign attack observations from variants_master.json.
    Returns list of (id, domain, attack_type, label, observation_text) tuples.
    Paired: for each malicious scenario we load the attack observation,
    for the corresponding benign we load the benign observation at the same index.
    """
    with open(master_path) as f:
        master = json.load(f)

    malicious = master["malicious"]
    benign = master["benign"]

    assert len(malicious) == len(benign), "Malicious/benign counts don't match"

    if sample is not None:
        rng = random.Random(seed)
        # Stratified sample: equal numbers per attack type
        by_type = {}
        for i, s in enumerate(malicious):
            at = s.get("attack_type") or s.get("attack", 0)
            by_type.setdefault(at, []).append(i)
        per_type = max(1, sample // len(by_type))
        indices = []
        for at, idxs in by_type.items():
            indices.extend(rng.sample(idxs, min(per_type, len(idxs))))
        indices = sorted(indices)
        malicious = [malicious[i] for i in indices]
        benign = [benign[i] for i in indices]

    def infer_attack_type(s):
        # Try explicit fields first
        at = s.get("attack_type") or s.get("attack")
        if at:
            return int(at)
        # Parse from seed_id or id: format XX-YY or XX-YY-VNN -> YY = attack type
        sid = s.get("seed_id") or s.get("id", "")
        parts = sid.split("-")
        if len(parts) >= 2:
            try:
                return int(parts[1])
            except ValueError:
                pass
        return 0

    observations = []
    for mal, ben in zip(malicious, benign):
        attack_type = infer_attack_type(mal)
        atk_idx = mal.get("attack_observation_idx", 0)

        if attack_type == 0:
            continue  # skip malformed entries with unknown attack type

        # Malicious: attack observation text
        mal_obs = mal["steps"][atk_idx]["observation"]
        observations.append((
            mal["id"], mal.get("domain", ""), attack_type, 1, mal_obs
        ))

        # Benign: same step index but from benign scenario
        ben_obs = ben["steps"][atk_idx]["observation"]
        observations.append((
            ben["id"], ben.get("domain", ""), attack_type, 0, ben_obs
        ))

    return observations


# ---------------------------------------------------------------------------
# Hidden state extraction
# ---------------------------------------------------------------------------

def extract_hidden_states(observations, model_name, device, layer_stride=4, hf_token=None):
    """
    For each observation text, run a forward pass and extract
    residual stream activations at the final token for selected layers.
    Returns all_hidden: (N, n_layers, hidden_dim) numpy array.
    """
    print("Loading model: {}".format(model_name))
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, token=hf_token, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
        device_map=device, token=hf_token, trust_remote_code=True)
    hf_model.eval()

    # Determine layer count and select layers
    n_layers = hf_model.config.num_hidden_layers
    layer_indices = list(range(0, n_layers + 1, layer_stride))
    if (n_layers) not in layer_indices:
        layer_indices.append(n_layers)
    print("Extracting from {} layers: {}".format(len(layer_indices), layer_indices))

    all_hidden = []
    for obs_tuple in tqdm(observations, desc="Extracting hidden states"):
        text = obs_tuple[4]
        inputs = tokenizer(
            text, return_tensors="pt",
            truncation=True, max_length=2048
        ).to(device)

        with torch.no_grad():
            outputs = hf_model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )

        # hidden_states: tuple of (n_layers+1) tensors, each (1, seq_len, hidden_dim)
        # Take last token position from each selected layer
        hidden = torch.stack([
            outputs.hidden_states[i][0, -1, :].float().cpu()
            for i in layer_indices
        ])  # (n_selected_layers, hidden_dim)
        all_hidden.append(hidden.numpy())

    del hf_model
    torch.cuda.empty_cache()
    return np.array(all_hidden), layer_indices  # (N, n_layers, hidden_dim)


# ---------------------------------------------------------------------------
# Probing utilities — with CV and permutation test
# ---------------------------------------------------------------------------

def pca_project(X, n_components=2):
    X_centered = X - X.mean(axis=0)
    pca = PCA(n_components=n_components)
    proj = pca.fit_transform(X_centered)
    return proj, pca


def linear_separability_cv(X, y, n_splits=5, seed=0):
    """
    Stratified k-fold cross-validation accuracy.
    Returns mean and std across folds.
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, C=1.0))
    ])
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
    return float(np.mean(scores)), float(np.std(scores))


def linear_separability_train(X, y):
    """In-distribution training accuracy (for comparison with seed results)."""
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_s, y)
    return accuracy_score(y, clf.predict(X_s))


def permutation_test(X, y, n_permutations=200, seed=0):
    """
    Permutation test for linear separability.
    Returns (observed_acc, p_value, permutation_accs).
    """
    rng = np.random.RandomState(seed)
    observed_acc, _ = linear_separability_cv(X, y)
    perm_accs = []
    for _ in range(n_permutations):
        y_perm = rng.permutation(y)
        acc, _ = linear_separability_cv(X, y_perm)
        perm_accs.append(acc)
    p_value = float(np.mean(np.array(perm_accs) >= observed_acc))
    return observed_acc, p_value, perm_accs


def probe_transfer_accuracy(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_train_s, y_train)
    return accuracy_score(y_test, clf.predict(X_test_s))


# ---------------------------------------------------------------------------
# Plotting — reused from seed script
# ---------------------------------------------------------------------------

def plot_layer_emergence_cv(all_hidden, observations, layer_indices, output_path, model_name):
    """
    Layer emergence plot using cross-validated accuracy (not training accuracy).
    """
    labels = np.array([o[3] for o in observations])
    attack_types = [o[2] for o in observations]
    unique_types = sorted(set(attack_types))

    accs_all_mean, accs_all_std = [], []
    accs_by_type = {at: {"mean": [], "std": []} for at in unique_types}

    print("  Computing CV probe accuracy per layer...")
    for li in tqdm(range(len(layer_indices)), desc="Layers"):
        X = all_hidden[:, li, :]
        m, s = linear_separability_cv(X, labels)
        accs_all_mean.append(m)
        accs_all_std.append(s)
        for at in unique_types:
            mask = [i for i, t in enumerate(attack_types) if t == at]
            X_t = X[mask]
            y_t = labels[mask]
            if len(set(y_t)) < 2:
                accs_by_type[at]["mean"].append(0.5)
                accs_by_type[at]["std"].append(0.0)
            else:
                m_t, s_t = linear_separability_cv(X_t, y_t)
                accs_by_type[at]["mean"].append(m_t)
                accs_by_type[at]["std"].append(s_t)

    accs_all_mean = np.array(accs_all_mean)
    accs_all_std = np.array(accs_all_std)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(layer_indices, accs_all_mean, "k-o", linewidth=2,
            markersize=5, label="All types", zorder=5)
    ax.fill_between(layer_indices,
                    accs_all_mean - accs_all_std,
                    accs_all_mean + accs_all_std,
                    alpha=0.15, color="black")

    for at in unique_types:
        m_arr = np.array(accs_by_type[at]["mean"])
        s_arr = np.array(accs_by_type[at]["std"])
        ax.plot(layer_indices, m_arr,
                color=ATTACK_COLORS[at], linestyle="--", linewidth=1.2,
                marker=".", markersize=4,
                label=ATTACK_TYPE_NAMES[at].replace("\n", " "))
        ax.fill_between(layer_indices, m_arr - s_arr, m_arr + s_arr,
                        alpha=0.08, color=ATTACK_COLORS[at])

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.6, label="Chance")
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("CV probe accuracy (5-fold)", fontsize=11)
    ax.set_title(
        "Emergence of Linear Separability (Cross-Validated) — {}\nn={}".format(
            model_name, len(observations)),
        fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=8, ncol=3, loc="lower right")
    ax.set_ylim(0.3, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: {}".format(output_path))
    return accs_all_mean, accs_all_std


def plot_pca_best_layer(all_hidden, observations, layer_indices, output_path, model_name):
    """
    PCA scatter at best CV layer, colored by label, faceted by attack type.
    """
    labels = np.array([o[3] for o in observations])
    attack_types = [o[2] for o in observations]
    unique_types = sorted(set(attack_types))

    # Find best layer by CV accuracy
    best_li, best_acc = 0, 0.0
    for li in range(len(layer_indices)):
        X = all_hidden[:, li, :]
        acc, _ = linear_separability_cv(X, labels)
        if acc > best_acc:
            best_acc = acc
            best_li = li
    best_layer = layer_indices[best_li]
    X_best = all_hidden[:, best_li, :]

    n_types = len(unique_types)
    fig, axes = plt.subplots(1, n_types, figsize=(4 * n_types, 4.5))
    fig.suptitle(
        "PCA of Hidden States — Malicious (red) vs. Benign (blue)\n"
        "{} | Layer {} | CV probe accuracy: {:.1%} | n={}".format(
            model_name, best_layer, best_acc, len(observations)),
        fontsize=11, fontweight="bold", y=1.02
    )

    for ax, at in zip(axes if n_types > 1 else [axes], unique_types):
        mask = [i for i, t in enumerate(attack_types) if t == at]
        X_type = X_best[mask]
        y_type = labels[mask]
        proj, _ = pca_project(X_type)
        cv_acc, _ = linear_separability_cv(X_type, y_type) if len(set(y_type)) > 1 else (0.5, 0.0)

        for yi, marker, color, face in [
            (1, "o", ATTACK_COLORS[at], ATTACK_COLORS[at]),
            (0, "^", ATTACK_COLORS[at], "none")
        ]:
            idx = [j for j, y in enumerate(y_type) if y == yi]
            if idx:
                ax.scatter(proj[idx, 0], proj[idx, 1],
                           c=face, edgecolors=color,
                           marker=marker, s=60, linewidths=1.5, alpha=0.8,
                           label="Malicious" if yi == 1 else "Benign")
        ax.set_title("{}\n(CV acc: {:.0%}, n={})".format(
            ATTACK_TYPE_NAMES[at].replace("\n", " "), cv_acc, len(mask)),
            fontsize=9)
        ax.set_xlabel("PC1", fontsize=8)
        ax.set_ylabel("PC2", fontsize=8)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: {}".format(output_path))
    return best_layer, best_acc


def plot_probe_transfer(all_hidden, observations, layer_indices, output_path, model_name):
    """
    Cross-attack-type probe transfer heatmap.
    """
    labels = np.array([o[3] for o in observations])
    attack_types = [o[2] for o in observations]
    unique_types = sorted(set(attack_types))

    # Best layer by CV
    best_li, best_acc = 0, 0.0
    for li in range(len(layer_indices)):
        X = all_hidden[:, li, :]
        acc, _ = linear_separability_cv(X, labels)
        if acc > best_acc:
            best_acc = acc
            best_li = li
    X_best = all_hidden[:, best_li, :]

    n = len(unique_types)
    transfer_matrix = np.zeros((n, n))
    for i, at_train in enumerate(unique_types):
        mask_train = [j for j, t in enumerate(attack_types) if t == at_train]
        X_train = X_best[mask_train]
        y_train = labels[mask_train]
        for k, at_test in enumerate(unique_types):
            mask_test = [j for j, t in enumerate(attack_types) if t == at_test]
            X_test = X_best[mask_test]
            y_test = labels[mask_test]
            transfer_matrix[i, k] = probe_transfer_accuracy(X_train, y_train, X_test, y_test)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(transfer_matrix, vmin=0.4, vmax=1.0, cmap="RdYlGn", aspect="auto")
    plt.colorbar(im, ax=ax, label="Probe accuracy")
    type_labels = ["T{}".format(at) for at in unique_types]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(type_labels)
    ax.set_yticklabels(type_labels)
    ax.set_xlabel("Test attack type")
    ax.set_ylabel("Train attack type")
    ax.set_title("Probe Transfer Accuracy — {}\n(train on row → test on column)".format(model_name))
    for i in range(n):
        for k in range(n):
            ax.text(k, i, "{:.0%}".format(transfer_matrix[i, k]),
                    ha="center", va="center", fontsize=9, fontweight="bold",
                    color="white" if transfer_matrix[i, k] < 0.7 else "black")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: {}".format(output_path))
    return transfer_matrix, unique_types


def plot_permutation_test(perm_accs, observed_acc, p_value, output_path, model_name):
    """
    Histogram of permutation null distribution vs observed accuracy.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(perm_accs, bins=30, color="#95A5A6", edgecolor="white",
            linewidth=0.5, label="Null distribution ({} permutations)".format(len(perm_accs)))
    ax.axvline(observed_acc, color="#E74C3C", linewidth=2,
               label="Observed CV accuracy: {:.1%}".format(observed_acc))
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="Chance")
    ax.set_xlabel("CV probe accuracy", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(
        "Permutation Test — Linear Separability\n"
        "{} | p = {:.4f}".format(model_name, p_value),
        fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: {}".format(output_path))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda:{}".format(args.gpu_id)

    # Load observations
    print("Loading observations from {}...".format(args.master_path))
    observations = load_observations_from_master(
        args.master_path, sample=args.sample, seed=args.seed)
    print("Loaded {} observations ({} mal + {} ben)".format(
        len(observations),
        sum(1 for o in observations if o[3] == 1),
        sum(1 for o in observations if o[3] == 0)))

    # Cache path for hidden states
    n_label = len(observations)
    cache_path = os.path.join(
        args.output_dir,
        "{}-expanded-n{}-hidden.npz".format(args.model_name, n_label))

    # Resolve model path
    import json as _json2
    _config_path = "model_configs/{}_config.json".format(args.model_name)
    if os.path.exists(_config_path):
        with open(_config_path) as _f2:
            _cfg2 = _json2.load(_f2)
        model_name = _cfg2["model_info"]["name"]
        _api_pos2 = int(_cfg2["api_key_info"]["api_key_use"])
        _hf_token2 = _cfg2["api_key_info"]["api_keys"][_api_pos2]
        if _hf_token2 == "YOUR_API_KEY": _hf_token2 = None
    else:
        model_name = args.model_name
        _hf_token2 = None

    if os.path.exists(cache_path) and not args.recompute:
        print("Loading cached hidden states from {}...".format(cache_path))
        cached = np.load(cache_path, allow_pickle=True)
        all_hidden = cached["all_hidden"]
        layer_indices = list(cached["layer_indices"])
        print("Loaded: {} shape".format(all_hidden.shape))
    else:
        all_hidden, layer_indices = extract_hidden_states(
            observations, model_name, device, args.layer_stride, hf_token=_hf_token2)
        np.savez(cache_path, all_hidden=all_hidden, layer_indices=layer_indices)
        print("Hidden states cached to: {}".format(cache_path))

    labels = np.array([o[3] for o in observations])
    attack_types = [o[2] for o in observations]
    unique_types = sorted(set(attack_types))

    # ── Layer emergence with CV ──────────────────────────────────────────────
    print("\nGenerating layer emergence plot (CV)...")
    accs_mean, accs_std = plot_layer_emergence_cv(
        all_hidden, observations, layer_indices,
        os.path.join(args.output_dir, "fig_layer_emergence_cv_{}.pdf".format(args.model_name)),
        args.model_name)

    # ── PCA at best layer ────────────────────────────────────────────────────
    print("\nGenerating PCA per attack type...")
    best_layer, best_cv_acc = plot_pca_best_layer(
        all_hidden, observations, layer_indices,
        os.path.join(args.output_dir, "fig_pca_per_type_expanded_{}.pdf".format(args.model_name)),
        args.model_name)
    print("  Best layer: {}, CV accuracy: {:.1%}".format(best_layer, best_cv_acc))

    # ── Probe transfer matrix ────────────────────────────────────────────────
    print("\nGenerating probe transfer matrix...")
    transfer_matrix, unique_types = plot_probe_transfer(
        all_hidden, observations, layer_indices,
        os.path.join(args.output_dir, "fig_probe_transfer_expanded_{}.pdf".format(args.model_name)),
        args.model_name)

    # ── Permutation test at best layer ───────────────────────────────────────
    print("\nRunning permutation test ({} permutations)...".format(args.n_permutations))
    # Find best layer index
    best_li = layer_indices.index(best_layer) if best_layer in layer_indices else 0
    X_best = all_hidden[:, best_li, :]
    observed_acc, p_value, perm_accs = permutation_test(
        X_best, labels, n_permutations=args.n_permutations, seed=args.seed)
    print("  Observed CV acc: {:.1%}, p = {:.4f}".format(observed_acc, p_value))

    plot_permutation_test(
        perm_accs, observed_acc, p_value,
        os.path.join(args.output_dir, "fig_permutation_test_{}.pdf".format(args.model_name)),
        args.model_name)

    # ── Per-layer CV accuracy summary ────────────────────────────────────────
    layer_accs = {
        str(layer_indices[i]): {
            "cv_mean": round(float(accs_mean[i]), 4),
            "cv_std": round(float(accs_std[i]), 4),
        }
        for i in range(len(layer_indices))
    }

    # ── Transfer matrix summary ──────────────────────────────────────────────
    transfer_summary = {
        "T{}_to_T{}".format(unique_types[i], unique_types[k]): round(float(transfer_matrix[i, k]), 4)
        for i in range(len(unique_types))
        for k in range(len(unique_types))
    }
    ood_accs = [
        transfer_matrix[i, k]
        for i in range(len(unique_types))
        for k in range(len(unique_types))
        if i != k
    ]

    # ── Save summary ─────────────────────────────────────────────────────────
    summary = {
        "model": args.model_name,
        "n_observations": len(observations),
        "n_malicious": int(sum(labels)),
        "n_benign": int(len(labels) - sum(labels)),
        "n_layers_extracted": len(layer_indices),
        "layer_stride": args.layer_stride,
        "best_layer": best_layer,
        "best_cv_accuracy": round(float(best_cv_acc), 4),
        "permutation_test": {
            "observed_cv_acc": round(float(observed_acc), 4),
            "p_value": round(float(p_value), 4),
            "n_permutations": args.n_permutations,
            "significant": p_value < 0.05,
        },
        "linear_separability_by_layer": layer_accs,
        "probe_transfer_matrix": transfer_summary,
        "ood_transfer_mean": round(float(np.mean(ood_accs)), 4),
        "ood_transfer_std": round(float(np.std(ood_accs)), 4),
        "ood_transfer_min": round(float(np.min(ood_accs)), 4),
        "ood_transfer_max": round(float(np.max(ood_accs)), 4),
    }

    summary_path = os.path.join(
        args.output_dir, "summary_expanded_{}.json".format(args.model_name))
    # Convert numpy types to native Python for JSON serialisation
    def to_python(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    def clean(d):
        if isinstance(d, dict): return {k: clean(v) for k, v in d.items()}
        if isinstance(d, list): return [clean(v) for v in d]
        return to_python(d)

    with open(summary_path, "w") as f:
        json.dump(clean(summary), f, indent=2)

    print("\n" + "=" * 60)
    print("Phase 2 Expanded — Summary")
    print("=" * 60)
    print("n = {} ({} mal + {} ben)".format(
        len(observations), int(sum(labels)), int(len(labels) - sum(labels))))
    print("Best layer: {} | CV accuracy: {:.1%}".format(best_layer, best_cv_acc))
    print("Permutation test: p = {:.4f} ({})".format(
        p_value, "SIGNIFICANT" if p_value < 0.05 else "not significant"))
    print("OOD transfer: mean={:.1%} range=[{:.1%}, {:.1%}]".format(
        np.mean(ood_accs), np.min(ood_accs), np.max(ood_accs)))
    print("\nAll outputs in: {}/".format(args.output_dir))
    print("Summary saved: {}".format(summary_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 2 representation analysis on expanded dataset")
    parser.add_argument("--model_name", type=str, default="qwen2.5-7b")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layer_stride", type=int, default=4)
    parser.add_argument("--master_path", type=str,
                        default="./variations/variants_master.json")
    parser.add_argument("--output_dir", type=str, default="./phase2_results_expanded")
    parser.add_argument("--sample", type=int, default=None,
                        help="Stratified sample of N pairs per run (default: all 1680)")
    parser.add_argument("--n_permutations", type=int, default=500,
                        help="Number of permutations for significance test (default: 500)")
    parser.add_argument("--recompute", action="store_true",
                        help="Recompute hidden states even if cached")
    args = parser.parse_args()
    main(args)