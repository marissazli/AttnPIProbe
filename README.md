# Attention Tracker: Detecting Prompt Injection Attacks in LLMs

Welcome to the official repository for **"Attention Tracker: Detecting Prompt Injection Attacks in LLMs"**. This repository provides scripts and tools to identify important attention heads and evaluate prompt injection attacks on large language models (LLMs).

Project page: https://huggingface.co/spaces/TrustSafeAI/Attention-Tracker 

Paper: https://arxiv.org/abs/2411.00348 

---

## Features
- **Identify Important Heads**: Determine which attention heads are critical for detecting prompt injection attacks.
- **Run Experiments**: Execute experiments on datasets to evaluate the model's effectiveness.
- **Test Queries**: Test individual queries against your chosen model.

---

## Getting Started

### Prerequisites
1. Ensure Python 3.10+ is installed.
2. Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### One Command Path for Comprehensive Results

Run these 3 commands in order (dataset build -> evaluation -> plots):

```bash
# 1) Build motivated adversarial query set (10 benign prompts x 5 variants = 50 samples)
python scripts/build_eval_dataset.py \
  --dataset_name deepset/prompt-injections \
  --split test \
  --num_benign 10 \
  --variants_per_prompt 5 \
  --seed 7 \
  --output_path ./data/generated/retrieval_queries_motivated.jsonl

# 2) Run full 4-condition retrieval comparison + threshold sweep diagnostics
python run_retrieval_experiment.py \
  --model_name qwen2-attn \
  --local_query_path ./data/generated/retrieval_queries_motivated.jsonl \
  --local_corpus_path ./data/samples/retrieval_corpus.jsonl \
  --max_samples 0 \
  --retrieval_top_k 3 \
  --threshold_sweep_min 0.0 \
  --threshold_sweep_max 1.0 \
  --threshold_sweep_step 0.05

# 3) Generate static visual summaries
python scripts/plot_retrieval_compare.py \
  --input_json ./result/retrieval_compare/qwen2-attn-0-retrieval_compare.json \
  --max_bar_items 60
```

### Key Output Files

- `./result/retrieval_compare/qwen2-attn-0-retrieval_compare.json`
  - Per-sample logs for all 4 conditions:
    - `normal_noext`
    - `adversarial_noext`
    - `normal_ext`
    - `adversarial_ext`
  - Summary metrics:
    - `mean_focus_scores`
    - `detection_rates`
    - `deltas`
    - `condition_stats` (`mean`, `std`, `min`, `max`)
    - `threshold_sweep` (`tpr`, `fpr`, `youden_j`, `balanced_accuracy`, `best_by_youden`)
    - `grouped_analysis` (`by_attack_type`, `by_source_prompt_id`)

- `./result/retrieval_compare/retrieval_compare_summary.jsonl`
  - One-line summary record per run.

- `./result/retrieval_compare/plots/`
  - `paired_bar_benign_vs_adversarial_noext.png`
  - `distribution_by_condition_boxplot.png`
  - `heatmap_attack_type_by_condition.png`

### License
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/deed.en)

### Citation
```
@misc{hung2024attentiontrackerdetectingprompt,
      title={Attention Tracker: Detecting Prompt Injection Attacks in LLMs}, 
      author={Kuo-Han Hung and Ching-Yun Ko and Ambrish Rawat and I-Hsin Chung and Winston H. Hsu and Pin-Yu Chen},
      year={2024},
      eprint={2411.00348},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2411.00348}, 
}
```
