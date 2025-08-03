
import sys
import os


attr_type = 'attntrace' # 'attntrace', 'avg_attention', 'tracllm', 'vanilla_perturb' or 'self_citation'

dataset_name= 'nq-poison' #choose from 'nq-poison','hotpotqa-poison','msmarco-poison'
#"gpt4o-mini", "gpt4.1-mini",'qwen2-7b','qwen1.5-7b','gemma3-1b','phi3-mini','qwen2.5-7b','deepseek-v3','deepseek-r1','gemini-2.0-flash', 'gemini-1.5-flash','claude-haiku-3.5','claude-haiku-3'
model_name = "llama3.1-8b"
data_num = 100 # number of evaluation data points
explanation_level = "segment" # 'sentence','paragraph' or 'segment'
K = 5 # find top-K most important text segments
avg_k = 5
q = 0.4
B = 30
gpu_id = 0# GPU ID
if not os.path.exists(f'./log/{model_name}/{dataset_name}/{attr_type}'):
    os.makedirs(f'./log/{model_name}/{dataset_name}/{attr_type}')

cmd = f'nohup python -u main.py \
--dataset_name {dataset_name} \
--attr_type {attr_type} \
--K {K} \
--avg_k {avg_k} \
--q {q} \
--B {B} \
--explanation_level {explanation_level} \
--retrieval_k 50 \
--model_name {model_name} \
--gpu_id {gpu_id} \
--verbose 0 \
--data_num {data_num} \
> ./log/{model_name}/{dataset_name}/{attr_type}/{dataset_name}_{model_name}_{attr_type}_{avg_k}_{q}_{B}_{K}.out \
2>&1 &'
os.system(cmd)
