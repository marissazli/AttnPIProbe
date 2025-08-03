
import sys
import os



attr_type = 'attntrace' 

dataset_name= 'musique' #choose from 'musique','narrativeqa', and 'qmsum'
prompt_injection_attack = 'default' 
inject_times = 3 # number of injected instructions
model_name = "llama3.1-8b"
data_num = 100 # number of evaluation data points
explanation_level = "segment" # 'sentence','paragraph' or 'segment'
K = 3 # find top-K most important text segments
avg_k = 5
q = 0.4
B = 30
gpu_id = 0# GPU ID

if not os.path.exists(f'./log/{model_name}/{dataset_name}/{attr_type}'):
    os.makedirs(f'./log/{model_name}/{dataset_name}/{attr_type}')

cmd = f'nohup python -u applications/prompt_injection_detection/main.py \
--dataset_name {dataset_name} \
--attr_type {attr_type} \
--K {K} \
--avg_k {avg_k} \
--q {q} \
--B {B} \
--explanation_level {explanation_level} \
--prompt_injection_attack {prompt_injection_attack} \
--retrieval_k 50 \
--inject_times {inject_times} \
--model_name {model_name} \
--gpu_id {gpu_id} \
--verbose 0 \
--data_num {data_num} \
> ./log/{model_name}/{dataset_name}/{attr_type}/data_sentinel_{dataset_name}_{model_name}_{prompt_injection_attack}_{inject_times}_{attr_type}_{avg_k}_{q}_{B}_{K}.out \
2>&1 &'
os.system(cmd)
