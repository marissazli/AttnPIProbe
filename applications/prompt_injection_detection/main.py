import argparse
import sys
import os

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.utils import _save_results, setup_seeds
from src.load_dataset import _load_dataset
from src.evaluate import *
from src.utils import *
from src.prompts import wrap_prompt_guardrail
import gc
import torch
import PromptInjectionAttacks as PI
from DataSentinelDetector import DataSentinelDetector
from src.utils import _read_results
def validate_score_funcs(score_func):
    valid_choices = ['stc', 'loo', 'shapley', 'denoised_shapley', 'lime']  # Add all valid choices here
    if score_func not in valid_choices:
        raise argparse.ArgumentTypeError(f"Invalid choice: {score_func}. Valid choices are {valid_choices}.")
    return score_func
def open_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config
def parse_args():
    parser = argparse.ArgumentParser(prog='RAGdebugging', description="test")

    # Base settings
    parser.add_argument("--attr_type", type=str, default="attntrace", 
                        choices=['tracllm','attntrace'],
                        help="Type of attribution method to use.")
    parser.add_argument('--K', type=int, default=5, 
                    help="Report top-K most important texts that lead to the output.")
    parser.add_argument("--explanation_level", type=str, default="segment", 
                    choices=['sentence', 'paragraph', 'segment'],
                    help="Level of explanation granularity.")

    # General args
    parser.add_argument('--model_name', type=str, default='llama3.1-8b', choices=['llama3.1-8b','llama3.1-70b','chatglm4-9b','gpt4o', "gpt4o-mini", "gpt4.1-mini",'qwen2-7b','qwen1.5-7b','gemma3-1b','phi3-mini','qwen2.5-7b','deepseek-v3','deepseek-r1','gemini-2.0-flash', 'gemini-1.5-flash','claude-haiku-3.5','claude-haiku-3'],
                        help="Name of the model to be used.")
    parser.add_argument("--dataset_name", type=str, default='nq-poison',
                        choices=["narrativeqa", "musique", "qmsum"], 
                        help="Name of the dataset to be used.")
    # attention args
    parser.add_argument('--avg_k', type=int, default=1, 
                        help="Average number of important texts to be used.")
    parser.add_argument('--q', type=float, default=0.5, 
                        help="Number of important texts to be used.")
    parser.add_argument('--B', type=int, default=20, 
                        help="Number of important texts to be used.")
    # Perturbation-based/TracLLM args
    parser.add_argument("--score_funcs", type=validate_score_funcs, nargs='+', default=["stc",'loo','denoised_shapley'], 
                        help="Scoring functions to be used (for tracllm/perturb). Input more than one score_funcs to ensemble.")
    parser.add_argument('--sh_N', type=int, default=20, 
                        help="Number of permutations for shapley/denoised_shapley.")
    parser.add_argument('--beta', type=float, default=0.2, 
                    help="Top percentage marginal contribution score considered for denoised_shapley. Default is 20%.")
    parser.add_argument('--w', type=int, default=2, 
                help="Scaling factor to upweight LOO for ensembling")

    # self_citation args
    parser.add_argument("--self_citation_model", type=str, default="self", 
                        choices=['gpt4o', "self"],
                        help="Model to use for self-citation. 'self' means using the inference model.")
    # context_cite args
    parser.add_argument("--cc_N", type=int, default=64, 
                        help="Size of training data for context-cite")

    #PoisonedRAG
    parser.add_argument('--retrieval_k', type=int, default=50, 
                        help="Number of top contexts to retrieve.")
    parser.add_argument("--retriever", type=str, default='contriever', 
                        help="Retriever model to be used.") # BEIR
    
    # prompt injection attack to LongBench
    parser.add_argument('--prompt_injection_attack', default='default', type=str, 
                        help="Type of prompt injection attack to perform.")
    parser.add_argument('--inject_times', type=int, default=5, 
                        help="Number of times to inject the prompt.")
    parser.add_argument('--injected_data_config_path', default='./injected_task_configs/sst2_config.json', type=str, 
                        help="Path to the configuration file for injected data.")
    parser.add_argument('--max_length', default=32000, type=int, 
                        help="Control the maximum length of the context.")

    # needle-in-haystack
    parser.add_argument('--context_length', type=int, default=-1, 
                        help="Length of the context to be used.")
    # other settings
    parser.add_argument('--gpu_id', type=str, default='0', 
                        help="ID of the GPU to be used.")
    parser.add_argument('--seed', type=int, default=12, 
                        help="Random seed for reproducibility.")
    parser.add_argument('--data_num', type=int, default=100, 
                        help="Number of evaluation data points.")
    parser.add_argument("--results_path", type=str, default="main", 
                        help="Path to save the results.")
    parser.add_argument('--evaluate_saved', action='store_true', 
                        help="Evaluate the saved results.")
    parser.add_argument('--verbose', type=int, default=1, 
                        help="Enable verbose mode for detailed logging.")

    args = parser.parse_args()
    print(args)
    return args

def main(args):
    
    # Load dataset and random select
    print("Loading Dataset!")
    dataset = _load_dataset(args.dataset_name, args.retriever, args.retrieval_k, 
                           model_name=args.model_name, shot=1, seed=args.seed,num_poison = args.inject_times)
    # Load LLM and init Attribution

    attacker = PI.create_attacker(args.prompt_injection_attack)
    guardrail = DataSentinelDetector('./applications/prompt_injection_detection/DataSentinel_models/checkpoint-5000')

    pred_results_path = args.results_path
    
    attr_result = _read_results(args,  pred_results_path)
    len_injected_data = len(attr_result)
    true_positive_counter = 0
    for i, dict in enumerate(attr_result):
        
        print(f"------------------------------Question number: {i}------------------------------")
        important_ids = dict['important_ids']
        importance_scores = dict['scores']
        assert len(importance_scores) == len(important_ids)
        question = dict['question']
        malicious_texts = dict['gt_important_texts']

        all_texts = split_context(args.explanation_level,dict['contexts'])
        gt_ids,_ = get_gt_ids(all_texts,malicious_texts)

        if gt_ids == []:
            continue

        top_k=top_k_indexes(importance_scores, args.K)
        topk_ids = [important_ids[j] for j in top_k]
        
        #remove top-K texts to check ASR change
        new_texts = add_specific_indexes(all_texts, topk_ids)
        prompt = wrap_prompt_guardrail(question, new_texts)
        guardrail_answer = guardrail.detect(prompt)
        print("DataSentinel (with AttnTrace)'s output for malicious context: [", guardrail_answer, "]")
        if clean_str(guardrail_answer) == "1":
            true_positive_counter += 1
    

    args.prompt_injection_attack = 'none'
    attr_result = _read_results(args,  pred_results_path)
    len_clean_data = len(attr_result)
    false_positive_counter = 0
    for i, dict in enumerate(attr_result):
        print(f"------------------------------Question number: {i}------------------------------")
        important_ids = dict['important_ids']
        importance_scores = dict['scores']
        assert len(importance_scores) == len(important_ids)
        question = dict['question']
        malicious_texts = dict['gt_important_texts']

        all_texts = split_context(args.explanation_level,dict['contexts'])

        top_k=top_k_indexes(importance_scores, args.K)
        topk_ids = [important_ids[j] for j in top_k]

        #remove top-K texts to check ASR change
        new_texts = add_specific_indexes(all_texts, topk_ids)
        prompt = wrap_prompt_guardrail(question, new_texts)
        guardrail_answer = guardrail.detect(prompt)
        print("DataSentinel (with AttnTrace)'s output for clean context: [", guardrail_answer, "]")
        if clean_str(guardrail_answer) == "1":
            false_positive_counter += 1

    print("DataSentinel(with AttnTrace) true positive rate: ", true_positive_counter / len_injected_data)
    print("DataSentinel(with AttnTrace) false positive rate: ", false_positive_counter / len_clean_data)
    # Run the garbage collector
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == '__main__':
    
    args = parse_args()
    sleep_time = random.randint(0, 10)
    print("sleep for ", sleep_time)
    time.sleep(sleep_time)
    wait_for_available_gpu_memory(20, device=int(args.gpu_id), check_interval=300)
    setup_seeds(args.seed)
    torch.cuda.empty_cache()
    if args.evaluate_saved == False:
        main(args)