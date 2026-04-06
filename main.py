import argparse
from src.models import create_model
from src.utils import _save_results, setup_seeds
from src.attribution import create_attr
from src.load_dataset import _load_dataset
from src.evaluate import *
from src.utils import *
from src.prompts import wrap_prompt
import gc
import torch
import PromptInjectionAttacks as PI
from method_configs.read_configs import load_method_configs

def validate_score_funcs(score_func):
    valid_choices = ['stc', 'loo', 'shapley', 'denoised_shapley', 'lime']  # Add all valid choices here
    if score_func not in valid_choices:
        raise argparse.ArgumentTypeError(f"Invalid choice: {score_func}. Valid choices are {valid_choices}.")
    return score_func

def parse_args():
    parser = argparse.ArgumentParser(prog='RAGdebugging', description="test")

    # Base settings
    parser.add_argument("--attr_type", type=str, default="attntrace", 
                        choices=['tracllm', 'loo','stc','shapley', 'self_citation','attntrace','avg_attention'],
                        help="Type of attribution method to use.")
    parser.add_argument('--K', type=int, default=5, 
                    help="Report top-K most important texts that lead to the output.")
    parser.add_argument("--explanation_level", type=str, default="segment", 
                    choices=['sentence', 'paragraph', 'segment'],
                    help="Level of explanation granularity.")

    # General args
    parser.add_argument('--model_name', type=str, default='llama3.1-8b', choices=['llama3.1-8b','llama3.1-70b','gpt4o', "gpt4o-mini", "gpt4.1-mini",'qwen2-7b','qwen1.5-7b','gemma3-1b','phi3-mini','qwen2.5-7b','deepseek-v3','deepseek-r1','gemini-2.0-flash', 'gemini-1.5-flash','claude-haiku-3.5','claude-haiku-3'],
                        help="Name of the model to be used.")
    parser.add_argument('--surrogate_model_name', type=str, default='llama3.1-8b', choices=['llama3.1-8b','llama3.1-70b','chatglm4-9b','gpt4o', "gpt4o-mini", "gpt4.1-mini",'qwen2-7b','qwen2.5-7b','deepseek-v3','deepseek-r1','gemini-2.0-flash', 'gemini-1.5-flash','claude-haiku-3.5','claude-haiku-3'],
                        help="Name of the surrogate model to be used (for closed-source models).")
    parser.add_argument("--dataset_name", type=str, default='nq-poison',
                        choices=['nq-poison', 'hotpotqa-poison', 'msmarco-poison', 'nq-poison-combinatorial','nq-poison-insufficient','nq-poison-correctness','nq-poison-hotflip','nq-poison-safety',# RAG with knowledge corruption attack
                                 "narrativeqa", "musique", "qmsum", # Prompt injection attack to LongBench datasets, please set '--prompt_injection_attack'.
                                 'srt', 'mrt'], # Needle-in-haystack datasets
                        help="Name of the dataset to be used.")
    # AttnTrace args
    parser.add_argument('--avg_k', type=int, default=5, 
                        help="Top-k averaging.")
    parser.add_argument('--q', type=float, default=0.4, 
                        help="Subsampling ratio.")
    parser.add_argument('--B', type=int, default=30, 
                        help="Number of subsamples.")
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
    
    # Load method-specific configs and apply them
    args = load_method_configs(args)
    
    print(args)
    return args



def main_attribute(args,attr, question: str, contexts: list, answer: str, citations: list, target_answer = None):
    """
    Perform attribution for a given question and its contexts.

    Args:
        attr: The attribution method to be used.
        question (str): The question being evaluated.
        topk_contexts (list): The top-k contexts retrieved for the question. For LongBench
        answer (str): The answer provided by the model.
        citations (list): Citations related to the answer.s
        target_answer (str, optional): The target answer for evaluation. Defaults to None.
    """
    texts,important_ids, importance_scores, time,ensemble_list = attr.attribute(question, contexts, answer)

    if args.verbose ==1:
        attr.visualize_results(texts,question,answer, important_ids,importance_scores)
    # Create a dictionary to store the results of the attribution process
    dp_result = {
        'question': question,           # The question being evaluated
        'contexts': contexts, # The contexts. Have length>1 if the context is already segmented (E.g., PoisonedRAG)
        'answer': answer,               # The answer provided by the model
        'gt_important_texts': citations,# Ground-truth texts that lead to the answer
        'scores': importance_scores,               # Importance scores for the contexts
        'important_ids': important_ids, # IDs of the important contexts
        'time': time,                   # Time taken for the attribution process
        'target_answer': target_answer, # The target answer for evaluation
        'ensemble_list': ensemble_list  # List of ensemble results
    }
    # Use evaluator to evaluate attribution
    return dp_result

def main(args):
    if args.dataset_name in ['nq-poison', 'hotpotqa-poison', 'msmarco-poison','nq-poison-combinatorial','nq-poison-insufficient','nq-poison-correctness','nq-poison-hotflip','nq-poison-safety']: 
        benchmark = 'PoisonedRAG'
    elif args.dataset_name in ["narrativeqa",  "musique",  "qmsum"]:
        benchmark = 'LongBench'
    else: raise KeyError(f"Please use supported datasets")
    results_path = args.results_path
    
    # Load dataset and random select
    print("Loading Dataset!")
    dataset = _load_dataset(args.dataset_name, args.retriever, args.retrieval_k, 
                           model_name=args.model_name, shot=1, seed=args.seed,num_poison = args.inject_times)
    # Load LLM and init Attribution
    print("Loading LLM!")
    llm = create_model(config_path = f'model_configs/{args.model_name}_config.json', device = f"cuda:{args.gpu_id}")
    
    if "gpt" in args.model_name or "deepseek" in args.model_name or "gemini" in args.model_name or "claude" in args.model_name:
        surrogate_llm = create_model(config_path = f'model_configs/{args.surrogate_model_name}_config.json', device = f"cuda:{args.gpu_id}")
        attr = create_attr(args, llm=surrogate_llm)
    else:
        attr = create_attr(args, llm=llm)
    attr_results = []
    if benchmark == "LongBench":
        attacker = PI.create_attacker(args.prompt_injection_attack)

    data_num = 0 #initialize a counter for data number
    ASV_counter = 0
    clean_ASV_counter = 0

    for idx, dp in enumerate(dataset):
        print(f"\n------------------Start question {idx} -------------------")
        
        # Save results every 10 questions
        if idx > 1 and idx % 10 == 0:
            _save_results(args, attr_results, results_path)

        if benchmark == 'LongBench':
            # Extract context and question for LongBench
            contexts = dp['context']
            question = dp["input"]
            gt_answer = dp["answers"]

            # Get the length of the context, if it is longer than max_length, truncate it
            context_length = llm.get_prompt_length(contexts)
            if context_length > args.max_length:
                contexts = llm.cut_context(contexts,args.max_length)
            print("Question:", question)
            print("Context length:", context_length)

            # Generate a clean prompt and query the LLM. Used to calculate attack success rate without attack
            clean_prompt = wrap_prompt(question, [contexts])
            clean_answer = llm.query(clean_prompt)

            # Inject adversarial content
            contexts= attacker.inject(args, contexts, query=question)
            gt_important_texts = attacker.get_injected_prompt()
            target_response = attacker.target_answer

            # Query the LLM with the injected context
            prompt = wrap_prompt(question, [contexts])
            answer = llm.query(prompt)
            print("LLM's answer: [", answer, "]")
            print("Target answer: [", target_response, "]")

            # Check if the target response is in the LLM's answer
            ASV = clean_str(target_response) in clean_str(answer)
            ASV_clean = clean_str(target_response) in clean_str(clean_answer)
            if ASV_clean:
                clean_ASV_counter += 1
            if not ASV and args.prompt_injection_attack != 'none':
                data_num += 1
                print(f"Attack fails, continue")
                continue
            else:
                data_num += 1
                ASV_counter += 1

            print("Current ASV: ", ASV_counter / data_num)
            print("Current ASV clean: ", clean_ASV_counter / data_num)
            
            # Perform attribution and append results
            print("-----Begin attribute---")
            dp_result = main_attribute(args,attr, question, [contexts], answer, gt_important_texts, target_response)
            attr_results.append(dp_result)
            print('done!')
            if data_num >= args.data_num:
                break

        elif benchmark == 'PoisonedRAG':
            # Extract question and contexts for PoisonedRAG
            question = dp['question']
            topk_contexts = dp['topk_contents']
            incorrect_answer = dp["incorrect_answer"]
            injected_adv = dp["injected_adv"]

            # Generate prompt and query the LLM
            prompt = wrap_prompt(question, topk_contexts, split_token = "\n")

            answer = llm.query(prompt)

            # Generate clean prompt without poisoned content
            injected_ids,_ = get_gt_ids(topk_contexts, injected_adv)
            clean_prompt = wrap_prompt(question, remove_specific_indexes(topk_contexts, injected_ids), split_token = "\n")
            clean_answer = llm.query(clean_prompt)

            print("Question: ", question)
            print("LLM's answer: [", answer, "]")
            print("Target answer: [", incorrect_answer, "]")

            ASV = clean_str(incorrect_answer) in clean_str(answer)
            ASV_clean = clean_str(incorrect_answer) in clean_str(clean_answer)

            if ASV_clean == False:
                clean_ASV_counter += 0
            else:
                clean_ASV_counter += 1
            if ASV == False:
                data_num += 1
                print(f"Attack fails, continue")
                continue
            else:
                data_num += 1
                ASV_counter += 1
            
            print("Current ASV: ", ASV_counter / data_num)
            print("Current ASV clean: ", clean_ASV_counter / data_num)

            # Perform attribution and append results
            print("-----Begin attribute---")
            topk_contexts = newline_pad_contexts(topk_contexts)
            dp_result = main_attribute(args,attr, question, topk_contexts, answer, injected_adv, incorrect_answer)
            attr_results.append(dp_result)
            print('done!')
            if data_num >= args.data_num:
                break
        
        
    # Save final results
    _save_results(args, attr_results, results_path)
    
    if args.dataset_name in ['musique', 'narrativeqa', 'qmsum']:
        evaluate_prompt_injection(args,llm)
    elif args.dataset_name in ['nq-poison', 'hotpotqa-poison', 'msmarco-poison','nq-poison-combinatorial','nq-poison-insufficient','nq-poison-correctness','nq-poison-hotflip','nq-poison-safety']:
        evaluate_poison_rag(args,llm)
    # Delete the model, tokenizer, and any other large objects to free memory
    del llm
    del attr  # Delete the attribution object
    del attr_results  # Delete results list
    
    # Force deletion of any CUDA tensors
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                del obj
        except:
            pass
            
    # Run garbage collection multiple times to ensure cleanup
    gc.collect()
    gc.collect()
    
    # Clear CUDA cache
    torch.cuda.empty_cache()
    
    # Reset peak memory stats
    torch.cuda.reset_peak_memory_stats()

if __name__ == '__main__':
    
    args = parse_args()
    sleep_time = random.randint(0, 10)
    time.sleep(sleep_time)
    wait_for_available_gpu_memory(20, device=int(args.gpu_id), check_interval=300)
    setup_seeds(args.seed)
    torch.cuda.empty_cache()
    if args.evaluate_saved == False:
        main(args)

    #if "attention" in args.attr_type:
    #    evaluate_dispersion_index(args)    

