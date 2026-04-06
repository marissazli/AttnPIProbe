from .attribute import *
import numpy as np
from src.utils import *
import time
import torch.nn.functional as F
import gc
from src.prompts import wrap_prompt_attention
from .attention_utils import *

class AttnTraceAttribution(Attribution):
    def __init__(self,  llm,explanation_level = "segment",K=5, avg_k=5, q=0.4, B=30, verbose =1):
        super().__init__(llm,explanation_level,K,verbose)
        self.model = llm.model # Use float16 for the model
        self.model_type = llm.provider
        self.tokenizer = llm.tokenizer
        self.avg_k = avg_k
        self.q = q
        self.B = B
        self.layers = range(len(self.model.model.layers))
        self.explanation_level = explanation_level
 
    def loss_to_importance(self,losses, sentences_id_list):

        importances = np.zeros(len(sentences_id_list))

        for i in range(1,len(losses)):
            group = np.array(losses[i][0])
            last_group = np.array(losses[i-1][0])
            
            group_loss=np.array(losses[i][1])
            last_group_loss=np.array(losses[i-1][1])
            if len(group)-len(last_group) == 1:
                feature_index = [item for item in group if item not in last_group]
                #print(feature_index)
                #print(last_group,group, last_group_label,group_label)
                importances[feature_index[0]]+=(last_group_loss-group_loss)
        return importances
    def attribute(self, question: str, contexts: list, answer: str, customized_template: str = None):
        start_time = time.time()
        model = self.model
        tokenizer = self.tokenizer
        model.eval()  # Set model to evaluation mode
        contexts = split_context(self.explanation_level, contexts)
        #print("contexts: ", contexts)
        # Get prompt and target token ids
        prompt_part1, prompt_part2 = wrap_prompt_attention(question,customized_template)
        prompt_part1_ids = tokenizer(prompt_part1, return_tensors="pt").input_ids.to(model.device)[0]
        context_ids_list = [tokenizer(context, return_tensors="pt").input_ids.to(model.device)[0][1:] for context in contexts]
        prompt_part2_ids = tokenizer(prompt_part2, return_tensors="pt").input_ids.to(model.device)[0]
        target_ids = tokenizer(answer, return_tensors="pt").input_ids.to(model.device)[0]
        avg_importance_values = np.zeros(len(context_ids_list))
        idx_frequency = {idx: 0 for idx in range(len(context_ids_list))}
        for t in range(self.B):
            # Combine prompt and target tokens
            
            # Randomly subsample half of the context_ids_list
            num_samples = int(len(context_ids_list)*self.q)
            sampled_indices = np.sort(np.random.permutation(len(context_ids_list))[:num_samples])
            
            sampled_context_ids = [context_ids_list[idx] for idx in sampled_indices]
            input_ids = torch.cat([prompt_part1_ids] + sampled_context_ids + [prompt_part2_ids, target_ids], dim=-1).unsqueeze(0)
            self.context_length = sum(len(context_ids) for context_ids in sampled_context_ids)
            self.prompt_length = len(prompt_part1_ids) + self.context_length + len(prompt_part2_ids)
            # Directly calculate the average attention of each answer token to the context tokens to save memory

            with torch.no_grad():
                outputs = model(input_ids, output_hidden_states=True)  # Choose the specific layer you want to use
            hidden_states = outputs.hidden_states
            with torch.no_grad():

                avg_attentions = None  # Initialize to None for accumulative average
                for i in self.layers:
                    attentions = get_attention_weights_one_layer(model, hidden_states, i, attribution_start=self.prompt_length,model_type=self.model_type)
                    batch_mean = attentions
                    if avg_attentions is None:
                        avg_attentions = batch_mean[:, :, :, len(prompt_part1_ids):len(prompt_part1_ids) + self.context_length]
                    else:
                        avg_attentions += batch_mean[:, :, :, len(prompt_part1_ids):len(prompt_part1_ids) + self.context_length]
                avg_attentions = (avg_attentions / (len(self.layers))).mean(dim=0).mean(dim=(0, 1)).to(torch.float16)
            
            importance_values = avg_attentions.to(torch.float32).cpu().numpy()

            # Decode tokens to readable format

            # Calculate cumulative sums of context lengths
            context_lengths = [len(context_ids) for context_ids in sampled_context_ids[:-1]]
            start_positions = np.cumsum([0] + context_lengths)
            
            # Calculate mean importance values for each context group
            group_importance_values = []
            for start, context_ids in zip(start_positions, sampled_context_ids):
                end = start + len(context_ids)
                values = np.sort(importance_values[start:end])
                k = min(self.avg_k, end-start) # Take min of 5 and actual length

                group_mean = np.mean(values[-k:]) # Take top k values
                group_importance_values.append(group_mean)
            
            group_importance_values = np.array(group_importance_values)

            
            for idx in sampled_indices:
                idx_frequency[idx] += 1

            for i, idx in enumerate(sampled_indices):
                avg_importance_values[idx] += group_importance_values[i]

        for i, idx in enumerate(context_ids_list):
            if idx_frequency[i] != 0:
                avg_importance_values[i] /= idx_frequency[i]

        # Plot sentence importance
        top_k_indices = np.argsort(avg_importance_values)[::-1][:self.K]
        # Get the corresponding importance scores
        top_k_scores = [avg_importance_values[i] for i in top_k_indices]
        
        end_time = time.time()

        gc.collect()
        torch.cuda.empty_cache()
        return contexts, top_k_indices, top_k_scores, end_time - start_time, None
