from .Model import Model
import tiktoken
from transformers import AutoTokenizer
import time
import google.generativeai as genai

class Gemini(Model):
    def __init__(self, config):
        super().__init__(config)
        api_keys = config["api_key_info"]["api_keys"]
        api_pos = int(config["api_key_info"]["api_key_use"])
        assert (0 <= api_pos < len(api_keys)), "Please enter a valid API key to use"
        self.max_output_tokens = int(config["params"]["max_output_tokens"])
        genai.configure(api_key=api_keys[api_pos])
        # Map the model name to a valid Gemini model

        self.model = genai.GenerativeModel(self.name)
        self.llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        self.seed = 10

    def query(self, msg, max_tokens=128000):
        super().query(max_tokens)
        while True:
            try:
                generation_config = genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_output_tokens,
                    candidate_count=1
                )
                
                
                response = self.model.generate_content(
                    contents=msg,
                    generation_config=generation_config

                )
                
                # Check if response was blocked by safety filters
                if response.candidates and response.candidates[0].finish_reason == 2:
                    blocked_filter = response.prompt_feedback.safety_ratings[0].category
                    print(f"Warning: Response was blocked by {blocked_filter} safety filter. Retrying with different prompt...")
                    continue
                
                if not response.text:
                    raise ValueError("Empty response from Gemini API")
                    
                time.sleep(1)
                break
            except Exception as e:
                print(f"Error in Gemini API call: {str(e)}")
                time.sleep(100)
        return response.text
    
    def get_prompt_length(self,msg):
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        num_tokens = len(encoding.encode(msg))
        return num_tokens
    
    def cut_context(self,msg,max_length):
        tokens = self.encoding.encode(msg)
        truncated_tokens = tokens[:max_length]
        truncated_text = self.encoding.decode(truncated_tokens)
        return truncated_text
