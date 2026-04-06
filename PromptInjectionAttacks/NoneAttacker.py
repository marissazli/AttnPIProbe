from .Attacker import Attacker
from src.utils import contexts_to_sentences
import random

#no attack
class NoneAttacker(Attacker):
    def __init__(self, attack_strategy):
        super().__init__(attack_strategy)

    def inject(self, args,clean_data, query = None):
        self.args = args
        self.dataset_name = self.args.dataset_name
        target_answer = self.generate_incorrect_answer(query)
        self.target_answer = target_answer
        inject_data = ""
       
        self.inject_data = inject_data
        all_sentences = contexts_to_sentences([clean_data])
        num_sentences = len(all_sentences)
        random.seed(num_sentences)
        # Generate a random position
        context = ''.join(all_sentences)
        return f'{context}\n'

    def get_injected_prompt(self):
        return [self.inject_data]
