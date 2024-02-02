import random
import re
import tokenize 

from typing import Callable, List, Dict
from regex import B
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from dataclasses import dataclass

from collections import defaultdict

MAX_ITER = 100

@dataclass
class Instruction:
    instruction_template: str
    substitutions: Dict[str, str]

    deltas = {}

    def get_tokenized_subs(self, tokenizer):
        tokenized_subs = {}
        for key, value in self.substitutions.items():
            tokenized_subs[key] = tokenizer.encode(" " + value, add_special_tokens=False)
        return tokenized_subs

    def to_string(self):
        instruction_str = self.instruction_template
        for key, value in self.substitutions.items():
            instruction_str = instruction_str.replace(key, value)

        return instruction_str

class InstructionDataset(Dataset):

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        instructions: List[Instruction],
        tokenize_instructions: Callable,
        device: str,
    ):
        self.tokenizer = tokenizer
        self.instructions = instructions
        self.device = device

        self.toks = tokenize_instructions(
            tokenizer=self.tokenizer,
            instructions=[instruction.to_string() for instruction in self.instructions],
        ).to(device)

        self.str_toks = [
            [self.tokenizer.decode(tok) for tok in self.toks[i]]
            for i in range(self.toks.shape[0])
        ]

        self.str_prompts = [
            self.tokenizer.decode(self.toks[i]) for i in range(self.toks.shape[0])
        ]

        self.deltas = []
        for i, instruction in enumerate(self.instructions):
            delta = {}
            inst_tok_subs = instruction.get_tokenized_subs(self.tokenizer)
            for key, value in inst_tok_subs.items():
                start_val = self._find_first_instance(value, self.toks[i].tolist())
                delta[key] = slice(start_val, start_val + len(value))
            self.deltas.append(delta)

    def _find_first_instance(self, tokenized_key, tokenized_str):
        for i in range(len(tokenized_str) - len(tokenized_key)):
            if tokenized_str[i:i+len(tokenized_key)] == tokenized_key:
                return i
        return -1

    def __len__(self):
        return len(self.instructions)

    def __getitem__(self, key):
        sliced_instructions = self.instructions[key]
        sliced_dataset = InstructionDataset(
            self.tokenizer,
            instructions=sliced_instructions,
            device=self.device,
        )
        return sliced_dataset

class PairedInstructionDataset(Dataset):

    def _assert_equal_tok_lens(self, str_0: str, str_1: str, tokenizer: AutoTokenizer):
        tok_len_0 = len(tokenizer.encode(str_0, add_special_tokens=False))
        tok_len_1 = len(tokenizer.encode(str_1, add_special_tokens=False))
        assert tok_len_0 == tok_len_1, f"token length mismatch:\n\t{str_0} ({tok_len_0})\n\t{str_1} ({tok_len_1})"

    def gen_paired_instructions_uniform(
        self,
        N: int,
        instruction_templates: List[str],
        harmful_substitution_map: Dict[str, List[str]],
        harmless_substitution_map: Dict[str, List[str]],
        tokenizer: AutoTokenizer,
    ) -> List[Instruction]:

        # map each variable to a dict, mapping token lengths to strings
        harmful_substitution_map_by_tok_length: Dict[str, Dict[int, List[str]]] = {}
        harmless_substitution_map_by_tok_length: Dict[str, Dict[int, List[str]]] = {}

        for variable, values in harmful_substitution_map.items():
            if variable not in harmful_substitution_map_by_tok_length: harmful_substitution_map_by_tok_length[variable] = {}
            for value in values:
                tok_length = len(tokenizer.encode(" " + value, add_special_tokens=False))
                if tok_length not in harmful_substitution_map_by_tok_length[variable]: harmful_substitution_map_by_tok_length[variable][tok_length] = []
                harmful_substitution_map_by_tok_length[variable][tok_length].append(value)

        for variable, values in harmless_substitution_map.items():
            if variable not in harmless_substitution_map_by_tok_length: harmless_substitution_map_by_tok_length[variable] = {}
            for value in values:
                tok_length = len(tokenizer.encode(" " + value, add_special_tokens=False))
                if tok_length not in harmless_substitution_map_by_tok_length[variable]: harmless_substitution_map_by_tok_length[variable][tok_length] = []
                harmless_substitution_map_by_tok_length[variable][tok_length].append(value)

        harmful_instructions, harmless_instructions = [], []

        for _ in range(N):
            instruction_template = random.choice(instruction_templates)

            harmful_substitutions = {}
            harmless_substitutions = {}

            for variable in harmful_substitution_map.keys():

                # draw substitutions until we pick one that has matching length
                for _ in range(MAX_ITER):
                    harmful_substitutions[variable] = random.choice(harmful_substitution_map[variable])
                    tok_len = len(tokenizer.encode(" " + harmful_substitutions[variable], add_special_tokens=False))
                    if tok_len in harmless_substitution_map_by_tok_length[variable]:
                        harmless_substitutions[variable] = random.choice(harmless_substitution_map_by_tok_length[variable][tok_len])
                        break
                    else: continue
                if variable not in harmless_substitutions:
                    raise ValueError(f"Could not find a harmless substitution for {variable} with token length {tok_len} after {MAX_ITER} iterations")

            harmful_instruction = Instruction(instruction_template, harmful_substitutions)
            harmless_instruction = Instruction(instruction_template, harmless_substitutions)

            harmful_instructions.append(harmful_instruction)
            harmless_instructions.append(harmless_instruction)
            self._assert_equal_tok_lens(harmful_instructions[-1].to_string(), harmless_instructions[-1].to_string(), tokenizer)

        return harmful_instructions, harmless_instructions

    def __init__(
        self,
        N: int,
        instruction_templates: List[str],
        harmful_substitution_map: Dict[str, List[str]],
        harmless_substitution_map: Dict[str, List[str]],
        tokenizer: AutoTokenizer,
        tokenize_instructions: Callable,
        seed: int = 42,
        device: str = "cuda",
    ):
        self.N = N
        self.harmful_substitution_map = harmful_substitution_map
        self.harmless_substitution_map = harmless_substitution_map
        self.tokenizer = tokenizer
        self.instruction_templates = instruction_templates
        self.seed = seed
        self.device = device

        random.seed(self.seed)

        harmful_instructions, harmless_instructions = self.gen_paired_instructions_uniform(
            N=self.N,
            instruction_templates=self.instruction_templates,
            harmful_substitution_map = self.harmful_substitution_map,
            harmless_substitution_map = self.harmless_substitution_map,
            tokenizer=self.tokenizer
        )

        self.harmful_dataset = InstructionDataset(
            tokenizer,
            instructions=harmful_instructions,
            tokenize_instructions=tokenize_instructions,
            device=self.device,
        )
        self.harmless_dataset = InstructionDataset(
            tokenizer,
            instructions=harmless_instructions,
            tokenize_instructions=tokenize_instructions,
            device=self.device,
        )