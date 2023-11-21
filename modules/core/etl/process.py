# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Any, Dict, Generator, List
from itertools import chain
from modules.util.constants import IGNORE_INDEX

@dataclass
class DataProcessor(object):

    prompt_column: str = "prompt"
    query_column: str = "query"
    history_column: str = "history"
    response_column: str = "response"
    system_column: str = "system"
    cutoff_len: int = None

    def construct_example(self,
                          examples: Dict[str, List[Any]]
                          ) -> Generator[Any, None, None]:
        for i in range(len(examples[self.prompt_column])):
            query, response = examples[self.prompt_column][i], examples[self.response_column][i]
            if self.query_column in examples and examples[self.query_column][i]:
                query = query + "\n" + examples[self.query_column][i]
            history = None
            if self.history_column in examples:
                history = examples[self.history_column][i]
            system = None
            if self.system_column in examples:
                system = examples[self.system_column]
            yield query, response, history, system

    def process(self,
                tokenizer,
                examples: Dict[str, List[Any]],
                tag: str
                ) -> Dict[str, Any]:
        if tag == "pt":
            kwargs = dict(add_special_tokens=True)
            if hasattr(tokenizer, "add_eos_token"):  # for LLaMA tokenizer
                setattr(tokenizer, "add_eos_token", True)

            tokenizer.pad_token = tokenizer.eos_token
            tokenized_examples = tokenizer(examples[self.prompt_column], **kwargs)
            concatenated_examples = {k: list(chain(*tokenized_examples[k])) for k in tokenized_examples.keys()}
            total_length = len(concatenated_examples[list(concatenated_examples.keys())[0]])
            block_size = self.cutoff_len
            # we drop the small remainder, and if the total_length < block_size, we exclude this batch
            total_length = (total_length // block_size) * block_size
            # split by chunks of cutoff_len
            result = {
                k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            return result

        else:
            for query, response, history, system in self.construct_example(examples):
                input_ids, labels = [], []







