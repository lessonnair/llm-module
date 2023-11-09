# -*- coding: utf-8 -*-

from typing import Any, Dict, Generator, List
from itertools import chain


def construct_example(examples: Dict[str, List[Any]]) -> Generator[Any, None, None]:
    for i in range(len(examples["prompt"])):
        query, response = examples["prompt"][i], examples["response"][i]
        query = query + "\n" + examples["query"][i] if "query" in examples and examples["query"][i] else query
        history = examples["history"][i] if "history" in examples else None
        system = examples["system"][i] if "system" in examples else None
        yield query, response, history, system


def preprocess_pretrain_dataset(tokenizer,
                                examples: Dict[str, List[Any]],
                                cutoff_len) -> Dict[str, Any]:
    kwargs = dict(add_special_tokens=True)

    if hasattr(tokenizer, "add_eos_token"):  # for LLaMA tokenizer
        setattr(tokenizer, "add_eos_token", True)

    tokenizer.pad_token = tokenizer.eos_token

    tokenized_examples = tokenizer(examples["instruction"], **kwargs)
    concatenated_examples = {k: list(chain(*tokenized_examples[k])) for k in tokenized_examples.keys()}
    total_length = len(concatenated_examples[list(concatenated_examples.keys())[0]])
    block_size = cutoff_len
    # we drop the small remainder, and if the total_length < block_size, we exclude this batch
    total_length = (total_length // block_size) * block_size
    # split by chunks of cutoff_len
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    return result


def preprocess_supervised_dataset(tokenizer,
                                  examples: Dict[str, List[Any]]):
    model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

    for query, response, history, system in construct_example(examples):
        input_ids, labels = [], []

        for turn_idx, (source_ids, target_ids) 


