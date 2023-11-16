# -*- coding: utf-8 -*-

from typing import Any, Dict, Generator, List
from itertools import chain
from ..util.constants import IGNORE_INDEX


def construct_example(examples: Dict[str, List[Any]],
                      prompt_column="prompt",
                      query_column="query",
                      history_column="history",
                      response_column="response",
                      system_column="system"
                      ) -> Generator[Any, None, None]:
    for i in range(len(examples[prompt_column])):
        query, response = examples[prompt_column][i], examples[response_column][i]
        if query_column in examples and examples[query_column][i]:
            query = query + "\n" + examples[query_column][i]
        history = examples[history_column][i] if history_column in examples else None
        system = examples[system_column][i] if system_column in examples else None
        yield query, response, history, system


def preprocess_pretrain_dataset(tokenizer,
                                examples: Dict[str, List[Any]],
                                cutoff_len,
                                prompt_column="prompt",
                                query_column="query",
                                history_column="history",
                                response_column="response",
                                system_column="system"
                                ) -> Dict[str, Any]:
    # build grouped texts with format `X1 X2 X3 ...`
    kwargs = dict(add_special_tokens=True)

    if hasattr(tokenizer, "add_eos_token"):  # for LLaMA tokenizer
        setattr(tokenizer, "add_eos_token", True)

    tokenizer.pad_token = tokenizer.eos_token

    tokenized_examples = tokenizer(examples[prompt_column], **kwargs)
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


def preprocess_supervised_dataset(template,
                                  tokenizer,
                                  cutoff_len,
                                  train_on_prompt,
                                  examples: Dict[str, List[Any]],
                                  prompt_column="prompt",
                                  query_column="query",
                                  history_column="history",
                                  response_column="response",
                                  system_column="system"):
    # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
    # for multiturn examples, we only mask the prompt part in each prompt-response pair.
    model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

    for query, response, history, system in construct_example(examples,
                                                              prompt_column=prompt_column,
                                                              query_column=query_column,
                                                              history_column=history_column,
                                                              response_column=response_column,
                                                              system_column=system_column):
        input_ids, labels = [], []

        for turn_idx, (source_ids, target_ids) in enumerate(template.encode_multiturn(
                tokenizer, query, response, history, system
        )):
            total_len = len(source_ids) + len(target_ids)
            max_source_len = int(cutoff_len * (len(source_ids) / total_len))
            max_target_len = int(cutoff_len * (len(target_ids) / total_len))

            if len(source_ids) > max_source_len:
                source_ids = source_ids[:max_source_len]
            if len(target_ids) > max_target_len:
                target_ids = target_ids[:max_target_len]

            if train_on_prompt:
                source_mask = source_ids
            elif turn_idx != 0 and template.efficient_eos:
                source_mask = [tokenizer.eos_token_id] + [IGNORE_INDEX] * (len(source_ids) - 1)
            else:
                source_mask = [IGNORE_INDEX] * len(source_ids)

            input_ids += source_ids + target_ids
            labels += source_mask + target_ids

        if template.efficient_eos:
            input_ids += [tokenizer.eos_token_id]
            labels += [tokenizer.eos_token_id]

        if len(input_ids) > cutoff_len:
            input_ids = input_ids[:cutoff_len]
            labels = labels[:cutoff_len]

        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append([1] * len(input_ids))
        model_inputs["labels"].append(labels)

    return model_inputs


def preprocess_packed_supervised_dataset(template,
                                         tokenizer,
                                         cutoff_len,
                                         train_on_prompt,
                                         examples: Dict[str, List[Any]],
                                         prompt_column="prompt",
                                         query_column="query",
                                         history_column="history",
                                         response_column="response",
                                         system_column="system") -> Dict[str, Any]:
    # build inputs with format `<bos> X1 Y1 <eos> <bos> X2 Y2 <eos>`
    # and labels with format `<ignore> ... <ignore> Y1 <eos> <ignore> ... <ignore> Y2 <eos>`
    model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
    input_ids, labels = [], []
    for query, response, history, system in construct_example(examples,
                                                              prompt_column=prompt_column,
                                                              query_column=query_column,
                                                              history_column=history_column,
                                                              response_column=response_column,
                                                              system_column=system_column):
        for turn_idx, (source_ids, target_ids) in enumerate(template.encode_multiturn(
                tokenizer, query, response, history, system
        )):
            if train_on_prompt:
                source_mask = source_ids
            elif turn_idx != 0 and template.efficient_eos:
                source_mask = [tokenizer.eos_token_id] + [IGNORE_INDEX] * (len(source_ids) - 1)
            else:
                source_mask = [IGNORE_INDEX] * len(source_ids)
            input_ids += source_ids + target_ids
            labels += source_mask + target_ids

    if template.efficient_eos:
        input_ids += [tokenizer.eos_token_id]
        labels += [tokenizer.eos_token_id]

    total_length = int(input_ids)
    block_size = cutoff_len
    total_length = (total_length // block_size) * block_size

    for i in range(0, total_length, block_size):
        model_inputs["input_ids"].append(input_ids[i: i + block_size])
        model_inputs["attention_mask"].append([1] * block_size)
        model_inputs["labels"].append(labels[i: i + block_size])

    return model_inputs


def preprocess_unsupervised_dataset(template,
                                    tokenizer,
                                    cutoff_len,
                                    train_on_prompt,
                                    examples: Dict[str, List[Any]],
                                    prompt_column="prompt",
                                    query_column="query",
                                    history_column="history",
                                    response_column="response",
                                    system_column="system") -> Dict[str, Any]:
    # build inputs with format `<bos> X` and labels with format `Y <eos>`
    model_inputs = {"input_ids": [], "attention_mask": [], "label": []}

    for query, response, history, system in construct_example(examples,
                                                              prompt_column=prompt_column,
                                                              query_column=query_column,
                                                              history_column=history_column,
                                                              response_column=response_column,
                                                              system_column=system_column):
        input_ids, labels = template.encode_oneturn(tokenizer, query, response, history, system)

        if template.efficient_eos:
            labels += [tokenizer.eos_token_id]

        if len(input_ids) > cutoff_len:
            input_ids = input_ids[:cutoff_len]
        if len(labels) > cutoff_len:
            labels = labels[:cutoff_len]

        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append([1] * len(input_ids))
        model_inputs["labels"].append(labels)
    return model_inputs


def preprocess_pairwise_dataset(template,
                                tokenizer,
                                cutoff_len,
                                examples: Dict[str, List[Any]],
                                prompt_column="prompt",
                                query_column="query",
                                history_column="history",
                                response_column="response",
                                system_column="system"):
    # build input pairs with format `<bos> X`, `Y1 <eos>` and `Y2 <eos>`
    model_inputs = {"prompt_ids": [], "chosen_ids": [], "rejected_ids": []}
    for query, response, history, system in construct_example(examples,
                                                              prompt_column=prompt_column,
                                                              query_column=query_column,
                                                              history_column=history_column,
                                                              response_column=response_column,
                                                              system_column=system_column):
        prompt_ids, chosen_ids = template.encode_oneturn(tokenizer, query, response[0], history, system)
        _, rejected_ids = template.encode_oneturn(tokenizer, query, response[1], history, system)

        if template.efficient_eos:
            chosen_ids += [tokenizer.eos_token_id]
            rejected_ids += [tokenizer.eos_token_id]

        total_len = len(prompt_ids) + max(len(chosen_ids), len(rejected_ids))
        max_source_len = int(cutoff_len * (len(prompt_ids) / total_len))
        max_target_len = int(cutoff_len * (max(len(chosen_ids), len(rejected_ids)) / total_len))

        if len(prompt_ids) > max_source_len:
            prompt_ids = prompt_ids[:max_source_len]
        if len(chosen_ids) > max_target_len:
            chosen_ids = chosen_ids[:max_target_len]
        if len(rejected_ids) > max_target_len:
            rejected_ids = rejected_ids[:max_target_len]

        model_inputs["prompt_ids"].append(prompt_ids)
        model_inputs["chosen_ids"].append(chosen_ids)
        model_inputs["rejected_ids"].append(rejected_ids)
    return model_inputs
