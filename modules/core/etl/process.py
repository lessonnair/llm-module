# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Union
from itertools import chain
from modules.core.render import Render
from modules.util.constants import IGNORE_INDEX


@dataclass
class DataProcessor(object):
    text_column: str = "text"
    prompt_column: str = "prompt"
    query_column: str = "query"
    history_column: str = "history"
    response_column: str = "response"
    system_column: str = "system"
    cutoff_len: int = None
    label_mask_prompt: bool = True
    sft_packing: bool = True

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

    def do_truncation(self, seq: List[Any], max_len):
        if len(seq) > max_len:
            seq = seq[:max_len]
        return seq

    def do_block_split(self,
                       seqs: Union[Dict[str, List[Any]], List[List[Any]]],
                       block_size: int):
        if isinstance(seqs, dict):
            total_len = len(seqs[list(seqs.keys())[0]])
            total_len = (total_len // block_size) * block_size

            res_map = {}
            for k, t in seqs.items():
                res = []
                for j in range(0, total_len, block_size):
                    res.append(t[j: j + block_size])
                res_map[k] = res
            return res_map
        else:
            total_len = len(seqs[0])
            total_len = (total_len // block_size) * block_size

            res_list = []
            for i, seq in enumerate(seqs):
                res = []
                for j in range(0, total_len, block_size):
                    res.append(seqs[i][j: j + block_size])
                res_list.append(res)
            return res_list

    def process(self,
                tokenizer,
                examples: Dict[str, List[Any]],
                stage: str,
                render_name: str = None
                ) -> Dict[str, Any]:
        if stage == "pt":
            kwargs = dict(add_special_tokens=True)
            if hasattr(tokenizer, "add_eos_token"):  # for LLaMA tokenizer
                setattr(tokenizer, "add_eos_token", True)

            tokenizer.pad_token = tokenizer.eos_token
            tokenized_examples = tokenizer(examples[self.text_column], **kwargs)
            concatenated_examples = {k: list(chain(*tokenized_examples[k])) for k in tokenized_examples.keys()}

            return self.do_block_split(concatenated_examples, self.cutoff_len)
        else:
            render = Render(render_name)
            if stage in ["rm", "ppo", "dpo"]:
                result = {"input_ids": [], "chosen_input_ids": [], "rejected_input_ids": []}
                for query, response, history, system in self.construct_example(examples):
                    source_ids, chosen_ids = render.render_with_history(tokenizer, query, response[0], history=history,
                                                                        system=system, multi_turn=False)
                    _, rejected_ids = render.render_with_history(tokenizer, query, response[1], history=history,
                                                                 system=system, multi_turn=False)
                    if render.efficient_eos:
                        chosen_ids += [tokenizer.eos_token_id]
                        rejected_ids += [tokenizer.eos_token_id]

                    total_len = len(source_ids) + max(len(chosen_ids), len(rejected_ids))
                    source_max_len = int(self.cutoff_len * (len(source_ids) / total_len))
                    target_max_len = int(self.cutoff_len * (max(len(chosen_ids), len(rejected_ids)) / total_len))

                    result["input_ids"].append(
                        self.do_truncation(source_ids, source_max_len)
                    )
                    result["chosen_input_ids"].append(
                        self.do_truncation(chosen_ids, target_max_len)
                    )
                    result["rejected_input_ids"].append(
                        self.do_truncation(rejected_ids, target_max_len)
                    )
                return result

            elif stage == "sft":
                result = {"input_ids": [], "attention_mask": [], "labels": []}
                if self.sft_packing:
                    input_ids, labels = [], []
                    for query, response, history, system in self.construct_example(examples):
                        for idx, (source_ids, target_ids) in enumerate(render.render_with_history(
                                tokenizer, query, response, history=history, system=system, multi_turn=True
                        )):
                            if not self.label_mask_prompt:
                                source_mask = source_ids
                            elif idx > 0 and render.efficient_eos:
                                source_mask = [tokenizer.eos_token_id] + [IGNORE_INDEX] * (len(source_ids) - 1)
                            else:
                                source_mask = [IGNORE_INDEX] * len(source_ids)
                            input_ids += source_ids + target_ids
                            labels += source_mask + target_ids

                    if render.efficient_eos:
                        input_ids += [tokenizer.eos_token_id]
                        labels += [tokenizer.eos_token_id]

                    result["input_ids"] = input_ids
                    result["attention_mask"] = [1] * len(input_ids)
                    result["labels"] = labels
                    return self.do_block_split(result, self.cutoff_len)

                else:

                    for query, response, history, system in self.construct_example(examples):
                        input_ids, labels = [], []
                        for idx, (source_ids, target_ids) in enumerate(render.render_with_history(
                                tokenizer, query, response, history=history, system=system, multi_turn=True
                        )):
                            total_len = len(source_ids) + len(target_ids)
                            source_max_len = int(self.cutoff_len * (len(source_ids) / total_len))
                            target_max_len = int(self.cutoff_len * (len(target_ids) / total_len))

                            source_ids = self.do_truncation(source_ids, source_max_len)
                            target_ids = self.do_truncation(target_ids, target_max_len)

                            if not self.label_mask_prompt:
                                source_mask = source_ids
                            elif idx > 0 and render.efficient_eos:
                                source_mask = [tokenizer.eos_token_id] + [IGNORE_INDEX] * (len(source_ids) - 1)
                            else:
                                source_mask = [IGNORE_INDEX] * len(source_ids)
                            input_ids += source_ids + target_ids
                            labels += source_mask + target_ids

                        if render.efficient_eos:
                            input_ids += [tokenizer.eos_token_id]
                            labels += [tokenizer.eos_token_id]

                        input_ids = self.do_truncation(input_ids, self.cutoff_len)
                        labels = self.do_truncation(labels, self.cutoff_len)

                        result["input_ids"].append(input_ids)
                        result["attention_mask"].append([1] * len(input_ids))
                        result["labels"].append(labels)

                return result
            else:
                result = {"input_ids": [], "attention_mask": [], "labels": []}

                for query, response, history, system in self.construct_example(examples):
                    source_ids, labels = render.render_with_history(
                        tokenizer, query, response, history=history, system=system, multi_turn=False
                    )
                    if render.efficient_eos:
                        labels += [tokenizer.eos_token_id]

                    source_ids = self.do_truncation(source_ids, self.cutoff_len)
                    labels = self.do_truncation(labels, self.cutoff_len)

                    result["input_ids"].append(source_ids)
                    result["attention_mask"].append([1] * len(source_ids))
                    result["labels"].append(labels)
                return result

    def do_filter(self, examples, stage: str):
        if stage == "pt":
            return examples[self.text_column]
        elif stage == "sft":
            return examples[self.prompt_column] and examples[self.response_column]
        elif stage == "rm":
            return examples[self.prompt_column] and len(examples[self.response_column]) > 1
        else:
            return examples[self.prompt_column]
