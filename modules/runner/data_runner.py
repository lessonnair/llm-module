# -*- coding: utf-8 -*-

from .basic_runner import Task
from typing import Any, Dict, List
from datasets import load_dataset
from itertools import chain


class DatasetLoader(Task):

    def __init__(self, config, name=None):
        super(DatasetLoader, self).__init__(config, name=name)

        self.path = self.get_config("path")
        self.data_files = self.get_config_list("data_files")
        self.streaming = self.get_config("streaming")

        params = self.get_section_params()
        params.pop("path")
        self.params = params

    def main_handle(self):
        self.inst = load_dataset(self.path, **self.params)


class DatasetProcess(Task):

    def __init__(self, config, name=None):
        super(DatasetProcess, self).__init__(config, name=name)

        self.tokenizer = self.get_instance("tokenizer")
        self.stage = self.get_config("stage")
        self.cutoff_len = self.get_config("cutoff_len")

    def main_handle(self):

        dataset = self.get_instance("dataset")
        column_names = list(next(iter(dataset)).keys())

        def preprocess_pretrain_dataset(examples: Dict[str, List[Any]]) -> Dict[str, Any]:

            kwargs = dict(add_special_tokens=True)

            if hasattr(self.tokenizer, "add_eos_token"):  # for LLaMA tokenizer
                setattr(self.tokenizer, "add_eos_token", True)

            self.tokenizer.pad_token = self.tokenizer.eos_token

            tokenized_examples = self.tokenizer(examples["instruction"], **kwargs)
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

        if self.stage == "pt":
            dataset = dataset.filter(lambda example: example["instruction"])
            kwargs = dict(
                num_proc=1,
                load_from_cache_file=False,
                desc="Running tokenizer on dataset"
            )
            dataset = dataset.map(
                preprocess_pretrain_dataset,
                batched=True,
                remove_columns=column_names,
                **kwargs
            )
            self.inst = dataset
