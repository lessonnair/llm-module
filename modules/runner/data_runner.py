# -*- coding: utf-8 -*-

from .basic_runner import Task
from typing import Any, Dict, List
from datasets import load_dataset
from itertools import chain
from modules.etl.preprocess import *
from modules.template.base_template import *


class DatasetLoader(Task):

    def __init__(self, config, name=None):
        super(DatasetLoader, self).__init__(config, name=name)

        params = self.get_section_params()

        self.path = self.pop_dict(params, "path")
        self.data_files = self.get_config_list("data_files")
        self.streaming = self.get_config("streaming")

        self.tokenizer = self.get_instance("tokenizer")
        self.cutoff_len = self.pop_dict(params, "cutoff_len")
        self.sft_packing = self.pop_dict(params, "sft_packing")

        template = self.pop_dict(params, "template")

        if template is not None:
            self.template = get_template_and_fix_tokenizer(template, self.tokenizer)
        else:
            self.template = None

        self.train_on_prompt = self.pop_dict(params, "train_on_prompt")

        params.pop("tokenizer")

        self.params = params

    def main_handle(self):
        dataset = load_dataset(self.path, **self.params)
        column_names = list(next(iter(dataset)).keys())

        if self.stage == "pt":
            def preprocess_func(examples):
                return preprocess_pretrain_dataset(self.tokenizer,
                                                   examples,
                                                   self.cutoff_len)

            dataset = dataset.filter(lambda example: example["instruction"])
        elif self.stage == "sft":
            def preprocess_func(examples):
                if self.sft_packing:
                    preprocess_supervised_func = preprocess_packed_supervised_dataset
                else:
                    preprocess_supervised_func = preprocess_supervised_dataset
                return preprocess_supervised_func(self.template,
                                                  self.tokenizer,
                                                  self.cutoff_len,
                                                  self.train_on_prompt,
                                                  examples)
            dataset = dataset.filter(lambda example: example["prompt"] and example["response"])
        elif self.stage == "rm":
            def preprocess_func(examples):
                return preprocess_pairwise_dataset(examples)
            dataset = dataset.filter(lambda example: example["prompt"] and len(example["response"]) > 1)
        else:
            def preprocess_func(examples):
                return preprocess_unsupervised_dataset(examples)
            dataset = dataset.filter(lambda example: example["prompt"])

        if not self.streaming:
            kwargs = dict(
                num_proc=1,
                load_from_cache_file=False,
                desc="Running tokenizer on dataset"
            )
        dataset = dataset.map(
            preprocess_func,
            batched=True,
            remove_columns=column_names,
            **kwargs
        )
        self.inst = dataset
