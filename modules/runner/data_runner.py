# -*- coding: utf-8 -*-

from .basic_runner import Task
from typing import Any, Dict, List
from datasets import load_dataset
from itertools import chain
from modules.etl.preprocess import *


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

        if self.stage == "pt":
            def preprocess_func(examples):
                return preprocess_pretrain_dataset(self.tokenizer,
                                                   examples,
                                                   self.cutoff_len)

            dataset = dataset.filter(lambda example: example["instruction"])

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
