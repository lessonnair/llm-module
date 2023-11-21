# -*- coding: utf-8 -*-

from .basic_runner import Task
from datasets import load_dataset
from modules.core.template.base_template import *
from trl.trainer import ConstantLengthDataset


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

        self.prompt_column = self.pop_dict(params, "prompt_column")
        self.query_column = self.pop_dict(params, "query_column")
        self.history_column = self.pop_dict(params, "history_column")
        self.response_column = self.pop_dict(params, "response_column")
        self.system_column = self.pop_dict(params, "system_column")

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

        data_kwargs = {
            "prompt_column": self.prompt_column,
            "query_column": self.query_column,
            "history_column": self.history_column,
            "response_column": self.response_column,
            "system_column": self.system_column
        }

        if self.stage == "pt":
            def preprocess_func(examples):
                return preprocess_pretrain_dataset(self.tokenizer,
                                                   examples,
                                                   self.cutoff_len,
                                                   **data_kwargs)

            dataset = dataset.filter(lambda example: example[self.prompt_column])
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
                                                  examples,
                                                  **data_kwargs)

            dataset = dataset.filter(lambda example: example[self.prompt_column] and example[self.response_column])
        elif self.stage == "rm":
            def preprocess_func(examples):
                return preprocess_pairwise_dataset(self.template,
                                                   self.tokenizer,
                                                   self.cutoff_len,
                                                   examples,
                                                   **data_kwargs)

            dataset = dataset.filter(
                lambda example: example[self.prompt_column] and len(example[self.response_column]) > 1)
        else:
            def preprocess_func(examples):
                return preprocess_unsupervised_dataset(examples, **data_kwargs)

            dataset = dataset.filter(lambda example: example[self.prompt_column])

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
