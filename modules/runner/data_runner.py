# -*- coding: utf-8 -*-

from .basic_runner import Task
from datasets import load_dataset
from modules.core.etl.process import DataProcessor


class DatasetLoader(Task):

    def __init__(self, config, name=None, stage=None):
        super(DatasetLoader, self).__init__(config, name=name)

        params = self.get_section_params()

        self.stage = stage

        self.path = self.pop_dict(params, "path")
        self.data_files = self.get_config_list("data_files")
        self.streaming = self.get_config("streaming")

        self.tokenizer = self.get_instance("tokenizer")

        text_column = self.pop_dict(params, "text_column")
        prompt_column = self.pop_dict(params, "prompt_column")
        query_column = self.pop_dict(params, "query_column")
        history_column = self.pop_dict(params, "history_column")
        response_column = self.pop_dict(params, "response_column")
        system_column = self.pop_dict(params, "system_column")
        cutoff_len = self.pop_dict(params, "cutoff_len")
        label_mask_prompt = self.pop_dict(params, "label_mask_prompt")
        sft_packing = self.pop_dict(params, "sft_packing")

        self.data_processor = DataProcessor(
            text_column=text_column,
            prompt_column=prompt_column,
            query_column=query_column,
            history_column=history_column,
            response_column=response_column,
            system_column=system_column,
            cutoff_len=cutoff_len,
            label_mask_prompt=label_mask_prompt,
            sft_packing=sft_packing
        )

        self.render = self.pop_dict(params, "render")

        params.pop("tokenizer")
        self.params = params

    def main_handle(self):
        dataset = load_dataset(self.path, **self.params)
        column_names = list(next(iter(dataset)).keys())
        dataset = dataset.filter(lambda x: self.data_processor.do_filter(x, self.stage))
        if not self.streaming:
            kwargs = dict(
                num_proc=1,
                load_from_cache_file=False,
                desc="Running tokenizer on dataset"
            )
        dataset = dataset.map(
            lambda x: self.data_processor.process(self.tokenizer, x, self.stage, self.render),
            batched=True,
            remove_columns=column_names,
            **kwargs
        )

        self.inst = dataset
