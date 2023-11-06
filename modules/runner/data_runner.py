# -*- coding: utf-8 -*-

from .basic_runner import Task
from datasets import load_dataset


class DatasetLoader(Task):

    def __init__(self, config, name=None):
        super(DatasetLoader, self).__init__(config, name=name)

        self.split_train_val = self.get_config("split_train_val")
        self.split_train_val_val_size = self.get_config("split_train_val_val_size")
        self.split_train_val_seed = self.get_config("split_train_val_seed")
        self.streaming = self.get_config("streaming")
        self.split_train_val_buffer_size = self.get_config("split_train_val_buffer_size")

        params = self.get_section_params()

        for c in ("split_train_val", "split_train_val_val_size", "split_train_val_seed", "split_train_val_buffer_size"):
            params.pop(c)

        self.params = params

    def main_handle(self):
        dataset = load_dataset(**self.params)

        res = {}

        if self.split_train_val:
            if self.streaming:
                val_set = dataset.take(self.split_train_val_val_size)
                train_set = dataset.skip(self.split_train_val_val_size)
                dataset = dataset.shuffle(buffer_size=self.split_train_val_buffer_size,
                                          seed=self.split_train_val_seed)
                res = {
                    "train_dataset": train_set, "eval_dataset": val_set
                }
            else:
                dataset = dataset.train_test_split(test_size=self.split_train_val_val_size,
                                                   seed=self.split_train_val_seed)
                res = {
                    "train_dataset": dataset["train"], "eval_dataset": dataset["test"]
                }
        else:
            if self.streaming:
                dataset = dataset.shuffle(buffer_size=self.split_train_val_buffer_size,
                                          seed=self.split_train_val_seed)
            res = {"train_dataset": dataset}

        self.inst = res
