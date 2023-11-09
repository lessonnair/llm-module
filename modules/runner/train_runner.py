# -*- coding: utf-8 -*-

from .basic_runner import Task
from modules.util.package_util import import_package
import math
import transformers
from modules.util.ploting import plot_loss


class TrainingArguments(Task):

    def __init__(self, config):
        super(TrainingArguments, self).__init__(config)

        self.params = self.get_section_params()

    def main_handle(self):
        self.inst = self.get_inst_clazz()(
            **self.params
        )


class DataCollator(Task):

    def __init__(self, config):
        super(DataCollator, self).__init__(config)

        self.tokenizer = self.get_instance("tokenizer")

        params = self.get_section_params()
        params.pop("tokenizer")
        self.params = params

    def main_handle(self):
        self.inst = self.get_inst_clazz()(
            tokenizer=self.tokenizer,
            **self.params
        )


class Trainer(Task):

    def __init__(self, config):
        super(Trainer, self).__init__(config)

        self.model = self.get_instance("model")
        self.tokenizer = self.get_instance("tokenizer")
        self.args = self.get_instance("args")
        self.data_collator = self.get_instance("data_collator")
        self.stage = self.get_config_list("stage")
        self.resume_from_checkpoint = self.get_config("resume_from_checkpoint")
        self.plot_loss = self.get_config("plot_loss")
        self.output_dir = self.get_config("output_dir")

        self.streaming = self.get_config("streaming")
        self.split_train_val = self.get_config("split_train_val")
        self.split_train_val_val_size = self.get_config("split_train_val_val_size")
        self.split_train_val_seed = self.get_config("split_train_val_seed")
        self.split_train_val_buffer_size = self.get_config("split_train_val_buffer_size")

    def split_dataset(self, dataset):
        res = {}
        if self.split_train_val:
            if self.streaming:
                val_set = dataset.take(self.split_train_val_val_size)
                train_set = dataset.skip(self.split_train_val_val_size)

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
        return res

    def main_handle(self):
        tmp = self.get_instance("dataset")
        datasets = self.split_dataset(self.get_instance("dataset"))

        self.model.train()
        if len(self.stage) == 1:
            stage = self.stage[0]
            if stage == 'eval':
                self.model.eval()
                dataset = {"eval_dataset": datasets.values[0]}

        trainer = transformers.Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=self.args,
            data_collator=self.data_collator,
            **datasets
        )

        if "train" in self.stage:
            train_result = trainer.train(resume_from_checkpoint=self.resume_from_checkpoint)
            trainer.log_metrics("train", train_result.metrics)
            trainer.save_metrics("train", train_result.metrics)
            trainer.save_state()
            trainer.save_model()
            # if trainer.is_world_process_zero() and self.plot_loss:
            #     plot_loss(self.output_dir, keys=["loss", "eval_loss"])

        if "eval" in self.stage:
            metrics = trainer.evaluate(metric_key_prefix="eval")
            try:
                perplexity = math.exp(metrics["eval_loss"])
            except OverflowError:
                perplexity = float("inf")

            metrics["perplexity"] = perplexity
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)
