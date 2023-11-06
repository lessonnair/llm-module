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
        self.inst = transformers.TrainingArguments(
            **self.params
        )


class DataCollator(Task):

    def __init__(self, config):
        super(DataCollator, self).__init__(config)

        self.module = self.get_config("class")
        self.tokenizer = self.get_instance("TokenizerLoader")

        params = self.get_section_params()
        params.pop("class")
        params.pop("tokenizer")
        self.params = params

    def main_handle(self):
        klass = import_package(self.module)
        self.inst = klass(
            tokenizer=self.tokenizer,
            **self.params
        )


class Trainer(Task):

    def __init__(self, config):
        super(Trainer, self).__init__(config)

        self.model = self.get_instance(self.get_config("model"))
        self.tokenizer = self.get_instance(self.get_config("tokenizer"))
        self.args = self.get_instance(self.get_config("args"))
        self.data_collator = self.get_instance(self.get_config("data_collator"))
        self.dataset = self.get_instance(self.get_config("dataset"))
        self.stage = self.get_config_list("stage")
        self.resume_from_checkpoint = self.get_config("resume_from_checkpoint")
        self.plot_loss = self.get_config("plot_loss")
        self.output_dir = self.get_config("output_dir")

    def main_handle(self):

        if len(self.stage) == 1:
            stage = self.stage[0]
            if stage == 'eval':
                self.dataset = {"eval_dataset": self.dataset.values[0]}

        trainer = transformers.Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=self.args,
            data_collator=self.data_collator,
            **self.dataset
        )

        if "train" in self.stage:
            train_result = trainer.train(resume_from_checkpoint=self.resume_from_checkpoint)
            trainer.log_metrics("train", train_result.metrics)
            trainer.save_metrics("train", train_result.metrics)
            trainer.save_state()
            trainer.save_model()
            if trainer.is_world_process_zero() and self.plot_loss:
                plot_loss(self.output_dir, keys=["loss", "eval_loss"])

        if "eval" in self.stage:
            metrics = trainer.evaluate(metric_key_prefix="eval")
            try:
                perplexity = math.exp(metrics["eval_loss"])
            except OverflowError:
                perplexity = float("inf")

            metrics["perplexity"] = perplexity
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)
