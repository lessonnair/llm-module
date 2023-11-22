# -*- coding: utf-8 -*-

from .basic_runner import Task
from modules.util.package_util import import_package
import math
import transformers
from transformers import DataCollatorForSeq2Seq, DataCollatorForLanguageModeling, DataCollatorWithPadding
from modules.util.ploting import plot_loss
from modules.util.constants import *
from modules.extras.collator import *
from modules.core.trainer import *
from modules.util.metric_util import *


class TrainingArguments(Task):

    def __init__(self, config, name=None):
        super(TrainingArguments, self).__init__(config, name=name)

        self.params = self.get_section_params()

    def main_handle(self):
        self.inst = self.get_inst_clazz()(
            **self.params
        )


class Trainer(Task):

    def __init__(self, config, name=None):
        super(Trainer, self).__init__(config, name=name)

        self.model = self.get_instance("model")
        self.tokenizer = self.get_instance("tokenizer")
        self.args = self.get_instance("args")
        self.steps = self.get_config_list("steps")
        self.resume_from_checkpoint = self.get_config("resume_from_checkpoint")
        self.plot_loss = self.get_config("plot_loss")
        self.output_dir = self.get_config("output_dir")

        self.streaming = self.get_config("streaming")
        self.split_train_val = self.get_config("split_train_val")
        self.split_train_val_val_size = self.get_config("split_train_val_val_size")
        self.split_train_val_seed = self.get_config("split_train_val_seed")
        self.split_train_val_buffer_size = self.get_config("split_train_val_buffer_size")
        self.ignore_pad_token_for_loss = self.get_config("ignore_pad_token_for_loss")
        self.predict_with_generate = self.get_config("predict_with_generate", False)

        self.data_collator = self.init_data_collator()

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

    def init_data_collator(self):
        data_collator = None
        if self.stage == "sft":
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                pad_to_multiple_of=4,  # for shift short attention
                label_pad_token_id=IGNORE_INDEX if self.ignore_pad_token_for_loss else self.tokenizer.pad_token_id
            )
        elif self.stage == "pt":
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
        elif self.stage == "ppo":
            data_collator = DPODataCollatorWithPadding(
                tokenizer=self.tokenizer,
                pad_to_multiple_of=4,
                label_pad_token_id=IGNORE_INDEX if self.ignore_pad_token_for_loss else self.tokenizer.pad_token_id
            )
        elif self.stage == "rm":
            data_collator = PairwiseDataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=4)

        return data_collator

    def main_handle(self):
        tmp = self.get_instance("dataset")
        datasets = self.split_dataset(self.get_instance("dataset"))

        self.model.train()
        if len(self.stage) == 1:
            stage = self.stage[0]
            if stage == 'eval':
                self.model.eval()
                dataset = {"eval_dataset": datasets.values[0]}

        trainer = None

        common_params = {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "args": self.args,
            "data_collator": self.data_collator,
        }
        common_params.update(datasets)

        gen_params = {}

        if self.stage in ["rm"]:
            common_params["compute_metrics"] = compute_accuracy
            trainer_clazz = PairwiseTrainer
        elif self.stage in ["sft"]:
            common_params["compute_metrics"] = ComputeMetrics(self.tokenizer) if self.predict_with_generate else None
            trainer_clazz = SFTSeq2SeqTrainer

            gen_params["eos_token_id"] = [self.tokenizer.eos_token_id] + self.tokenizer.additional_special_tokens_ids
            gen_params["pad_token_id"] = self.tokenizer.pad_token_id

        else:
            trainer_clazz = transformers.Trainer

        trainer = trainer_clazz(**common_params)

        if "train" in self.steps:
            train_result = trainer.train(resume_from_checkpoint=self.resume_from_checkpoint)
            trainer.log_metrics("train", train_result.metrics)
            trainer.save_metrics("train", train_result.metrics)
            trainer.save_state()
            trainer.save_model()
            # if trainer.is_world_process_zero() and self.plot_loss:
            #     plot_loss(self.output_dir, keys=["loss", "eval_loss"])

        if "eval" in self.steps:
            metrics = trainer.evaluate(metric_key_prefix="eval", **gen_params)
            if self.stage == "pt":
                try:
                    perplexity = math.exp(metrics["eval_loss"])
                except OverflowError:
                    perplexity = float("inf")

                metrics["perplexity"] = perplexity
            elif self.stage == "sft":
                if self.predict_with_generate:
                    metrics.pop("eval_loss", None)
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)



        if self.stage != "pt" and "predict" in self.steps:

            predictions = trainer.predict(datasets["eval_dataset"], metric_key_prefix="predict", **gen_params)
            if self.predict_with_generate:
                predictions.metrics.pop("predict_loss", None)
            trainer.log_metrics("predict", predictions.metrics)
            trainer.save_metrics("predict", predictions.metrics)
            trainer.save_predictions(predictions)
