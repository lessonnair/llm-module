# -*- coding: utf-8 -*-

from .basic_runner import Task
import math
import transformers
from transformers import DataCollatorForLanguageModeling
from transformers.utils.versions import require_version
from modules.util.ploting import plot_loss
from modules.core.data.collator import *
from modules.core.trainer import *
from modules.util.analyze_util import *
from modules.util.metric_util import *
from modules.util.util import *
from modules.extras.callbacks import *
from torch.optim import AdamW
from transformers.optimization import get_scheduler


class Trainer(Task):

    def __init__(self, config, name=None):
        super(Trainer, self).__init__(config, name=name)

        self.tokenizer = self.get_instance("tokenizer")
        self.args = self.get_instance("args")
        self.ppo_args = self.get_instance("ppo_args")
        self.generate_args = self.get_instance("generate_args")
        self.stage = self.get_config("stage")
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

        self.args.predict_with_generate = self.predict_with_generate
        self.data_collator = self.init_data_collator()
        self.model = self.init_model()

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
        elif self.stage in ("ppo"):
            data_collator = PPODataCollatorWithPadding(
                tokenizer=self.tokenizer,
                pad_to_multiple_of=4,
                label_pad_token_id=IGNORE_INDEX if self.ignore_pad_token_for_loss else self.tokenizer.pad_token_id
            )
        elif self.stage in ("dpo"):
            data_collator = DPODataCollatorWithPadding(self.tokenizer)
        elif self.stage == "rm":
            data_collator = PairwiseDataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=4)

        return data_collator

    def init_model(self):
        model = self.get_instance("model", is_trainable=True)

        if self.stage in ("rm", "ppo"):
            from trl import AutoModelForCausalLMWithValueHead
            require_version("trl>=0.7.1", "To fix: pip install trl>=0.7.1")
            model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
            model._keys_to_ignore_on_save = None

            # load valuehead weights to evaluate reward model
            if self.stage == "rm" and self.checkpoint_dir is not None:
                self.logger.warning("Only the last checkpoint containing valuehead will be loaded.")
                if load_valuehead_params(model, self.checkpoint_dir):
                    model.v_head.load_state_dict({
                        "summary.weight": getattr(model, "reward_head_weight"),
                        "summary.bias": getattr(model, "reward_head_bias")
                    })
            # load reward model
            if self.stage == "ppo":
                reward_model = self.get_config("reward_model")
                assert reward_model is not None and len(reward_model) > 0, "reward_model must be set since stage is ppo"
                if getattr(model, "is_peft_model", False):
                    model.pretrained_model.load_adapter(reward_model, "reward")
                assert load_valuehead_params(model, reward_model), "Reward model is not correctly loaded."

        trainable_params, all_param = count_parameters(model)

        self.logger.info("trainable params: {:d} || all params: {:d} || trainable%: {:.4f}".format(
            trainable_params, all_param, 100 * trainable_params / all_param
        ))

        return model

    def main_handle(self):

        dataset = self.new_instance("dataset", stage=self.stage)

        if self.stage != "ppo":
            datasets = self.split_dataset(dataset)
        else:
            datasets = {"dataset": dataset}

        self.model.train()
        if len(self.stage) == 1:
            stage = self.stage[0]
            if stage == 'eval':
                self.model.eval()
                dataset = {"eval_dataset": datasets.values[0]}

        common_params = {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "args": self.args,
            "data_collator": self.data_collator,
            "callbacks": [LogCallback(), SavePeftModelCallback()]
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

        elif self.stage in ["dpo"]:
            trainer_clazz = DPOTrainer

        elif self.stage in ["ppo"]:
            from trl import PPOConfig
            ppo_config_params = self.ppo_args
            ppo_config_params["learning_rate"] = self.args.learning_rate
            ppo_config_params[
                "batch_size"] = self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps
            ppo_config_params["gradient_accumulation_steps"] = self.args.gradient_accumulation_steps
            ppo_config_params["ppo_epochs"] = 1
            ppo_config_params["max_grad_norm"] = self.args.max_grad_norm
            ppo_config_params["seed"] = self.args.seed
            ppo_config_params["remove_unused_columns"] = False

            ppo_config = PPOConfig(**ppo_config_params)
            optimizer = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.learning_rate)
            total_train_batch_size = (
                    self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps * self.args.world_size
            )
            num_training_steps = self.args.num_train_epochs * math.ceil(len(datasets) / total_train_batch_size)
            lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps
            )
            model_args = {
                "upcast_layernorm": True,
                "compute_dtype": get_dtype("fp16")
            }

            common_params.update({
                "model_args": model_args,
                "generating_args": self.generate_args,
                "config": ppo_config,
                "ref_model": None,
                "optimizer": optimizer,
                "lr_scheduler": lr_scheduler
            })
            trainer_clazz = PPOTrainer

        else:
            trainer_clazz = transformers.Trainer

        trainer = trainer_clazz(**common_params)

        if "train" in self.steps:
            if self.stage == "ppo":
                trainer.train(resume_from_checkpoint=self.resume_from_checkpoint)
                trainer.save_model()
                trainer.save_state()  # must be called after save_model to have a folder
                if trainer.is_world_process_zero() and self.plot_loss:
                    plot_loss(self.args.output_dir, keys=["loss", "reward"])
            else:
                train_result = trainer.train(resume_from_checkpoint=self.resume_from_checkpoint)
                trainer.log_metrics("train", train_result.metrics)
                trainer.save_metrics("train", train_result.metrics)
                trainer.save_state()
                trainer.save_model()
                if trainer.is_world_process_zero() and self.plot_loss:
                    plot_loss(self.args.output_dir, keys=["loss", "eval_loss"])

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

        if self.stage not in ["pt", "dpo"] and "predict" in self.steps:
            predictions = trainer.predict(datasets["eval_dataset"], metric_key_prefix="predict", **gen_params)
            if self.predict_with_generate:
                predictions.metrics.pop("predict_loss", None)
            trainer.log_metrics("predict", predictions.metrics)
            trainer.save_metrics("predict", predictions.metrics)
            trainer.save_predictions(predictions)
