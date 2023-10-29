
import logging

import torch
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

class LLMModel:

    def __init__(self, pretrained_model_name_or_path: str):
        self.logger = logging.getLogger(__name__)

        self.load_model(pretrained_model_name_or_path)
        self.tokenizer = self.get_tokenizer(pretrained_model_name_or_path)


    def load_model(self, pretrained_model_name_or_path: str) -> AutoModelForCausalLM:
        self.logger.info(f"Loading model for {pretrained_model_name_or_path}")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)

        self.model_hidden_size = config.hidden_size
        self.model = model


    def get_tokenizer(self, pretrained_tokenizer_name_or_path: str)  -> PreTrainedTokenizer:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer_name_or_path)
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer


    def fit(self,
            local_output_dir: str,
            epochs: int,
            per_device_train_batch_size: int,
            per_device_eval_batch_size: int,
            lr: float,
            weight_decay=1,
            ):

        training_args = TrainingArguments(
            output_dir=local_output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            learning_rate=lr,
            num_train_epochs=epochs
        )

        dataset = get_dataset(model_args, data_args)
        model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args, training_args.do_train, stage="pt")
        dataset = preprocess_dataset(dataset, tokenizer, data_args, training_args, stage="pt")
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            tokenizer=self.tokenize,
            data_collator=data_collator,
            callbacks=callbacks,
            **split_dataset(dataset, data_args, training_args)
        )
