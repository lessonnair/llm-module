# -*- coding: utf-8 -*-

import torch
from tqdm import tqdm
from typing import List, Optional
import trl
from modules.util.checkpoint_util import *
from modules.util.custom_log import get_logger

logger = get_logger(__name__)


class PPOTrainer(trl.PPOTrainer):

    def __init__(self,
                 model_args: "ModelArguments",
                 args: "Seq2SeqTrainingArguments",
                 generating_args: "GeneratingArguments",
                 callbacks: List["TrainerCallback"],
                 **kwargs):
        trl.PPOTrainer.__init__(self, **kwargs)

        self.model_args = model_args
        self.args = args
        self.generating_args = generating_args
        self.log_callback, self.save_callback = callbacks[0], callbacks[1]

    def train(self, resume_from_checkpoint=None):
        unwrapped_model: "AutoModelForCausalLMWithValueHead" = self.accelerator.unwrap_model(self.model)

        for epoch, batch in tqdm(enumerate(self.dataloader)):
            unwrapped_model.gradient_checkpointing_disable()
            unwrapped_model.config.use_cache = True
            self.model.eval()

            query_tensors = [query for query in batch["input_ids"]]

            response_tensors = []
            for query in query_tensors:
                response = self.generate(query, **self.generating_args)
                response_tensors.append(response.squeeze())

            batch["response"] = [self.tokenizer.decode(r.squeeze()) for r in response_tensors]

            rewards = self.get_rewards(query_tensors, response_tensors, unwrapped_model)
            unwrapped_model.gradient_checkpointing_enable()
            unwrapped_model.config.use_cache = False

            self.model.train()

            # run ppo step
            stats = self.step(query_tensors, response_tensors, rewards)
            self.log_stats(stats, batch, rewards)

    def get_rewards(self,
                    queries: List[torch.Tensor],
                    responses: List[torch.Tensor],
                    unwrapped_model: "AutoModelForCausalLMWithValueHead") -> List[torch.Tensor]:
        replace_model(unwrapped_model, target="reward")
        batch = self.prepare_model_inputs(queries, responses)

        with torch.cuda.amp.autocast(dtype=self.model_args.get("compute_dtype")):  # support bf16
            _, _, values = self.model(**batch, output_hidden_states=True, return_dict=True)

        if values.size(0) != batch["input_ids"].size(0):
            values = torch.transpose(values, 0, 1)

        rewards = []
        for i in range(values.size(0)):
            end_index = batch["attention_mask"][i].nonzero()[-1]
            rewards.append(values[i, end_index].float().detach().cpu())

        replace_model(unwrapped_model, target="default")
        return rewards

    def save_model(self, output_dir: Optional[str] = None) -> None:
        if self.args.should_save:
            self._save(output_dir)
