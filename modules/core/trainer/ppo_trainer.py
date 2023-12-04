# -*- coding: utf-8 -*-

import torch
from tqdm import tqdm
from typing import List
from modules.util.custom_log import get_logger
import trl
from modules.util.checkpoint_util import *


logger = get_logger(__name__)


class PPOTrainer(trl.PPOTrainer):

    def __init__(self,
                 generation_kwargs: "GeneratingArguments",
                 **kwargs):
        trl.PPOTrainer.__init__(self, **kwargs)

        self.generation_kwargs = generation_kwargs


    def train(self, resume_from_checkpoint=None):

        for epoch, batch in tqdm(enumerate(self.dataloader)):
            query_tensors = batch["input_ids"]

            response_tensors = []
            for query in query_tensors:
                response = self.generate(query, **self.generation_kwargs)
                response_tensors.append(response.squeeze())
            batch["response"] = [self.tokenizer.decode(r.squeeze()) for r in response_tensors]

            rewards = self.get_rewards(query_tensors, response_tensors)

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






