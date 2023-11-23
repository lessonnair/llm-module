import os
import json
import torch
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union
from transformers import Trainer

if TYPE_CHECKING:
    from transformers.trainer import PredictionOutput
    from transformers.modeling_utils import PreTrainedModel

from modules.util.custom_log import get_logger

logger = get_logger(__name__)


class PairwiseTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.can_return_loss = True

    def compute_loss(self,
                     model: "PreTrainedModel",
                     inputs: Dict[str, torch.Tensor],
                     return_outputs: Optional[bool] = False
                     ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        r"""
        Computes pairwise loss. The first n examples are chosen and the last n examples are rejected.
        """
        _, _, values = model(**inputs, output_hidden_states=True, return_dict=True)
        if values.size(0) != inputs["input_ids"].size(0):
            values = torch.transpose(values, 0, 1)

        # split the inputs and rewards into two parts, chosen and rejected
        batch_size = inputs["input_ids"].size(0) // 2
        chosen_input_ids, rejected_input_ids = inputs["input_ids"][:batch_size], inputs["input_ids"][batch_size:]
        chosen_attn_mask, rejected_attn_mask = (
            inputs["attention_mask"][:batch_size], inputs["attention_mask"][batch_size:]
        )
        chosen_rewards, rejected_rewards = values[:batch_size], values[batch_size:]
        chosen_scores, rejected_scores = [], []

        loss = 0
        for i in range(batch_size):
            chosen_length = chosen_attn_mask[i].nonzero()[-1] + 1
            rejected_length = rejected_attn_mask[i].nonzero()[-1] + 1
            check_divergence = (chosen_input_ids[i] != rejected_input_ids[i]).nonzero()

            if len(check_divergence) == 0:
                end_index = chosen_length
                div_index = end_index - 1
            else:
                end_index = max(chosen_length, rejected_length)
                div_index = check_divergence[0]

            # assert div_index > 0
            chosen_trunc_rewards = chosen_rewards[i, div_index: end_index]
            rejected_trunc_rewards = rejected_rewards[i, div_index: end_index]
            if return_outputs:
                chosen_scores.append(chosen_rewards[i, chosen_length - 1])
                rejected_scores.append(rejected_rewards[i, rejected_length - 1])
            loss += -torch.nn.functional.logsigmoid(chosen_trunc_rewards - rejected_trunc_rewards).mean()

        loss = loss / batch_size
        if return_outputs:
            chosen_scores, rejected_scores = torch.stack(chosen_scores), torch.stack(rejected_scores)
            return loss, [loss, chosen_scores, rejected_scores]

        return loss

    def save_prediction(self, predict_results: "PredictionOutput"):
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info(f"Saving prediction results to {output_prediction_file}")

        chosen_scores, rejected_scores = predict_results.predictions

        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            res: List[str] = []
            for c_score, r_score in zip(chosen_scores, rejected_scores):
                res.append(json.dumps({"chosen": round(float(c_score), 2), "rejected": round(float(r_score), 2)}))
            writer.write("\n".join(res))
