import math
from tqdm import tqdm
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
import torch
from transformers import GenerationConfig, Trainer, TrainerState, TrainerControl
import trl
from trl.core import PPODecorators, logprobs_from_logits
from modules.util.custom_log import get_logger
from modules.util.analyze_util import count_parameters
from modules.util.metric_util import AverageMeter, get_logits_processor
from modules.util.checkpoint_util import *

logger = get_logger(__name__)


class PPOTrainer(trl.PPOTrainer, Trainer):

    def __init__(self,
                 model_args: "ModelArguments",
                 args: "Seq2SeqTrainingArguments",
                 generating_args: "GeneratingArguments",
                 callbacks: List["TrainerCallback"],
                 **kwargs):
        trl.PPOTrainer.__init__(self, **kwargs)
        if getattr(self.accelerator.state, "deepspeed_plugin", None) is not None:
            raise ValueError("PPOTrainer is incompatible with DeepSpeed.")

        self.args = args
        self.model_args = model_args
        self.generation_config = GenerationConfig(
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=[self.tokenizer.eos_token_id] + self.tokenizer.additional_special_tokens_ids,
            **generating_args.to_dict()
        )
        self.state = TrainerState()
        self.control = TrainerControl()
        self.log_callback, self.save_callback = callbacks[0], callbacks[1]

    def ppo_train(self):
        total_train_batch_size = (
                self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps * self.args.world_size
        )
        len_dataloader = len(self.dataloader)
        num_examples = len(self.dataset)
        num_train_epochs = self.args.num_train_epochs
        max_steps = math.ceil(num_train_epochs * len_dataloader)

        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        if self.is_world_process_zero():
            logger.info("***** Running training *****")
            logger.info(f"  Num examples = {num_examples}")
            logger.info(f"  Num Epochs = {num_train_epochs}")
            logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}")
            logger.info(
                f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
            logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
            logger.info(f"  Total optimization steps = {max_steps}")
            logger.info(f"  Number of trainable parameters = {count_parameters(self.model)[0]}")

        unwrapped_model: "AutoModelForCausalLMWithValueHead" = self.accelerator.unwrap_model(self.model)
        data_iter = iter(self.dataloader)
        steps_trained = 0
        loss_meter = AverageMeter()
        reward_meter = AverageMeter()
        self.log_callback.on_train_begin(self.args, self.state, self.control)

        for step in tqdm(range(max_steps), disable=not self.is_local_process_zero()):
            batch = next(data_iter)
            steps_trained += 1

            unwrapped_model.gradient_checkpointing_disable()
            unwrapped_model.config.use_cache = True
            self.model.eval()

            queries, responses = self.get_inputs(batch)
            self.tokenizer.padding_side = "right"
            rewards = self.get_rewards(queries, responses, unwrapped_model)

            unwrapped_model.gradient_checkpointing_enable()
            unwrapped_model.config.use_cache = False
            self.model.train()

            stats = self.step(queries, responses, rewards)
            self.tokenizer.padding_side = "left"
            loss_meter.update(float(stats["ppo/loss/total"]), n=len(rewards))
            reward_meter.update(torch.stack(rewards).mean().item(), n=len(rewards))

            if self.config.log_with is not None:
                try:
                    batch["query"] = self.tokenizer.batch_decode(queries, skip_special_tokens=True)
                    batch["response"] = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
                    self.log_stats(stats, batch, rewards)
                except:
                    logger.warning("Failed to save stats due to unknown errors.")

            self.state.global_step += 1
            self.log_callback.on_step_end(self.args, self.state, self.control)

            if self.is_local_process_zero() and (step + 1) % self.args.logging_steps == 0:
                logs = dict(
                    loss=round(loss_meter.avg, 4),
                    reward=round(reward_meter.avg, 4),
                    learning_rate=stats["ppo/learning_rate"],
                    epoch=round(step / len_dataloader, 2)
                )
                tqdm.write(str(logs))
                logs["step"] = step
                self.state.log_history.append(logs)
                self.log_callback.on_log(self.args, self.state, self.control)
                loss_meter.reset()
                reward_meter.reset()

            if (step + 1) % self.args.save_steps == 0:  # save checkpoint
                self.save_model(os.path.join(
                    self.args.output_dir, "{}-{}".format(PREFIX_CHECKPOINT_DIR, self.state.global_step)
                ))
                self.save_callback.on_save(
                    self.args, self.state, self.control, model=self.accelerator.unwrap_model(self.model)
                )

            if self.control.should_epoch_stop or self.control.should_training_stop:
                break

            if steps_trained == len_dataloader:
                data_iter = iter(self.dataloader)
                steps_trained = 0

        self.log_callback.on_train_end(self.args, self.state, self.control)
        self.save_callback.on_train_end(
            self.args, self.state, self.control, model=self.accelerator.unwrap_model(self.model)
        )

    @torch.no_grad()
    def get_inputs(self, batch: Dict[str, torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        if self.model_args.upcast_layernorm:
            layernorm_params = dump_layernorm(self.model)

        unwrapped_model: "AutoModelForCausalLMWithValueHead" = self.accelerator.unwrap_model(self.model)
        response: torch.Tensor = unwrapped_model.generate(
            generation_config=self.generation_config,
            logits_processor=get_logits_processor(),
            **batch
        )

        if self.model_args.upcast_layernorm:
            restore_layernorm(self.model, layernorm_params)

        query = batch["input_ids"].detach().cpu()
        response = response[:, batch["input_ids"].size(-1):].detach().cpu()
        queries, responses = [], []
        for i in range(len(query)):
            query_length = (query[i] != self.tokenizer.pad_token_id).nonzero()[0]
            response_index = (response[i] != self.tokenizer.pad_token_id).nonzero()

            if response_index == 0:
                response_length = 1
            elif self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                response_length = response_index[-1] + 2
            else:
                response_length = response_index[-1] + 1

            queries.append(query[i, query_length:])
            responses.append(response[i, :response_length])

        return queries, responses

    def get_rewards(self,
                    queries: List[torch.Tensor],
                    responses: List[torch.Tensor],
                    unwrapped_model: "AutoModelForCausalLMWithValueHead") -> List[torch.Tensor]:
        replace_model(unwrapped_model, target="reward")
        batch = self.prepare_model_inputs(queries, responses)

        with torch.cuda.amp.autocast(dtype=self.model_args.compute_dtype):  # support bf16
            _, _, values = self.model(**batch, output_hidden_states=True, return_dict=True)

        if values.size(0) != batch["input_ids"].size(0):
            values = torch.transpose(values, 0, 1)

        rewards = []
        for i in range(values.size(0)):
            end_index = batch["attention_mask"][i].nonzero()[-1]
            rewards.append(values[i: end_index].float().detach().cpu())

        replace_model(unwrapped_model, target="default")
        return rewards

    @PPODecorators.empty_device_cache()
    def batched_forward_pass(
            self,
            model: "AutoModelForCausalLMWithValueHead",
            queries: torch.Tensor,
            responses: torch.Tensor,
            model_inputs: dict,
            return_logits: bool = False,
            response_masks: Optional[torch.Tensor] = None,
    ):
        r"""
        Calculates model outputs in multiple batches.

        Subclass and override to inject custom behavior.
        """
        batch_size = len(queries)
        mini_batch_size = self.config.mini_batch_size
        all_logprobs = []
        all_logits = []
        all_masks = []
        all_values = []

        for i in range(math.ceil(batch_size / mini_batch_size)):
            input_kwargs = {key: value[i * mini_batch_size: (i + 1) * mini_batch_size] for key, value in
                            model_inputs.items()}
            query_batch = queries[i * mini_batch_size: (i + 1) * mini_batch_size]
            response_batch = responses[i * mini_batch_size: (i + 1) * mini_batch_size]
            if response_masks is not None:
                response_masks_batch = response_masks[i * mini_batch_size: (i + 1) * mini_batch_size]
            input_ids = input_kwargs["input_ids"]
            attention_mask = input_kwargs["attention_mask"]

            with torch.cuda.amp.autocast(dtype=self.model_args.compute_dtype):  # support bf16
                logits, _, values = model(**input_kwargs)

            if values.size(0) != input_ids.size(0):  # adapt to chatglm2
                values = torch.transpose(values, 0, 1)

            logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
            masks = torch.zeros_like(attention_mask)
            masks[:, :-1] = attention_mask[:, 1:]

            for j in range(len(query_batch)):
                start = len(query_batch[j]) - 1
                if attention_mask[j, 0] == 0:
                    start += attention_mask[j, :].nonzero()[0]
                end = start + len(response_batch[j])

                if response_masks is not None:
                    response_masks_batch = torch.cat(
                        (torch.zeros_like(query_batch[j]), response_masks_batch[j])
                    )[1:]

                masks[j, : start] = 0
                masks[j, end:] = 0
                if response_masks is not None:
                    masks[j, start:end] = masks[j, start:end] * response_masks_batch[j][start:end]

            if return_logits:
                all_logits.append(logits)
            else:
                del logits

            all_values.append(values)
            all_logprobs.append(logprobs)
            all_masks.append(masks)

        return (
            torch.cat(all_logprobs),
            torch.cat(all_logits)[:, :-1] if return_logits else None,
            torch.cat(all_values)[:, :-1],
            torch.cat(all_masks)[:, :-1],
        )

    def save_model(self, output_dir: Optional[str] = None) -> None:
        r"""
        Saves model checkpoint.

        Subclass and override to inject custom behavior.
        """
        if self.args.should_save:
            self._save(output_dir)
