# -*- coding: utf-8 -*-

import torch
from types import MethodType
from typing import TYPE_CHECKING, List, Optional
from peft import (
    PeftModel,
    TaskType,
    LoraConfig,
    get_peft_model
)
from modules.util.constants import LAYERNORM_NAMES
from modules.util.custom_log import get_logger
from modules.util.model_util import find_all_linear_modules

logger = get_logger(__name__)

if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel


def prepare_model_for_train(
    model: "PreTrainedModel",
    finetuning_args: "FinetuningArguments",
    output_layer_name: Optional[str] = "lm_head",
    use_gradient_checkpointing: Optional[bool] = True,
    layernorm_names: Optional[List[str]] = LAYERNORM_NAMES
) -> "PreTrainedModel":

    if finetuning_args is not None and finetuning_args.upcast_layernorm:
        for name, param in model.named_parameters():
            if param.ndim == 1 and any(ln_name in name for ln_name in layernorm_names):
                param.data = param.data.to(torch.float32)

    if finetuning_args is not None and finetuning_args.neft_alpha is not None and finetuning_args.neft_alpha > 1e-6:
        input_embed: torch.nn.Embedding = model.get_input_embeddings()

        def noisy_forward(self: torch.nn.Embedding, x: torch.Tensor) -> torch.Tensor:
            embeddings = torch.nn.Embedding.forward(self, x)
            if self.training:
                dims = self.num_embeddings * self.embedding_dim
                mag_norm = finetuning_args.neft_alpha / (dims ** 0.5)
                embeddings += torch.zeros_like(embeddings).uniform_(-mag_norm, mag_norm)
            return embeddings

        input_embed.forward = MethodType(noisy_forward, input_embed)

    if use_gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    if finetuning_args is not None and finetuning_args.type != "full" and hasattr(model, output_layer_name):
        output_layer: torch.nn.Linear = getattr(model, output_layer_name)
        input_dtype = output_layer.weight.dtype

        def forward_in_fp32(self, x: torch.Tensor) -> torch.Tensor:
            return torch.nn.Linear.forward(self, x.to(input_dtype)).to(torch.float32)

        output_layer.forward = MethodType(forward_in_fp32, output_layer)

    return model


def init_adapter(
    model: "PreTrainedModel",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool,
    is_mergeable: bool,
    quantization_bit=None,
    ) -> "PreTrainedModel":
    checkpoint_dir = finetuning_args.checkpoint_dir
    if finetuning_args.type == "none" and is_trainable:
        raise ValueError("You cannot use finetuning_type=none while training.")

    if finetuning_args.type == "full" and is_trainable:
        logger.info("Fine-tuning method: Full")
        model = model.float()

    elif finetuning_args.type == "freeze":
        logger.info("Fine-tuning method: Freeze")
        num_layers = getattr(model.config, "num_layers")
        if finetuning_args.num_layer_trainable > 0:  # fine-tuning the last n layers if num_layer_trainable > 0
            trainable_layer_ids = [num_layers - k - 1 for k in range(finetuning_args.num_layer_trainable)]
        else:  # fine-tuning the first n layers if num_layer_trainable < 0
            trainable_layer_ids = [k for k in range(-finetuning_args.num_layer_trainable)]

        trainable_layers = ["{:d}.{}".format(idx, finetuning_args.name_module_trainable) for idx in trainable_layer_ids]
        for name, param in model.named_parameters():
            if not any(trainable_layer in name for trainable_layer in trainable_layers):
                param.requires_grad_(False)
            else:
                param.data = param.data.to(torch.float32)

    elif finetuning_args.type == "lora":
        logger.info("Fine-tuning method: LoRA")
        latest_checkpoint = None

        if checkpoint_dir is not None:
            if (is_trainable and finetuning_args.resume_lora_training) or (not is_mergeable):  # continually fine-tuning
                checkpoints_to_merge, latest_checkpoint = checkpoint_dir[:-1], checkpoint_dir[-1]
            else:
                checkpoints_to_merge = checkpoint_dir

            for checkpoint in checkpoints_to_merge:
                model = PeftModel.from_pretrained(model, checkpoint)
                model = model.merge_and_unload()

            if len(checkpoints_to_merge) > 0:
                logger.info("Merged {} model checkpoint(s).".format(len(checkpoints_to_merge)))

            if latest_checkpoint is not None:  # resume lora training or quantized inference
                model = PeftModel.from_pretrained(model, latest_checkpoint, is_trainable=is_trainable)

        if is_trainable and latest_checkpoint is None:  # create new lora weights while training

            model = get_peft_model(model, finetuning_args.lora_config)
            if id(model.peft_config) != id(
                    model.base_model.peft_config):  # https://github.com/huggingface/peft/issues/923
                model.base_model.peft_config = model.peft_config

    if checkpoint_dir is not None:
        logger.info("Loaded fine-tuned model from checkpoint(s): {}".format(",".join(checkpoint_dir)))

    return model
