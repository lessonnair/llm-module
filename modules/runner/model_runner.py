# -*- coding: utf-8 -*-

from .basic_runner import Task
import math
from transformers import AutoConfig, AutoTokenizer
import os
import peft
import torch
from types import MethodType
from transformers import BitsAndBytesConfig, PreTrainedTokenizerBase
from transformers.models.llama import modeling_llama as LlamaModule
from transformers.utils.versions import require_version
from modules.extras import llama_patch


class TokenizerLoader(Task):

    def __init__(self, config):
        super(TokenizerLoader, self).__init__(config)

        self.model_path = self.get_config("pretrained_model_name_or_path")
        params = self.get_section_params()
        params.pop("pretrained_model_name_or_path")
        self.params = params

        if len(self.proxies) > 0:
            self.params["proxies"] = self.proxies

    def main_handle(self):
        config = AutoConfig.from_pretrained(self.model_path, **self.params)
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            **self.params
        )
        if getattr(config, "model_type", None) == "chatglm":
            tokenizer._pad = MethodType(PreTrainedTokenizerBase._pad, tokenizer)

        if getattr(config, "model_type", None) == "qwen":
            for dtype_name, dtype in [("fp16", torch.float16), ("bf16", torch.bfloat16), ("fp32", torch.float32)]:
                setattr(config, dtype_name, getattr(config, "torch_dtype", None) == dtype)

        self.inst = tokenizer


class ModelLoader(Task):

    def __init__(self, config):
        super(ModelLoader, self).__init__(config)

        self.model_path = self.get_config("pretrained_model_name_or_path")
        self.print_model_structure = self.get_config("print_model_structure")

        params = self.get_section_params()
        for c in ("pretrained_model_name_or_path", "print_model_structure"):
            params.pop(c)

        config_kwargs = {}
        for c in ("trust_remote_code", "cache_dir", "revision", "use_auth_token"):
            if c in params:
                config_kwargs[c] = params.pop(c)

        self.is_trainable = self.pop_dict(params, "is_trainable")
        self.rope_scaling = self.pop_dict(params, "rope_scaling")
        self.model_max_length = self.pop_dict(params, "model_max_length")

        self.flash_attn = self.pop_dict(params, "flash_attn")
        self.shift_attn = self.pop_dict(params, "shift_attn")
        self.quantization_bit = self.pop_dict(params, "quantization_bit")

        self.config_kwargs = config_kwargs
        self.params = params

        if len(self.proxies) > 0:
            self.config_kwargs["proxies"] = self.proxies



    def main_handle(self):

        config = AutoConfig.from_pretrained(self.model_path, **self.config_kwargs)

        if getattr(config, "model_type", None) == "qwen":
            for dtype_name, dtype in [("fp16", torch.float16), ("bf16", torch.bfloat16), ("fp32", torch.float32)]:
                setattr(config, dtype_name, getattr(config, "torch_dtype", None) == dtype)

        # set rope scaling
        if self.rope_scaling is not None:
            # for qwen model
            if hasattr(config, "use_dynamic_ntk"):
                if self.is_trainable:
                    self.logger.warning("Qwen model does not support RoPE scaling in training")
                else:
                    setattr(config, "use_dynamic_ntk", True)
                    setattr(config, "use_logn_attn", True)
                    self.logger.info("using dynamic ntk scaling")

            # for llama and falcon models
            elif hasattr(config, "rope_scaling"):
                if self.is_trainable:
                    if self.rope_scaling == "dynamic":
                        self.logger.warning(
                            "Dynamic NTK may not work well with fine-tuning. "
                            "See: https://github.com/huggingface/transformers/pull/24653"
                        )
                    current_max_length = getattr(config, "max_position_embeddings", None)
                    if current_max_length and self.model_max_length > current_max_length:
                        scaling_factor = float(math.ceil(self.model_max_length / current_max_length))
                    else:
                        self.logger.warning("Input length is smaller than max length. Consider increase input length.")
                        scaling_factor = 1.0
                else:
                    scaling_factor = 2.0

                setattr(config, "rope_scaling", {"type": self.rope_scaling, "factor": scaling_factor})
                self.logger.info("Using {} scaling strategy and setting scaling factor to {}".format(
                    self.rope_scaling, scaling_factor
                ))
            else:
                self.logger.warning("Current model does not support RoPE scaling.")

        if self.flash_attn:
            if getattr(config, "model_type", None) == "llama":
                LlamaModule.LlamaAttention = llama_patch.LlamaFlashAttention2
                LlamaModule.LlamaModel._prepare_decoder_attention_mask = llama_patch._prepare_decoder_attention_mask
                self.logger.info("Using FlashAttention-2 for faster training and inference.")
            elif getattr(config, "model_type", None) == "qwen":
                self.logger.info("Qwen models automatically enable FlashAttention if installed.")
            else:
                self.logger.warning("Current model does not support FlashAttention-2.")
        elif self.is_trainable and self.shift_attn and getattr(config, "model_type", None) == "llama":
            LlamaModule.LlamaAttention = llama_patch.LlamaShiftShortAttention
            self.logger.warning("Using `--flash_attn` for faster training in large context length.")


        if self.is_trainable and self.shift_attn:
            if getattr(config, "model_type", None) == "llama":
                setattr(config, "group_size_ratio", 0.25)
                self.logger.info("Using shift short attention with group_size_ratio=1/4.")
            else:
                self.logger.warning("Current model does not support shift short attention.")

        is_mergeable = True
        if self.quantization_bit is not None:
            if self.quantization_bit == 8:
                require_version("bitsandbytes>=0.37.0", "To fix: pip install bitsandbytes>=0.37.0")
                self.config_kwargs["load_in_8bit"] = True
                self.config_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            elif self.quantization_bit == 4:
                require_version("bitsandbytes>=0.39.0", "To fix: pip install bitsandbytes>=0.39.0")
                self.config_kwargs["load_in_4bit"] = True
                self.config_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=self.params.get("torch_dtype"),
                    bnb_4bit_use_double_quant=self.pop_dict(self.params, "double_quantization", None),
                    bnb_4bit_quant_type=self.pop_dict(self.params, "quantization_type", None)
                )
            is_mergeable = False
            self.config_kwargs["device_map"] = {"": int(os.environ.get("LOCAL_RANK", "0"))} if self.is_trainable else "auto"
            self.logger.info("Quantizing model to {} bit.".format(self.quantization_bit))

        self.inst = getattr(self.get_inst_clazz(), "from_pretrained")(
            self.model_path,
            config=config,
            **self.params,
            **self.config_kwargs
        )

        if self.print_model_structure:
            self.logger.info("model", self.inst)


class LoraConfig(Task):

    def __init__(self, config):
        super(LoraConfig, self).__init__(config)

        target_modules = self.get_config_list("target_modules")
        params = self.get_section_params()
        params["target_modules"] = target_modules

        self.params = params

    def main_handle(self):
        self.inst = peft.LoraConfig(**self.params)


class AdapterLoader(Task):

    def __init__(self, config):
        super(AdapterLoader, self).__init__(config)

        self.model = self.get_instance("model")
        self.config_checkpoint_dir = self.get_config_list("config_checkpoint_dir")
        self.lora_config = self.get_instance("lora_config")

    def main_handle(self):
        if self.config_checkpoint_dir is not None and len(self.config_checkpoint_dir) > 0:
            for config_dir in self.config_checkpoint_dir:
                model = peft.PeftModel.from_pretrained(self.model, config_dir)
                model = model.merge_and_unload()
            self.logger.info("Loaded {} model checkpoint(s).".format(len(self.config_checkpoint_dir)))
        else:
            model = peft.get_peft_model(self.model, self.lora_config)
            if id(model.peft_config) != id(model.base_model.peft_config):
                model.base_model.peft_config = model.peft_config
        self.inst = model
