# -*- coding: utf-8 -*-

from .basic_runner import Task
import math
from transformers import AutoConfig, AutoTokenizer, PretrainedConfig
import os
import peft
import torch
from types import MethodType
from transformers import BitsAndBytesConfig, PreTrainedTokenizerBase, PreTrainedModel
from transformers.models.llama import modeling_llama as LlamaModule
from transformers.utils.versions import require_version
from modules.util.checkpoint_util import *
from modules.util.analyze_util import *
from modules.util.util import *

try:
    from transformers.integrations import is_deepspeed_zero3_enabled
except ImportError:  # https://github.com/huggingface/transformers/releases/tag/v4.33.1
    from transformers.deepspeed import is_deepspeed_zero3_enabled


class TokenizerLoader(Task):

    def __init__(self, config, name=None):
        super(TokenizerLoader, self).__init__(config, name=name)

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

        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token

        if getattr(config, "model_type", None) == "chatglm":
            tokenizer._pad = MethodType(PreTrainedTokenizerBase._pad, tokenizer)

        if getattr(config, "model_type", None) == "qwen":
            for dtype_name, dtype in [("fp16", torch.float16), ("bf16", torch.bfloat16), ("fp32", torch.float32)]:
                setattr(config, dtype_name, getattr(config, "torch_dtype", None) == dtype)
                if "torch_dtype" in self.config_kwargs:
                    self.config_kwargs["torch_dtype"] = dtype

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
        self.double_quantization = self.pop_dict(params, "double_quantization", None)
        self.quantization_type = self.pop_dict(params, "quantization_type", None)
        self.reward_model = self.pop_dict(params, "reward_model")
        self.config_checkpoint_dir = self.pop_dict(params, "config_checkpoint_dir")

        self.type = self.pop_dict(params, "type")
        self.lora_config = self.load_lora_config(params)

        self.config_kwargs = config_kwargs
        self.params = params

        if len(self.proxies) > 0:
            self.config_kwargs["proxies"] = self.proxies

    def load_lora_config(self, params):
        lora_params = {}
        for k in list(params.keys()):
            if k.startswith("lora_config"):
                v = self.pop_dict(params, k)
                k = k.split("lora_config_")[1]
                if k == "target_modules":
                    v = [i.strip() for i in v.split(",")]
                lora_params[k] = v
        return peft.LoraConfig(**lora_params)

    def main_handle(self):

        config = AutoConfig.from_pretrained(self.model_path, **self.config_kwargs)

        if getattr(config, "model_type", None) == "qwen":
            for dtype_name, dtype in [("fp16", torch.float16), ("bf16", torch.bfloat16), ("fp32", torch.float32)]:
                setattr(config, dtype_name, getattr(config, "torch_dtype", None) == dtype)

        if "torch_dtype" in self.params:
            torch_dtype = self.params["torch_dtype"]
            self.params["torch_dtype"] = get_dtype(torch_dtype)

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

            # for LLaMA and Falcon models
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
            from modules.extras import llama_patch
            if getattr(config, "model_type", None) == "llama":
                LlamaModule.LlamaAttention = llama_patch.LlamaFlashAttention2
                LlamaModule.LlamaModel._prepare_decoder_attention_mask = llama_patch._prepare_decoder_attention_mask
                self.logger.info("Using FlashAttention-2 for faster training and inference.")
            elif getattr(config, "model_type", None) == "qwen":
                self.logger.info("Qwen models automatically enable FlashAttention if installed.")
            else:
                self.logger.warning("Current model does not support FlashAttention-2.")
        elif self.is_trainable and self.shift_attn and getattr(config, "model_type", None) == "llama":
            from modules.extras import llama_patch
            LlamaModule.LlamaAttention = llama_patch.LlamaShiftShortAttention
            self.logger.warning("Using `--flash_attn` for faster training in large context length.")

        # set shift short attention
        if self.is_trainable and self.shift_attn:
            if getattr(config, "model_type", None) == "llama":
                setattr(config, "group_size_ratio", 0.25)
                self.logger.info("Using shift short attention with group_size_ratio=1/4.")
            else:
                self.logger.warning("Current model does not support shift short attention.")

        is_mergeable = True
        if self.quantization_bit is not None:
            if is_deepspeed_zero3_enabled():
                raise ValueError("DeepSpeed ZeRO-3 is incompatible with quantization.")

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
                    bnb_4bit_use_double_quant=self.double_quantization,
                    bnb_4bit_quant_type=self.quantization_type
                )
            is_mergeable = False
            self.config_kwargs["device_map"] = {
                "": int(os.environ.get("LOCAL_RANK", "0"))} if self.is_trainable else "auto"
            self.logger.info("Quantizing model to {} bit.".format(self.quantization_bit))

        model = getattr(self.get_inst_clazz(), "from_pretrained")(
            self.model_path,
            config=config,
            **self.params,
            **self.config_kwargs
        )
        # Disable custom generate method (for Qwen and Baichuan2)
        if isinstance(model, PreTrainedModel) and "GenerationMixin" not in str(model.generate.__func__):
            model.generate = MethodType(PreTrainedModel.generate, model)

        # Fix LM head (for ChatGLM2)
        if getattr(config, "model_type", None) == "chatglm":
            setattr(model, "lm_head", model.transformer.output_layer)

        # Register auto class to save the custom code files.
        if isinstance(config, PretrainedConfig) and "AutoConfig" in getattr(config, "auto_map", {}):
            config.__class__.register_for_auto_class()
        if isinstance(model, PreTrainedModel) and "AutoModelForCausalLM" in getattr(config, "auto_map", {}):
            model.__class__.register_for_auto_class()

        if self.type == "lora":
            if self.config_checkpoint_dir is not None and len(self.config_checkpoint_dir) > 0:
                for config_dir in self.config_checkpoint_dir:
                    model = peft.PeftModel.from_pretrained(model, config_dir)
                    model = model.merge_and_unload()
                self.logger.info("Loaded {} model checkpoint(s).".format(len(self.config_checkpoint_dir)))
            else:
                model = peft.get_peft_model(model, self.lora_config)
                if id(model.peft_config) != id(model.base_model.peft_config):
                    model.base_model.peft_config = model.peft_config

        if self.is_trainable:
            model = model.train()
        else:
            model = model.eval()

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
                if getattr(model, "is_peft_model", False):
                    model.pretrained_model.load_adapter(self.reward_model, "reward")
                assert load_valuehead_params(model, self.reward_model), "Reward model is not correctly loaded."


        # Prepare model for inference
        if not self.is_trainable:
            model.requires_grad_(False)
            if self.quantization_bit is None:
                model = model.to(self.compute_dtype)

        trainable_params, all_param = count_parameters(model)

        self.logger.info("trainable params: {:d} || all params: {:d} || trainable%: {:.4f}".format(
            trainable_params, all_param, 100 * trainable_params / all_param
        ))

        if not self.is_trainable:
            self.logger.info(
                "This IS expected that the trainable params is 0 if you are using model for inference only.")

        self.inst = model

        if self.print_model_structure:
            self.logger.info("model", self.inst)


class AdapterLoader(Task):

    def __init__(self, config):
        super(AdapterLoader, self).__init__(config)

        self.model = self.get_instance("model")
        self.config_checkpoint_dir = self.get_config_list("config_checkpoint_dir")

        self.lora_config = self.load_lora_config()

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

    def load_lora_config(self):
        params = self.get_section_params()
        lora_params = {}
        for k in params:
            if k.startswith("lora_config"):
                v = params[k]
                k = k.split("lora_config_")[1]
                if k == "target_modules":
                    v = [i.strip() for i in v.split(",")]
                lora_params[k] = v
        return peft.LoraConfig(**lora_params)
