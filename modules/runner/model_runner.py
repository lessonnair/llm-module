# -*- coding: utf-8 -*-

from .basic_runner import Task
import math
from transformers import AutoConfig, AutoTokenizer, PretrainedConfig, PreTrainedModel
from transformers import BitsAndBytesConfig, PreTrainedTokenizerBase
from transformers.models.llama import modeling_llama as LlamaModule
from transformers.utils.versions import require_version
from modules.util.model_util import *
from modules.util.util import *
from modules.core.model.util import *

try:
    from transformers.integrations import is_deepspeed_zero3_enabled
except ImportError:  # https://github.com/huggingface/transformers/releases/tag/v4.33.1
    from transformers.deepspeed import is_deepspeed_zero3_enabled


class TokenizerLoader(Task):

    def __init__(self, config, name=None, model_path=None):
        super(TokenizerLoader, self).__init__(config, name=name)
        if model_path is None:
            self.model_path = self.get_config("pretrained_model_name_or_path")
        else:
            self.model_path = model_path
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

        if getattr(tokenizer, "eos_token", None) is None:
            tokenizer.eos_token = tokenizer.cls_token

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

    def __init__(self,
                 config,
                 model_path=None,
                 is_trainable=False,
                 ):

        super(ModelLoader, self).__init__(config)

        if model_path is None:
            self.model_path = self.get_config("pretrained_model_name_or_path")
        else:
            self.model_path = model_path
        self.is_trainable = is_trainable
        self.print_model_structure = self.get_config("print_model_structure")

        self.finetune_args = self.get_instance("finetune_args")

        params = self.get_section_params()
        for c in ("pretrained_model_name_or_path", "print_model_structure", "finetune_args"):
            if c in params:
                params.pop(c)

        self.use_gradient_checkpointing = self.pop_dict(params, "use_gradient_checkpointing", True)

        config_kwargs = {}
        for c in ("trust_remote_code", "cache_dir", "revision", "use_auth_token"):
            if c in params:
                config_kwargs[c] = params.pop(c)

        self.rope_scaling = self.pop_dict(params, "rope_scaling")
        self.model_max_length = self.pop_dict(params, "model_max_length")

        self.flash_attn = self.pop_dict(params, "flash_attn")
        self.shift_attn = self.pop_dict(params, "shift_attn")
        self.quantization_bit = self.pop_dict(params, "quantization_bit")
        self.double_quantization = self.pop_dict(params, "double_quantization", None)
        self.quantization_type = self.pop_dict(params, "quantization_type", None)
        self.checkpoint_dir = self.pop_dict(params, "checkpoint_dir")

        self.config_kwargs = config_kwargs
        self.params = params

        if len(self.proxies) > 0:
            self.config_kwargs["proxies"] = self.proxies

    def set_model_path(self, model_path):
        self.model_path = model_path

    def set_trainable(self, is_trainable):
        self.is_trainable = is_trainable

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
            from ..core.model import llama_patch
            if getattr(config, "model_type", None) == "llama":
                LlamaModule.LlamaAttention = llama_patch.LlamaFlashAttention2
                LlamaModule.LlamaModel._prepare_decoder_attention_mask = llama_patch._prepare_decoder_attention_mask
                self.logger.info("Using FlashAttention-2 for faster training and inference.")
            elif getattr(config, "model_type", None) == "qwen":
                self.logger.info("Qwen models automatically enable FlashAttention if installed.")
            else:
                self.logger.warning("Current model does not support FlashAttention-2.")
        elif self.is_trainable and self.shift_attn and getattr(config, "model_type", None) == "llama":
            from ..core.model import llama_patch
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

        if self.is_trainable:
            model = prepare_model_for_train(model=model,
                                            finetuning_args=self.finetune_args,
                                            use_gradient_checkpointing=self.use_gradient_checkpointing)
        if self.finetune_args is not None:
            model = init_adapter(model,
                                 self.finetune_args,
                                 self.is_trainable,
                                 is_mergeable,
                                 quantization_bit=self.quantization_bit,
                                 )

        if self.is_trainable:
            model = model.train()
        else:
            model = model.eval()

        # Prepare model for inference
        if not self.is_trainable:
            model.requires_grad_(False)
            # if self.quantization_bit is None:
            #     model = model.to(self.compute_dtype)

        self.inst = model

        if self.print_model_structure:
            self.logger.info("model", self.inst)
