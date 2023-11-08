# -*- coding: utf-8 -*-

from .basic_runner import Task
from transformers import AutoTokenizer, AutoModelForCausalLM
import peft


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
        self.inst = AutoTokenizer.from_pretrained(
            self.model_path,
            **self.params
        )


class ModelLoader(Task):

    def __init__(self, config):
        super(ModelLoader, self).__init__(config)

        self.model_path = self.get_config("pretrained_model_name_or_path")
        params = self.get_section_params()
        params.pop("pretrained_model_name_or_path")
        self.params = params

        if len(self.proxies) > 0:
            self.params["proxies"] = self.proxies

    def main_handle(self):
        self.inst = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            **self.params
        )


class LoraConfig(Task):

    def __init__(self, config):
        super(LoraConfig, self).__init__(config)
        self.params = self.get_section_params()

    def main_handle(self):
        self.inst = peft.LoraConfig(**self.params)


class AdapterLoader(Task):

    def __int__(self, config):
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
