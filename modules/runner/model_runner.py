# -*- coding: utf-8 -*-

from .basic_runner import Task
from transformers import AutoTokenizer, AutoModelForCausalLM


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
