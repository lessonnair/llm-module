# -*- coding: utf-8 -*-

from .basic_runner import Task
import peft


class PPOArguments(Task):
    def __init__(self, config, name=None):
        super(PPOArguments, self).__init__(config, name=name)
        self.params = self.get_section_params(parse_json=True)

    def main_handle(self):
        self.inst = self.params


class FinetuneArguments(Task):
    def __init__(self, config, name=None):
        super(FinetuneArguments, self).__init__(config, name=name)

        params = self.get_section_params()

        self.lora_config = self.load_lora_config(params)
        self.__dict__.update(params)
        self.checkpoint_dir = self.get_config_list("checkpoint_dir")

    def __getattr__(self, attr):
        return None

    def main_handle(self):
        self.inst = self

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


class TrainingArguments(Task):
    def __init__(self, config, name=None):
        super(TrainingArguments, self).__init__(config, name=name)
        self.params = self.get_section_params()

    def main_handle(self):
        self.inst = self.get_inst_clazz()(
            **self.params
        )


class GenerateArguments(Task):
    def __init__(self, config, name=None):
        super(GenerateArguments, self).__init__(config, name=name)
        self.params = self.get_section_params()

    def main_handle(self):
        self.inst = self.params
