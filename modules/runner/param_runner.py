# -*- coding: utf-8 -*-

from .basic_runner import Task


class PPOArguments(Task):
    def __init__(self, config, name=None):
        super(PPOArguments, self).__init__(config, name=name)
        self.params = self.get_section_params(parse_json=True)

    def main_handle(self):
        self.inst = self.params


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
        self.inst = self.get_inst_clazz()(
            **self.params
        )