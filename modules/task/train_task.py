# -*- coding: utf-8 -*-

from .basic_runner import Task
import transformers

class Trainer(Task):

    def __init__(self, config):
        super(Trainer, self).__init__(config)

        self.trainer = transformers.Trainer(
            model=model
        )



    def main_handle(self):
        pass
