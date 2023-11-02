# -*- coding: utf-8 -*-

from ..custom_log import Logger
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys


class Task(object):

    def __init__(self, config):
        self.config = config
        self.name = self.__class__.__name__
        self.logger = Logger(self.name)

        self.proxies = self.load_proxies()


    def load_proxies(self):
        res = {}
        proxies = self.config.get("Project", "proxies", fallback={})
        if proxies != None and len(proxies)>0:
            res = json.loads(proxies)
        return res


    def get_config(self, field_name):
        return self.config.get(self.name, field_name)

    def get_section_params(self):
        return self.config.get_section_kvs(self.name)

    def run(self):
        self.logger.info("Task {} start ...".format(self.name))
        self.main_handle()

        self.clear()
        self.logger.info("Task {} end.".format(self.name))

    def clear(self):
        pass


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


class Project(Task):
    def __init__(self, config):
        super(Project, self).__init__(config)

        self.pipeline = self.config.parse_data_path_list("Project", "pipeline")

        self.project_info = "{}_{} by {}".format(
            self.get_config("name"), self.get_config("version"),
            self.get_config("user")
        )
        self.logger.info(f"**************** {self.project_info} ****************")

    def run(self):
        self.logger.info("{} start ...".format(self.project_info))

        for task_name in self.pipeline:
            task_inst = getattr(sys.modules[__name__], task_name)(self.config)
            task_inst.run()

        self.logger.info("pipeline {} end.".format(self.project_info))



