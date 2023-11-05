# -*- coding: utf-8 -*-

from ..custom_log import Logger
from ..pool import InstancePool
import importlib
import json
from ..package_util import import_package
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys


class Task(object):

    def __init__(self, config):
        self.config = config
        self.name = self.__class__.__name__
        self.inst = None
        self.instance_pool = InstancePool()
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

    def get_instance(self, key):
        inst = self.instance_pool.get(key)
        if inst != None:
            return inst
        else:
            task_inst = getattr(sys.modules[__name__], key)(self.config)
            task_inst.run()
            return task_inst.inst


    def run(self):
        self.logger.info("Task {} start ...".format(self.name))
        self.main_handle()

        if self.inst != None:
            self.instance_pool.put(self.name, self.inst)

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


class TrainingArguments(Task):

    def __init__(self, config):
        super(TrainingArguments, self).__init__(config)
        self.params = self.get_section_params()

    def main_handle(self):
        self.inst = transformers.TrainingArguments(
            **self.params
        )


class DataCollator(Task):

    def __init__(self, config):
        super(DataCollator, self).__init__(config)

        self.module = self.get_config("class")
        self.tokenizer = self.get_instance("TokenizerLoader")

        params = self.get_section_params()
        params.pop("class")
        params.pop("tokenizer")
        self.params = params

    def main_handle(self):
        klass = import_package(self.module)
        self.inst = klass(
            tokenizer=self.tokenizer,
            **self.params
        )



class Trainer(Task):

    def __init__(self, config):
        super(Trainer, self).__init__(config)

        self.model = self.get_instance(self.get_config("model"))
        self.tokenizer = self.get_instance(self.get_config("tokenizer"))
        self.args = self.get_instance(self.get_config("args"))
        self.data_collator = self.get_instance(self.get_config("data_collator"))


    def main_handle(self):
        self.inst = transformers.Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=self.args,
            data_collator=self.data_collator
        )
        # args = training_args,
        # data_collator = data_collator,
        # callbacks = callbacks,
        # ** split_dataset(dataset, data_args, training_args)



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




