# -*- coding: utf-8 -*-

from modules.util.custom_log import Logger
from modules.util.pool import InstancePool
import json
import re
import sys


class Task(object):

    def __init__(self, config, name=None):
        self.config = config

        if name is None or len(name) == 0:
            self.name = self.__class__.__name__
        else:
            self.name = name

        self.inst = None
        self.instance_pool = InstancePool()
        self.logger = Logger(self.name)

        self.proxies = self.load_proxies()

    def load_proxies(self):
        res = {}
        proxies = self.config.get("Project", "proxies", fallback={})
        if proxies is not None and len(proxies) > 0:
            res = json.loads(proxies)
        return res

    def get_config(self, field_name):
        return self.config.get(self.name, field_name)

    def get_config_list(self, field_name):
        vs = self.config.get(self.name, field_name)
        return [s.strip() for s in vs.split(",")]

    def get_section_params(self):
        return self.config.get_section_kvs(self.name)

    def get_instance(self, key):
        inst = self.instance_pool.get(key)
        if inst is not None:
            return inst
        else:
            if re.match("^DatasetLoader_[0-9]+$", key):
                class_name = key.split("_")[0]
                task_inst = getattr(sys.modules["modules.runner"], class_name)(self.config, key)
            else:
                task_inst = getattr(sys.modules["modules.runner"], key)(self.config)
            task_inst.run()

            return task_inst.inst

    def run(self):
        self.logger.info("Task {} start ...".format(self.name))
        self.main_handle()

        if self.inst is not None:
            self.instance_pool.put(self.name, self.inst)

        self.clear()
        self.logger.info("Task {} end.".format(self.name))

    def clear(self):
        pass
