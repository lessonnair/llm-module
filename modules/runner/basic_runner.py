# -*- coding: utf-8 -*-

from modules.util.custom_log import Logger
from modules.util.package_util import import_package
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
        self.checkpoint_dir = self.config.get("Project", "checkpoint_dir", fallback=None)

    def load_proxies(self):
        res = {}
        proxies = self.config.get("Project", "proxies", fallback={})
        if proxies is not None and len(proxies) > 0:
            res = json.loads(proxies)
        return res

    def get_config(self, field_name, default=None):
        return self.config.parse_value(self.config.get(self.name, field_name, fallback=default), empty_to_none=True)

    def get_config_list(self, field_name, default=None):
        vs = self.config.get(self.name, field_name)
        if vs is not None and len(vs) > 0:
            return [s.strip() for s in vs.split(",")]
        else:
            return default

    def get_section_params(self, parse_json=False):
        params = self.config.get_section_kvs(self.name, empty_to_none=True, parse_json=parse_json)
        if "class" in params:
            params.pop("class")
        return params

    def get_instance(self, key, **kwargs):
        key = self.get_config(key)
        if key is None or len(key) <= 0:
            return None
        inst = self.instance_pool.get(key)
        if inst is not None:
            return inst
        else:
            if re.match("^\\w+_[0-9]+$", key):
                class_name = key.split("_")[0]
                task_inst = getattr(sys.modules["modules.runner"], class_name)(self.config, key, **kwargs)
            else:
                task_inst = getattr(sys.modules["modules.runner"], key)(self.config, **kwargs)
            task_inst.run()

            return task_inst.inst

    def new_instance(self, key, **kwargs):
        key = self.get_config(key)
        if key is None or len(key) <= 0:
            return None

        if re.match("^\\w+_[0-9]+$", key):
            class_name = key.split("_")[0]
            task_inst = getattr(sys.modules["modules.runner"], class_name)(self.config, key, **kwargs)
        else:
            task_inst = getattr(sys.modules["modules.runner"], key)(self.config, **kwargs)
        task_inst.run()

        return task_inst.inst

    def clear_cache(self):
        self.instance_pool.clear()


    def new_instance_task(self, key, **kwargs):
        key = self.get_config(key)
        if key is None or len(key) <= 0:
            return None

        if re.match("^\\w+_[0-9]+$", key):
            class_name = key.split("_")[0]
            task_inst = getattr(sys.modules["modules.runner"], class_name)(self.config, key, **kwargs)
        else:
            task_inst = getattr(sys.modules["modules.runner"], key)(self.config, **kwargs)

        return task_inst

    def pop_dict(self, params, k, default=None):
        res = default
        if k in params:
            res = params.pop(k)
        return res

    def run(self):
        self.logger.info("Task {} start ...".format(self.name))
        self.main_handle()

        if self.inst is not None:
            self.instance_pool.put(self.name, self.inst)

        self.clear()
        self.logger.info("Task {} end.".format(self.name))

    def clear(self):
        pass

    def get_inst_clazz(self):
        clazz = self.get_config("class", default=None)
        if clazz is not None:
            return import_package(clazz)
