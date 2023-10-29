# -*- coding: utf-8 -*-

from ..custom_log import Logger
import sys


class Task(object):

    def __init__(self, config):
        self.config = config
        self.name = self.__class__.__name__

        self.logger = Logger(self.name)


    def get_config(self, field_name):
        return self.config.get(self.name, field_name)


    def run(self):
        self.logger.info("Task {} start ...".format(self.name))
        self.main_handle()

        self.clear()
        self.logger.info("Task {} end.".format(self.name))


    def clear(self):
        pass


class AutoTokenizerLoader(Task):

    def __init__(self, config):
        super(AutoTokenizerLoader, self).__init__(config)





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



