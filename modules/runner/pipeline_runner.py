# -*- coding: utf-8 -*-

from .basic_runner import Task
import sys

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
            task_inst = getattr(sys.modules["modules.runner"], task_name)(self.config)
            task_inst.run()

        self.logger.info("pipeline {} end.".format(self.project_info))