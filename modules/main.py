# -*- coding: utf-8 -*-

from modules.task.basic_runner import Project
from config_util import TaskConfig
import sys

if __name__=="__main__":

    config_path = sys.argv[1]
    assert config_path != None and len(config_path)>0, "project config path must be set"

    config = TaskConfig(config_path)
    task = Project(config)
    task.run()



