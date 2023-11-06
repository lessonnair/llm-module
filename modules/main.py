# -*- coding: utf-8 -*-

from modules.runner import Project
from modules.util.config_util import TaskConfig
import os

if __name__ == "__main__":
    config_path = "../example/trainer.ini"

    assert config_path is not None and len(config_path) > 0, "project config path must be set"
    assert os.path.isfile(config_path) and os.access(config_path,
                                                     os.R_OK), f"config_path {config_path} does not exist or you don't have the permission to read"

    config = TaskConfig(config_path)
    task = Project(config)
    task.run()
