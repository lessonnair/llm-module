# -*- coding: utf-8 -*-

import os

IGNORE_INDEX = -100

LAYERNORM_NAMES = ["norm", "ln_f", "ln_attn", "ln_mlp", "ln_1", "ln_2"]


BASE_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir, os.pardir)

RENDER_FILE_PATH = os.path.join(BASE_PATH, "config", "render.ini")

LOG_PATH = os.path.join(BASE_PATH, "logs")

LOG_FILE_NAME = "train_log"


