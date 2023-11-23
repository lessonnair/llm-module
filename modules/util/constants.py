# -*- coding: utf-8 -*-

import os

IGNORE_INDEX = -100

BASE_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir, os.pardir)

RENDER_FILE_PATH = os.path.join(BASE_PATH, "config", "render.ini")

LOG_PATH = os.path.join(BASE_PATH, "logs")
