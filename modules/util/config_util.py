# -*- coding: utf-8 -*-

import configparser
import json
import re
from modules.util.util import *
from modules.util.custom_log import get_logger
from modules.util import constants

SCIENTIFIC_NOTATION_PATTERN = re.compile("^([\\+|-]?\\d+(.{0}|.\\d+))[Ee]{1}([\\+|-]?\\d+)$")

logger = get_logger(__name__)

def read_config_file(config_path, encoding="utf-8"):
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(config_path, encoding=encoding)
    return config


class TaskConfig(object):

    def __init__(self, config_path):
        self.config = read_config_file(config_path)

    def get(self, *args, **kwargs):
        return self.config.get(*args, **kwargs)

    def _parse_field_list_str(self, s):
        ret = None
        if s is not None and len(s) > 0:
            s = s.strip()
            ret = [p.strip() for p in s.split(",")]
        return ret

    def parse_data_path_list(self,
                             section_name,
                             field_name,
                             iter=False,
                             default=None):
        ret = []
        dataPathList = self._parse_field_list_str(self.config.get(section_name, field_name))

        if iter:
            for s in dataPathList:
                ret.append(self.config.get(section_name, s))
        else:
            ret = dataPathList

        if ret is None:
            ret = default
        return ret

    def get_section_field_value(self, section_name, field_name):
        return self.config.get(section_name, field_name)

    def parse_value(self, v, empty_to_none=False, parse_json=False):
        v = str(v)
        res = v
        if v == 'None':
            res = None
        elif v in ('True', 'true'):
            res = True
        elif v in ('False', 'false'):
            res = False
        elif isFloat(v) or v.isdecimal() or SCIENTIFIC_NOTATION_PATTERN.match(v):
            res = eval(v)
        elif parse_json:
            res = json.loads(v)
        elif v == '' and empty_to_none:
            res = None
        return res

    def get_section_kvs(self, section_name,
                        empty_to_none=False,
                        parse_json=False):
        kvs = self.config.items(section_name)
        kvs = {k: self.parse_value(v, empty_to_none=empty_to_none, parse_json=parse_json) for k, v in kvs}
        return kvs


class RenderConfig(object):

    def __init__(self, config_path):
        self.config = read_config_file(config_path)

    def get(self, *args, **kwargs):
        return self.config.get(*args, **kwargs)

    def get_section_field_value(self, section_name, field_name):
        return self.config.get(section_name, field_name)

    def parse_value(self, v, empty_to_none=False):
        v = str(v)
        res = v
        if v == 'None':
            res = None
        elif v in ('True', 'true'):
            res = True
        elif v in ('False', 'false'):
            res = False
        elif isFloat(v) or v.isdecimal() or SCIENTIFIC_NOTATION_PATTERN.match(v):
            res = eval(v)
        elif v == '' and empty_to_none:
            res = ''
        else:
            try:
                res = json.loads(v)
            except Exception as e:
                logger.warning(e)
                res = None
        return res

    def get_section_kvs(self, section_name,
                        empty_to_none=False,
                        parse_json=False):
        kvs = self.config.items(section_name)
        kvs = {k: self.parse_value(v, empty_to_none=empty_to_none, parse_json=parse_json) for k, v in kvs}
        return kvs


render_config = RenderConfig(constants.RENDER_FILE_PATH)


if __name__ == '__main__':
    config = TaskConfig("../../config/template.ini")

    # print(config.get_section_field_value("vanilla", "pretrained_model_name_or_path"))
    params = config.get_section_kvs("llama2")
    pass
