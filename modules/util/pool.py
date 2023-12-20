# -*- coding: utf-8 -*-

import torch
import threading
from modules.util.custom_log import get_logger

logger = get_logger(__name__)

lock = threading.Lock()

class InstancePool(object):
    _instance = None
    pool = {}

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            try:
                lock.acquire()
                if not cls._instance:
                    cls._instance = super().__new__(cls, *args, **kwargs)
                    cls.pool = {}
                lock.release()
            except Exception as e:
                logger.error(e)
        return cls._instance

    def put(self, key, obj):
        self.pool[key] = obj

    def get(self, key):
        return self.pool.get(key)

    def clear(self):
        lock.acquire()
        for obj in self.pool.pop():
            del obj
        lock.release()
        torch.cuda.empty_cache()

