# -*- coding: utf-8 -*-

class InstancePool(object):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        self.pool = {}

    def put(self, key, obj):
        self.pool[key] = obj

    def get(self, key):
        return self.pool.get(key)
