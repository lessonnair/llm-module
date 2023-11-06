# -*- coding: utf-8 -*-

def import_package(name):
    components = name.rsplit('.', 1)
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod