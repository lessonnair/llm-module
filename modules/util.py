# -*- coding: utf-8 -*-

def isFloat(s):
    s = s.split('.')
    if len(s) > 2:
        return False
    else:
        for si in s:
            if not si.isdigit():
                return False
    return True