# -*- coding: utf-8 -*-

import torch

def isFloat(s):
    s = s.split('.')
    if len(s) > 2:
        return False
    else:
        for si in s:
            if not si.isdigit():
                return False
    return True

def get_dtype(torch_dtype_str):
    res = None
    for dtype_name, dtype in [("fp16", torch.float16), ("bf16", torch.bfloat16), ("fp32", torch.float32)]:
        if torch_dtype_str == dtype_name:
            res = dtype
            break
    return res