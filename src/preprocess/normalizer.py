# -*- coding: utf-8 -*-

import re

def normalize(source_code):
    res = re.sub("import.*;", "", source_code)
    res = re.sub("\".+\"", "\"\"", res)
    res = re.sub("\t+", "", res)
    res = re.sub("\n", "", res)
    res = re.sub("\s+", " ", res)
    res = re.sub("({|})", " ", res)
    res = re.sub("\d+", "t0", res)
    res = re.sub("(for|while)", "loop", res)
    return res
