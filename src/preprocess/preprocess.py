# -*- coding: utf-8 -*-

# Preprocess the tsv by adding headers, concatenating source code files and
# calculate loc

import json
import pandas as pd

df = pd.read_csv('../data/ohpe2016s.tsv', 
                 sep='\t',
                 header=None,
                 names=['course', 'exercise', 'student', 'source_code'])

df['source_code'] = df['source_code'].apply(json.loads)


def stringify(sc_dict):
    base = ""
    for k in sc_dict.keys():
        base += sc_dict[k] + "\n"
    return base

def get_loc(sc_string):
    return sc_string.count("\n")
    

df['source_code'] = df.source_code.apply(stringify)
df['loc'] = df.source_code.apply(get_loc)

df.to_csv('../data/ohpe2016s_processed.csv')
