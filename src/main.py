import json

import pandas as pd

#%% read

df = pd.read_csv('./data/placeholder.csv')
df['source_code'] = df['source_code'].apply(json.loads)

#%% concatenate source codes into one text file 

def stringify(sc_dict):
    base = ""
    for k in sc_dict.keys():
        base += sc_dict[k] + "\n"
    return base

df['source_code'] = df.source_code.apply(stringify)
