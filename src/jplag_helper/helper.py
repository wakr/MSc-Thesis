# -*- coding: utf-8 -*-



import pandas as pd

def write_df_to_source_codes(df):
    for i in range(df.shape[0]):
        id_ = df.student.iloc[i]
        sc = df.source_code.iloc[i]
        with open("test_files/" + str(id_) + ".java", 'w') as f:
            f.write(sc)
            
    print(f"{df.shape[0]} files written")



#%%