# -*- coding: utf-8 -*-


import glob
import numpy as np
import pandas as pd

def write_df_to_source_codes(df):
    for i in range(df.shape[0]):
        id_ = df.student.iloc[i]
        sc = df.source_code.iloc[i]
        with open("test_files/" + str(id_) + ".java", 'w', encoding='utf8') as f:
            	f.write(sc)
            
    print(f"{df.shape[0]} files written")



#%% 
def get_jplag_results_as_authors(exercise): 
    max_sim = 0.8
    
    res_file_path = exercise
    
    result_data = None
    
    with open(res_file_path) as f:
        result_data = f.read()
        
    lines = result_data.split("\n")
    lines = lines[9:] # 8 first lines are jplag output
    
    detected_authors = []
    
    for l in lines:
            if not "Comparing" in l:
                continue
            splitted = l.split(" ") # Comparing xxx.java-yyy.java: d
            file_pair, score = splitted[1], float(splitted[2])
            score_dec = score / 100 # get sim [0, 1]
            f1, f2 = file_pair.split("-")
            f1_id = f1.strip(".java")
            f2_id = f2.strip(".java:")
            
            if score_dec >= max_sim:
                detected_authors.append(f1_id)
                detected_authors.append(f2_id)
                
    return list(set(detected_authors))

#%%
def get_sd_printed_to_authors(res_file_path):
    detected_authors = []
    with open(res_file_path) as f:
         s = f.read()
         cluster_dict = eval(s)
         for k in cluster_dict.keys():
             for res in cluster_dict[k]:
                 a1, a2, score = res
                 detected_authors.append(a1)
                 detected_authors.append(a2)
    return list(set(detected_authors))
#%%
    
import glob
from metrics.eval_metrics import jaccard_similarity

sim_results_ohpe = sorted(glob.glob("../results/sd/ohpe/*.txt"))
sim_results_ohja = sorted(glob.glob("../results/sd/ohja/*.txt"))

jplag_results_ohpe = sorted(glob.glob("../data/jplag/ohpe_results/*.txt"))
jplag_results_ohja = sorted(glob.glob("../data/jplag/ohja_results/*.txt"))

#%%

for i in range(len(jplag_results_ohpe)):
    file_jp = jplag_results_ohpe[i]
    file_sd = sim_results_ohpe[i]
    print("Comparing: ", file_jp)
    jp_auth = get_jplag_results_as_authors(file_jp)
    sd_auth = get_sd_printed_to_authors(file_sd)
    print(jaccard_similarity(jp_auth, sd_auth))
    print("----------")
    

for i in range(len(jplag_results_ohja)):
    file_jp = jplag_results_ohja[i]
    file_sd = sim_results_ohja[i]
    print("Comparing: ", file_jp)
    jp_auth = get_jplag_results_as_authors(file_jp)
    sd_auth = get_sd_printed_to_authors(file_sd)
    print(jaccard_similarity(jp_auth, sd_auth))
    print("----------")









