# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 11:38:47 2018

@author: Kristian
"""

import glob
import pprint
sim_results_ohpe = sorted(glob.glob("sd/ohpe/*.txt"))
sim_results_ohja = sorted(glob.glob("sd/ohja/*.txt"))

#%%

ohpe_files = sim_results_ohpe[2:3] # 3rd. task
ohja_files = sim_results_ohja # all ohja's tasks

#%%

def show_pruned(clusters):
    res = {}
    for k in clusters.keys():
        res[k] = []
        for p1, p2, score in clusters[k]:
            if len(clusters[k]) > 40:
                if score == 1.0:
                    res[k].append((p1, p2))
            else:
                res[k].append((p1, p2))
    return res

def res_to_authors(clusters):
    import itertools
    list2d = clusters.values()
    list2d_p = list(itertools.chain.from_iterable(list2d))
    return set(list(itertools.chain.from_iterable(list2d_p)))

#%%
ohpe_det_res = []

for f in ohpe_files:
    with open(f) as file:
        print(f)
        res = file.read()
        res = eval(res)
        res = show_pruned(res)
        ohpe_det_res.append(res)
        print(res.keys())
        for k in res.keys():
            print(f"{k}, size: {len(res[k])}")
        
        
#%%

ohja_det_res = []

for f in ohja_files:
    with open(f) as file:
        print(f)
        res = file.read()
        res = eval(res)
        res = show_pruned(res)
        ohja_det_res.append(res)
        print(res.keys())
        for k in res.keys():
            print(f"{k}, size: {len(res[k])}")
            
#%% True results

print("*********************TRUE************")
 
sim_results_ohpe = sorted(glob.glob("sd/ohpe/true/*true.txt"))
sim_results_ohja = sorted(glob.glob("sd/ohja/true/*true.txt"))

ohpe_files = sim_results_ohpe # 3rd. task
ohja_files = sim_results_ohja # all ohja's tasks

ohpe_true_res = []

for f in ohpe_files:
    with open(f) as file:
        print(f)
        res = file.read()
        res = eval(res)
        ohpe_true_res.append(res)
        print(res.keys())
        for k in res.keys():
            print(f"{k+1}, size: {len(res[k])}")     
        
ohja_true_res = []

for f in ohja_files:
    with open(f) as file:
        print(f)
        res = file.read()
        res = eval(res)
        ohja_true_res.append(res)
        print(res.keys())
        for k in res.keys():
            print(f"{k+1}, size: {len(res[k])}")
            
#%%
print("----------------TP/FP--------")
for i in range(len(ohpe_det_res)):
    ohpe_det_auths = res_to_authors(ohpe_det_res[i])
    ohpe_true_auths = res_to_authors(ohpe_true_res[i])
    print("*******")
    print(f"{len(ohpe_det_auths)}: {ohpe_det_auths}")
    print(f"{len(ohpe_true_auths)}: {ohpe_true_auths}")

print("---------OHJA-------")
    
for i in range(len(ohja_det_res)):
    ohja_det_auths = res_to_authors(ohja_det_res[i])
    ohja_true_auths = res_to_authors(ohja_true_res[i])
    print("*******")
    print(f"{len(ohja_det_auths)}: {ohja_det_auths}")
    print(f"{len(ohja_true_auths)}: {ohja_true_auths}")
            
            
            