# -*- coding: utf-8 -*-

#%% eval jplag vs soco


#bash_cmd = f"java -jar {jplag_path} -l java17 -m 60% {soco_files}"
#java -jar jplag-2.11.9-SNAPSHOT-jar-with-dependencies.jar./../data/soco/java > result.txt
import glob
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

res_file_path = "../data/jplag/result.txt"

result_data = None

with open(res_file_path) as f:
    result_data = f.read()


#%%
   

lines = result_data.split("\n")
lines = lines[9:] # 8 first lines are jplag output
    
#%% GROUND TRUTH

def filepair_to_id(f1f2):
    f1,f2 = f1f2.split(" ")
    id1 = int(f1.split(".")[0])
    id2 = int(f2.split(".")[0])
    return id1, id2

sc_files = sorted(glob.glob("../data/soco/java/*.java"))

re_use = [line.rstrip('\n') for line in open('../data/soco/SOCO14-java.txt')]
re_use = [filepair_to_id(x) for x in re_use]

ground_truth = {i: 0 for i in range(len(sc_files))}
for p1, p2 in re_use:
    ground_truth[p1] = 1
    ground_truth[p2] = 1

y_plag = DictVectorizer(sparse=False).fit_transform(ground_truth)[0]

#%% PREDICT AND EVAL
    
allowed_disssim = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


for dis in allowed_disssim:
    max_sim = 1 - dis
    y_hat = [0] * len(y_plag)
    
    for l in lines:
        if not "Comparing" in l:
            continue
        splitted = l.split(" ") # Comparing xxx.java-yyy.java: d
        file_pair, score = splitted[1], float(splitted[2])
        score_dec = score / 100 # get sim [0, 1]
        f1, f2 = file_pair.split("-")
        f1_id = int(f1.strip(".java"))
        f2_id = int(f2.strip(".java:"))
        
        if score_dec >= max_sim:
            y_hat[f1_id] = 1
            y_hat[f2_id] = 1

    print("allowed dissimilarity:", dis)
    print(classification_report(y_plag, y_hat, target_names=["non-plag", "plag"]))
    print("ACC:", accuracy_score(y_plag, y_hat))
    print(confusion_matrix(y_plag, y_hat))
    tn, fp, fn, tp = confusion_matrix(y_plag, y_hat).ravel()
    print(f"tn: {tn}, fp: {fp}, fn: {fn}, tp: {tp}")
    print("-------------------")
    






















