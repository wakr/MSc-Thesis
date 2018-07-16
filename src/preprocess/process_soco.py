# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plot
import numpy as np
from antlr.parser import Parser
from sklearn.feature_extraction import DictVectorizer
from normalizer import normalize_for_ast
import glob


def filepair_to_id(f1f2):
    f1,f2 = f1f2.split(" ")
    id1 = int(f1.split(".")[0])
    id2 = int(f2.split(".")[0])
    return id1, id2

def parse_source_code_columns(x):
    return Parser(x["ID"], x["source_code"]).parse_to_ast()
    
    
#%%

df = pd.DataFrame(columns=["ID", "source_code"])
#%%

sc_files = sorted(glob.glob("../data/soco/java/*.java"))

re_use = [line.rstrip('\n') for line in open('../data/soco/SOCO14-java.txt')]
re_use = [filepair_to_id(x) for x in re_use]

ground_truth = {i: 0 for i in range(len(sc_files))}
for p1, p2 in re_use:
    ground_truth[p1] = 1
    ground_truth[p2] = 1


y_plag = DictVectorizer(sparse=False).fit_transform(ground_truth)[0]  # plag y/n
#%%

for i, fname in enumerate(sc_files):
    content = open(fname).read()
    df = df.append({"ID": i, "source_code": content}, ignore_index=True)

df["source_code"] = df.source_code.apply(normalize_for_ast)
df["ast_repr"] = df.source_code.apply(lambda s: Parser(s).parse_to_ast())
    
#%% DATA SUMM

#dataset_summary(df)


#%% FEATURE EXTR

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

def build_sim_det_model(corpus, epsilon, ngram_size, minPts=2):
    ngram_vectorizer = CountVectorizer(analyzer='word', 
                                       ngram_range=(ngram_size,ngram_size),
                                       token_pattern="[\S]+",
                                       lowercase= False,
                                       strip_accents="ascii")

    tfidf_trans = TfidfTransformer(smooth_idf=True, norm='l2')
    X = ngram_vectorizer.fit_transform(corpus)
    W = tfidf_trans.fit_transform(X)
    sim_matrix = np.around(cosine_similarity(W), decimals=8)
    dist_matrix = np.subtract(np.ones(sim_matrix.shape, dtype=np.int8), sim_matrix) # sim <=>
    
    db = DBSCAN(min_samples=minPts, metric="precomputed", eps=epsilon).fit(dist_matrix)
    
    labels = db.labels_ # -1 = noise
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    unique_labels = set(labels) 

    labels

    clusters = {}

    for i, c in enumerate(labels):
        if c == -1: continue
        if not c in clusters.keys():
            clusters[c] = [i]
        else:
            clusters[c].append(i)
    clusters
    
    return labels, clusters


#%% 
    
corpus = df.ast_repr
authors = df.ID

hyperparam_ngram = [4,5,6,7,8]
hyperparam_epsilon = [0.4, 0.5, 0.6, 0.7] # lower eps indicate higher density necessary to form a cluster.

for n in hyperparam_ngram:
    for e in hyperparam_epsilon:
        
        print(f"running test run with n={n}, e={e}")
        
        labels, omega = build_sim_det_model(corpus, epsilon=e, ngram_size=n)

        y_hat = np.zeros(len(authors))

        for i, l in enumerate(labels): # 0 = free, 1 = plag
            if l != -1: 
                y_hat[i] = 1
        
        prec, recall, fscore, supp = precision_recall_fscore_support(y_plag, y_hat)
        print(prec) # plag-class
        print(confusion_matrix(y_plag, y_hat))
        tn, fp, fn, tp = confusion_matrix(y_plag, y_hat).ravel()
        print(f"(tn: {tn}, fp: {fp}, fn: {fn}, tp:{tp})")
        print("ACC:", accuracy_score(y_plag, y_hat))
        print("-------------------")
    print("***************************")


#%% test on SOCO test C1
        
df = pd.DataFrame(columns=["ID", "source_code"])

sc_files = sorted(glob.glob("../data/soco_test/C1/*"))

re_use = [line.rstrip('\n') for line in open('../data/soco_test/C1_test.txt')]
re_use = [filepair_to_id(x) for x in re_use]

ground_truth = {i: 0 for i in range(len(sc_files))}
for p1, p2 in re_use:
    ground_truth[p1] = 1
    ground_truth[p2] = 1

y_plag = DictVectorizer(sparse=False).fit_transform(ground_truth)[0]  # plag y/n

for i, fname in enumerate(sc_files):
    content = open(fname, encoding="utf8").read()
    df = df.append({"ID": i, "source_code": content}, ignore_index=True)

df["source_code"] = df.source_code.apply(normalize_for_ast)
df["ast_repr"] = df.source_code.apply(lambda s: Parser(s).parse_to_ast()) 

#%% test test C1

corpus = df.ast_repr
authors = df.ID

hyperparam_ngram = [9,10]
hyperparam_epsilon = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6] # lower eps indicate higher density necessary to form a cluster.

for n in hyperparam_ngram:
    for e in hyperparam_epsilon:
        
        print(f"running test run with n={n}, e={e}")
        
        labels, omega = build_sim_det_model(corpus, epsilon=e, ngram_size=n)

        y_hat = np.zeros(len(authors))

        for i, l in enumerate(labels): # 0 = free, 1 = plag
            if l != -1: 
                y_hat[i] = 1
        
        prec, recall, fscore, supp = precision_recall_fscore_support(y_plag, y_hat)
        print(prec) # plag-class
        print(confusion_matrix(y_plag, y_hat))
        tn, fp, fn, tp = confusion_matrix(y_plag, y_hat).ravel()
        print(f"(tn: {tn}, fp: {fp}, fn: {fn}, tp:{tp})")
        print(classification_report(y_plag, y_hat))
        print("ACC:", accuracy_score(y_plag, y_hat))
        print("-------------------")
    print("***************************") 
    
#%% C2
    
df = pd.DataFrame(columns=["ID", "source_code"])

sc_files = sorted(glob.glob("../data/soco_test/C2/*"))

re_use = [line.rstrip('\n') for line in open('../data/soco_test/C2_test.txt')]
re_use = [[int(y[2:]) for y in x.split(" ")] for x in re_use] # C2XXXX

ground_truth = {i: 0 for i in range(len(sc_files))}
for p1, p2 in re_use:
    ground_truth[p1] = 1
    ground_truth[p2] = 1

y_plag = DictVectorizer(sparse=False).fit_transform(ground_truth)[0]  # plag y/n
    
for i, fname in enumerate(sc_files):
    content = open(fname, encoding="utf8").read()
    df = df.append({"ID": i, "source_code": content}, ignore_index=True)

df["source_code"] = df.source_code.apply(normalize_for_ast)
df["ast_repr"] = df.source_code.apply(lambda s: Parser(s).parse_to_ast())

#%% test test C2    

corpus = df.ast_repr
authors = df.ID

hyperparam_ngram = [10]
hyperparam_epsilon = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6] # lower eps indicate higher density necessary to form a cluster.

for n in hyperparam_ngram:
    for e in hyperparam_epsilon:
        
        print(f"running test run with n={n}, e={e}")
        
        labels, omega = build_sim_det_model(corpus, epsilon=e, ngram_size=n)

        y_hat = np.zeros(len(authors))

        for i, l in enumerate(labels): # 0 = free, 1 = plag
            if l != -1: 
                y_hat[i] = 1
        
        prec, recall, fscore, supp = precision_recall_fscore_support(y_plag, y_hat)
        print(prec) # plag-class
        print(confusion_matrix(y_plag, y_hat))
        tn, fp, fn, tp = confusion_matrix(y_plag, y_hat).ravel()
        print(f"(tn: {tn}, fp: {fp}, fn: {fn}, tp:{tp})")
        print(classification_report(y_plag, y_hat))
        print("ACC:", accuracy_score(y_plag, y_hat))
        print("-------------------")
    print("***************************")    