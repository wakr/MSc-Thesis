# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plot
import numpy as np
from antlr.parser import Parser
from sklearn.feature_extraction import DictVectorizer
from normalizer import normalize
from visual.visualizer import dataset_summary
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

#df["source_code"] = df.source_code.apply(normalize)
#df["ast_repr"] = df.apply(parse_source_code_columns, axis=1)
    
#%%    

dataset_summary(df)



#%% FEATURE EXTR
    
    
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

ngram_vectorizer = CountVectorizer(analyzer='char', 
                                   ngram_range=(12,12),
                                   max_df=0.6,
                                   min_df=0.1,
                                   strip_accents="ascii")

transformer = TfidfTransformer(smooth_idf=False)

corpus = df.source_code
authors = df.ID

X = ngram_vectorizer.fit_transform(corpus)
tfidf = transformer.fit_transform(X)

#%%

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from sklearn import manifold

from mpl_toolkits.mplot3d import Axes3D

# TODO: Add treshold

sim_matrix = np.around(cosine_similarity(tfidf), decimals=8)
dist_matrix = np.subtract(np.ones(sim_matrix.shape,  dtype=np.int8), sim_matrix) # sim <=> 1 - dist

#%%

X_reduced = TruncatedSVD(n_components=50, random_state=0).fit_transform(tfidf)
X_embedded = TSNE(n_components=2, perplexity=50, verbose=2).fit_transform(X_reduced)

plot.scatter(X_embedded[:, 0], X_embedded[:, 1], marker="x")
"""
for label, x, y in zip(authors, X_embedded[:, 0], X_embedded[:, 1]):
    plot.annotate(
        label,
        xy = (x, y), xytext = (-20, 20),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.1),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
"""
#%%

mds = manifold.MDS(n_components=2, dissimilarity="precomputed")
results = mds.fit(dist_matrix)

coords = results.embedding_

plot.subplots_adjust(bottom = 0.1)
plot.scatter(
    coords[:, 0], coords[:, 1], marker = 'x'
    )

#%%

plot.imshow(sim_matrix, zorder=2, cmap='Blues', interpolation='nearest')
plot.colorbar()


#%%

from sklearn.cluster import DBSCAN

db = DBSCAN(min_samples=2, metric="precomputed", eps=0.2).fit(dist_matrix)
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

#%%

def visualize_clusters(cluster, sample_matrix):
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.preprocessing import StandardScaler


    core_samples_mask = np.zeros_like(cluster.labels_, dtype=bool)
    core_samples_mask[cluster.core_sample_indices_] = True
    labels = cluster.labels_
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = 'k'

        class_member_mask = (labels == k)  # generator comprehension 
        # X is your data matrix

        xy = sample_matrix[class_member_mask & core_samples_mask]

        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=8, alpha=0.7)

        xy = sample_matrix[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=5, alpha=0.7)

#%%

visualize_clusters(db, coords)

#%%
from sklearn.metrics import classification_report

y_hat = np.zeros(len(authors))

for i, l in enumerate(labels):
    if l != -1: 
        y_hat[i] = 1
        
print(classification_report(y_plag, y_hat))
#%%

plot.scatter(
    coords[:, 0], coords[:, 1], marker = 'x'
    )
"""
for author, label, x, y in zip(authors, labels, coords[:, 0], coords[:, 1]):
    lbl = str(label) + "({})".format(str(author))
    if label == -1: 
        plot.annotate(
            lbl,
            xy = (x, y), xytext = (-10, 10),
            textcoords = 'offset points', ha = 'right', va = 'bottom',
            bbox = dict(boxstyle = 'round,pad=0.5', fc = 'white', alpha = 0.1),
            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
    else:
        plot.annotate(
            label,
            xy = (x, y), xytext = (-20, 20),
            textcoords = 'offset points', ha = 'right', va = 'bottom',
            bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.2),
            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
    
 """

#%%

mds = manifold.MDS(n_components=3, dissimilarity="precomputed")
results = mds.fit(dist_matrix)

coords = results.embedding_

fig = plot.figure()
ax = Axes3D(fig)
ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], marker="x")   
    
    
    
    
    
    
    
    
    
    
    
    
    
    