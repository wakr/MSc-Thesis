import pandas as pd
import matplotlib.pyplot as plot
import numpy as np
from antlr.parser import Parser


def categorize_columns(df):
    df_temp = df.copy()
    df_temp['course'] = df_temp['course'].astype('category')
    df_temp['exercise'] = df_temp['exercise'].astype('category')
    df_temp['student'] = df_temp['student'].astype('category')
    return df_temp

def parse_source_code_columns(df):
    df["ast_repr"] = df["source_code"].apply(lambda s: Parser(s).parse_to_ast())

def main():
    pass

if __name__ == "__main__":
    main()
   
#%%
    
df = pd.read_csv('./data/ohpe2016s_processed.csv')

# (full_course, exam)
course_info = {'ohpe': ('hy-s2016-ohpe', 'hy-s2016-ohpe-konekoe-3'),
               'ohja': ('hy-s2016-ohja', 'hy-s2016-ohja-konekoe-3')}

cur_exercise = "viikko07-Viikko07_127.Arvosanajakauma"

subset_df = df[df.course == course_info["ohpe"][0]]
subset_df = subset_df[subset_df.exercise == cur_exercise]
subset_df = categorize_columns(subset_df)

#%%
example_code = subset_df.iloc[0].source_code

#parse_source_code_columns(subset_df)

#%%

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

ngram_vectorizer = CountVectorizer(analyzer='char', 
                                   ngram_range=(5,5), 
                                   token_pattern="[\S]+", 
                                   strip_accents="ascii")

transformer = TfidfTransformer(smooth_idf=False)

corpus = subset_df.source_code
authors = subset_df.student

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

X_reduced = TruncatedSVD(n_components=50, random_state=0).fit_transform(tfidf)
X_embedded = TSNE(n_components=2, perplexity=40, verbose=2).fit_transform(X_reduced)
#%%
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

#%%

plot.subplots_adjust(bottom = 0.1)
plot.scatter(
    coords[:, 0], coords[:, 1], marker = 'x'
    )
"""
for label, x, y in zip(authors, coords[:, 0], coords[:, 1]):
    plot.annotate(
        label,
        xy = (x, y), xytext = (-20, 20),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.1),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
"""
#%%

plot.imshow(sim_matrix, zorder=2, cmap='Blues', interpolation='nearest')
plot.colorbar();

#%%

mds = manifold.MDS(n_components=3, dissimilarity="precomputed")
results = mds.fit(dist_matrix)

coords = results.embedding_

fig = plot.figure()
ax = Axes3D(fig)
ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], marker="x")


#%%

from sklearn.cluster import DBSCAN

db = DBSCAN(min_samples=2, metric="precomputed", eps=0.05).fit(dist_matrix)
labels = db.labels_ # -1 = noise
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
unique_labels = set(labels) 

labels

#%%

clusters = {}

for i, c in enumerate(labels):
    if c == -1: continue
    if not c in clusters.keys():
        clusters[c] = [i]
    else:
        clusters[c].append(i)









