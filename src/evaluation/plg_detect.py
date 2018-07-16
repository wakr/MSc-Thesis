import pandas as pd
import matplotlib.pyplot as plot
import numpy as np
from antlr.parser import Parser
from preprocess.normalizer import normalize_for_ai, normalize_for_ast
from sklearn.preprocessing import LabelEncoder

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support



import random



def categorize_columns(df_):
    df_temp = df_.copy()
    df_temp['course'] = df_temp['course'].astype('category')
    df_temp['exercise'] = df_temp['exercise'].astype('category')
    df_temp['student'] = df_temp['student'].astype('category')
    df_temp['source_code'] = df_temp['source_code'].astype(str)
    return df_temp

def add_week_exer_column(df_):
    def assign_week(exer_name):
        if not "viikko" in exer_name:
            return 999
        else:
            strip_week = lambda s: s.split("-")[0] # viikkoXX-ViikkoXX_000.FOO
            numerical = strip_week(exer_name)[-2:] #viikkoXX --> XX
            return int(numerical)
        
    def assign_exercise_id(exer_name):
        if not "viikko" in exer_name:
            return 999
        elif "Kurssipalaute" in exer_name:
            return -1
        else:
            strip_week = lambda s: s.split("_")[1] # viikkoXX-ViikkoXX_000.FOO
            numerical = None # unknown, usually pairprogramming_id
            try:
                 numerical = int(strip_week(exer_name)[:3]) #000.FOO --> 000
            except:
                 numerical = -1
            return numerical

        
        
    df_temp = df_.copy()
    df_temp["week"] = df_temp["exercise"].apply(assign_week)
    df_temp["exercise_num"] = df_temp["exercise"].apply(assign_exercise_id)
    return df_temp

    
#%% read and choose only OHPE/OHJA
    
df = pd.read_csv('../data/ohpe2016s_processed.csv')

# (full_course, exam)
course_info = {'ohpe': ('hy-s2016-ohpe', 'hy-s2016-ohpe-konekoe-3'),
               'ohja': ('hy-s2016-ohja', 'hy-s2016-ohja-konekoe-3')}

subset_df = df[df.course.isin(course_info["ohpe"])]
subset_df = categorize_columns(subset_df)

pair_programming_tasks = [x for x in subset_df.exercise.cat.categories if "Pariohjelmointi" in x]

subset_df = subset_df[~subset_df.exercise.isin(pair_programming_tasks)]
#%% Add week and remove pair programming

subset_df = add_week_exer_column(subset_df) # 99 = examm, -1 noise

subset_df = subset_df[subset_df.exercise_num != -1] # remove pair prog

#%% Exclude/include exam

wo_exam_df = subset_df[subset_df.exercise_num != 999] # all except exam
exam_df = subset_df[subset_df.exercise_num == 999] # exams

#%% SD


exerc_tasks = exam_df.exercise.unique().tolist()
example_task = exerc_tasks[1]
print(example_task)

test_df = exam_df[exam_df.exercise == example_task].copy()
test_df["source_code"] = test_df.source_code.apply(normalize_for_ast)
test_df["ast_repr"] = test_df.source_code.apply(lambda s: Parser(s).parse_to_ast())

#%%




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
    
    print(W.shape)
    
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
    
    return labels, clusters, sim_matrix


#%% 
    
corpus = test_df.ast_repr
authors = test_df.student
lb = LabelEncoder().fit(authors)

sim_tresh = 0.4

labels, omega, sim_matrix = build_sim_det_model(corpus, epsilon=1-sim_tresh, ngram_size=7)

#%%

def clusters_to_sim_pairs(omega_, sim_matrix_, label_encoder, sim_tresh):
    import warnings
    warnings.filterwarnings(action='ignore', category=DeprecationWarning)
    c = {}
    for k in omega_.keys():
        suspects = omega_[k]
        c[k] = []
        for sa in suspects:
            for sb in suspects:
                if sa == sb: 
                    continue
                sim = sim_matrix_[sa, sb]
                if sim < sim_tresh:
                    continue
                sa_key = label_encoder.inverse_transform(sa)
                sb_key = label_encoder.inverse_transform(sb)
                already_exists_flipped = False
                for sc in c[k]: # (a,b) == (b,a) <-- remove duplicates
                    if sc[0] == sb_key and sc[1] == sa_key:
                        already_exists_flipped = True
                if already_exists_flipped:
                    continue
                c[k].append((sa_key, sb_key, sim))
    return c

asd = clusters_to_sim_pairs(omega, sim_matrix, lb, sim_tresh)

cluster_sizes = [len(omega[k]) for k in omega.keys()]

max(cluster_sizes) / sum(cluster_sizes)
