#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import random

random.seed(5)

from preprocess.normalizer import normalize_for_ai

df = pd.read_csv('../data/ohpe2016s_processed.csv')

#%%


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


#%%
course_info = {'ohpe': ('hy-s2016-ohpe', 'hy-s2016-ohpe-konekoe-3'),
               'ohja': ('hy-s2016-ohja', 'hy-s2016-ohja-konekoe-3')}

subset_df = df[df.course.isin(course_info["ohpe"])]
subset_df = categorize_columns(subset_df)

subset_df["source_code"] = subset_df["source_code"].apply(normalize_for_ai)

#%% Add week and remove pair programming

subset_df = add_week_exer_column(subset_df) # 99 = examm, -1 noise

subset_df = subset_df[subset_df.exercise_num != -1] # remove pair prog

#%% Exclude/include exam

wo_exam_df = subset_df[subset_df.exercise_num != 999] # all except exam
exam_df = subset_df[subset_df.exercise_num == 999] # exams

#%% Split function: last exercise test, preceding 80% train and 20% validation

def create_split(df_, week_target, max_author_count=None):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import OneHotEncoder
    
    last_exer_id = df_[df_.week <= week_target].exercise_num.max()
    last_exer_name = df_[df_.exercise_num == last_exer_id].iloc[0].exercise
    print(f"Week {week_target}, last exercise {last_exer_name}")
    df_ = df_[df_.week <= week_target]
    test_df = df_[df_.exercise_num == last_exer_id]
    temp_df = df_[df_.exercise_num != last_exer_id]

    
    # inspect only last-exer students
    students_submitted = test_df.student.unique().tolist()
    if max_author_count:
        students_submitted = random.sample(students_submitted, max_author_count)
    
    print("Students: ", len(students_submitted))
    print("profile avg", temp_df.groupby(temp_df.student).source_code.count().mean())
    temp_df = temp_df[temp_df.student.isin(students_submitted)]
    test_df = test_df[test_df.student.isin(students_submitted)]
    
    label_encoder = LabelEncoder() # encode student_id
    # label_encoder.inverse_transform <-- to turn int back to hash_id
    y_test = label_encoder.fit_transform(test_df.student)
    X_test = test_df.source_code
    
    X_train, X_val, y_train, y_val = train_test_split(temp_df.source_code, 
                                                      temp_df.student, 
                                                      test_size=0.2)
    
    
    return label_encoder, (X_train, label_encoder.transform(y_train)), (X_val, label_encoder.transform(y_val)),(X_test, y_test)



#%% create split

label_encoder, train, val, test = create_split(wo_exam_df,
                                               week_target=7,
                                               max_author_count=50)

#%% concatenate train_X per student

"""
TRAIN
1. concate all files per student
2. L most frequent ngrams extracted
--> profile of student

CLASSIFY
1. Get file and extract ngrams
2. Compare to each profile and get the intersect of same ngrams
3. Calculate the size
"""

# train + val vs test
full_train_X = pd.concat([train[0], val[0]])
full_y_train = np.concatenate((train[1], val[1]))





train_DF = pd.DataFrame({"X": full_train_X, "y": full_y_train})

train_S = train_DF.groupby("y")["X"].apply(lambda s: s.sum())

#%%
def counts_to_most_L_frequent(df_, L):
    res = {}
    for profile in df_.index.tolist():
        res[profile] = df_.loc[profile].nlargest(L).index.tolist()
    return res

# {}, []
def decide_nearest(profiles_, document_profile):
    sims = [] # id, sim
    for k in profiles_.keys():
        ngrams = profiles_[k]
        size_of_intersect = 0
        for ng1 in ngrams:
            for ng2 in document_profile:
                if ng1 == ng2:
                    size_of_intersect = size_of_intersect + 1
        sims.append((k, size_of_intersect))
    sorted_by_second = sorted(sims, key=lambda tup: tup[1], reverse=True)
    return sorted_by_second

L = 10**2
n = 10

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(lowercase=False, 
                             ngram_range=(n, n), 
                             analyzer="char",
                             token_pattern=u'(?u)\b\w\w+\b')

W = vectorizer.fit_transform(train_S, train_S.index.tolist())
count_vect_df = pd.DataFrame(W.todense(), columns=vectorizer.get_feature_names())

profile_dict = counts_to_most_L_frequent(count_vect_df, L)

#%% get test

test_DF = pd.DataFrame({"X": test[0], "y": test[1]})

vectorizer = CountVectorizer(lowercase=False, 
                             ngram_range=(n, n), 
                             analyzer="char",
                             token_pattern=u'(?u)\b\w\w+\b')

W_test = vectorizer.fit_transform(test_DF.X)
count_vect_test_df = pd.DataFrame(W_test.todense(), columns=vectorizer.get_feature_names())

#%%
y_true_authors = test_DF.y.tolist()
y_pred_authors = []

for i in range(test_DF.shape[0]):
    test_profile = count_vect_test_df.iloc[i].nlargest(L).index.tolist()

    res = decide_nearest(profile_dict,  test_profile)
    print(f"{len(y_pred_authors)}, ", end=" ")
    y_pred = res[0][0] # sorted (a, sim)
    y_pred_authors.append(y_pred)
#%%
print()
print(y_true_authors)
print(y_pred_authors)

from sklearn.metrics import accuracy_score, classification_report

print(accuracy_score(y_true_authors, y_pred_authors))
print(classification_report(y_true_authors, y_pred_authors))