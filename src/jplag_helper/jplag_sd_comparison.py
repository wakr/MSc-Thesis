# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 16:30:52 2018

@author: Kristian
"""

import pandas as pd
import matplotlib.pyplot as plot
import numpy as np

from sklearn.preprocessing import LabelEncoder



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
example_task = exerc_tasks[0]
print(example_task)

test_df = exam_df[exam_df.exercise == example_task].copy()

#%%
