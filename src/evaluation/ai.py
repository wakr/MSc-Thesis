import pandas as pd
import matplotlib.pyplot as plot
import numpy as np
from antlr.parser import Parser
from preprocess.normalizer import normalize_for_ai

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

import random

random.seed(5)



def categorize_columns(df_):
    df_temp = df_.copy()
    df_temp['course'] = df_temp['course'].astype('category')
    df_temp['exercise'] = df_temp['exercise'].astype('category')
    df_temp['student'] = df_temp['student'].astype('category')
    df_temp['source_code'] = df_temp['source_code'].astype(str)
    return df_temp

def parse_source_code_columns(df):
    df["ast_repr"] = df["source_code"].apply(lambda s: Parser(s).parse_to_ast())

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




#%%

ohpe_weeks = [1,2,3,4,5,6,7]
ohja_weeks = [8,9,10,11,12,13,14]




from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier


def print_top10(vectorizer, clf, class_labels):
    """Prints features with the highest coefficient values, per class"""
    feature_names = vectorizer.get_feature_names()
    for i, class_label in enumerate(class_labels):
        top10 = np.argsort(clf.coef_[i])[-10:]
        print("%s: %s" % (class_label,
              ", ".join(feature_names[j] for j in top10)))

#%%

for week in []:
    label_encoder, train, val, test = create_split(wo_exam_df,
                                                   week_target=week)
    X_train, y_train = train
    X_train = X_train.apply(normalize_for_ai)

    X_val, y_val = val
    X_val = X_val.apply(normalize_for_ai)

    X_test, y_test = test

    ngram_range = [10]
    for ng in ngram_range:
        ngram_vectorizer = CountVectorizer(analyzer='char', 
                                           ngram_range=(ng, ng),
                                           token_pattern=u'(?u)\b\w\w+\b',
                                           lowercase= False,
                                           strip_accents="ascii")

        tfidf_trans = TfidfTransformer(smooth_idf=True, norm='l2')
        X_tr = ngram_vectorizer.fit_transform(X_train)
        W_train = tfidf_trans.fit_transform(X_tr)
        
        
        print(W_train.shape)
        #print(ngram_vectorizer.vocabulary_.keys())
        
        nb = MultinomialNB().fit(W_train, y_train) # train model
        
        #print_top10(ngram_vectorizer, nb, nb.classes_)
        
        X_vl = ngram_vectorizer.transform(X_val)
        W_val = tfidf_trans.transform(X_vl)
        
        y_val_hat = nb.predict(W_val)
        
        
        
        print(f"week: {week}, ng: {ng}")
        print(f1_score(y_val, y_val_hat, average="macro"))
        print("ACC: ", accuracy_score(y_val, y_val_hat))
        print(classification_report(y_val, y_val_hat))
        print("")
    print("***************************")





#%% 


## misclasmaxrate: bd272d95 = 12, befa8db7 = 15, c5762555 = 28, c5ea8819 = 32

from collections import Counter

#print(Counter(y_val))

#print(Counter(y_val_hat))

#%% TEST
week = 7
ng = 10


label_encoder, train, val, test = create_split(wo_exam_df,
                                               week_target=week)
X_train, y_train = train
X_train = X_train.apply(normalize_for_ai)

X_val, y_val = val
X_val = X_val.apply(normalize_for_ai)

X_train = pd.concat([X_train, X_val])
y_train = np.concatenate((y_train, y_val))

X_test, y_test = test
X_test = X_test.apply(normalize_for_ai)


ngram_vectorizer = CountVectorizer(analyzer='char', 
                                           ngram_range=(ng, ng),
                                           token_pattern=u'(?u)\b\w\w+\b',
                                           lowercase= False,
                                           strip_accents="ascii")

tfidf_trans = TfidfTransformer(smooth_idf=True, norm='l2')
X_tr = ngram_vectorizer.fit_transform(X_train)
W_train = tfidf_trans.fit_transform(X_tr)

        
nb = MultinomialNB().fit(W_train, y_train) 
        
        
X_tst = ngram_vectorizer.transform(X_test)
W_tst = tfidf_trans.transform(X_tst)


y_test_hat = nb.predict(W_tst)

print(f"week: {week}, ng: {ng}")
print(f1_score(y_test, y_test_hat, average="macro"))
print("ACC: ", accuracy_score(y_test, y_test_hat))
print(classification_report(y_test, y_test_hat))
print("")