import pandas as pd

#%% read

df = pd.read_csv('./data/ohpe2016s_processed.csv')


df['course'] = df['course'].astype('category')
df['exercise'] = df['exercise'].astype('category')
df['student'] = df['student'].astype('category')


# (full_course, exam)
course_info = {'ohpe': ('hy-s2016-ohpe', 'hy-s2016-ohpe-konekoe-3'),
               'ohja': ('hy-s2016-ohja', 'hy-s2016-ohja-konekoe-3')}


df.head(5)

#%% 

subset_df = df[df.course == course_info["ohpe"][0]].head(5)