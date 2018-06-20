#%% show how program is turned into token stream

from antlr.parser import Parser


with open("../data/appendix/A.java") as f:
    stringed = f.read()
    print(stringed)
    ast = Parser(0, stringed).parse_to_ast()
    print(ast)

#%% normalization procedure for identification
  
from preprocess.normalizer import normalize_for_ai    

with open("../data/appendix/A.java") as f:
     stringed = f.read()
     print(stringed)
     print("***********************************************")
     normalized = normalize_for_ai(stringed)
     print(normalized)
     
     
#%% ngram extraction
import pandas as pd
import glob

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from preprocess.normalizer import normalize_for_ai 
from antlr.parser import Parser

df = pd.DataFrame(columns=["source_code", "token_stream", "normalized"])

sc_files = sorted(glob.glob("../data/appendix/*.java"))

for i, fname in enumerate(sc_files):
    content = open(fname).read()
    token_stream = Parser(0, content).parse_to_ast()
    normalized = normalize_for_ai(content)
    df = df.append({"source_code": content, 
                    "token_stream": token_stream,
                    "normalized": normalized}, ignore_index=True)
    
ngram_vectorizer = CountVectorizer(analyzer='word', 
                                   ngram_range=(2, 2),
                                   token_pattern="[\S]+",
                                   lowercase= False,
                                   strip_accents="ascii")

transformer = TfidfTransformer(smooth_idf=True, norm=None)

X = ngram_vectorizer.fit_transform(df.token_stream)
tfidf = transformer.fit_transform(X)

res_tf = pd.DataFrame(X.A, columns=ngram_vectorizer.get_feature_names())

res_idf = pd.DataFrame(tfidf.A, columns=ngram_vectorizer.get_feature_names())

#%% t-SNE visual

from sklearn.manifold import TSNE
import matplotlib.pylab as plt
from sklearn.decomposition import TruncatedSVD

from matplotlib2tikz import save as tikz_save

X_reduced = TruncatedSVD(n_components=2, random_state=0).fit_transform(tfidf.todense())

plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)

example_code_classes = ["A", "B", "C"]

for i in range(0, 3):
    plt.annotate(example_code_classes[i],
            xy=(0, 0), xycoords='data',
            xytext=(X_reduced[i, 0], X_reduced[i, 1]), textcoords='data',
            arrowprops=dict(arrowstyle="<-",
                            connectionstyle="arc3"),
            )
            
