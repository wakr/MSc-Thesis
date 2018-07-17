from matplotlib2tikz import save as tikz_save
import matplotlib.pylab as plt
import numpy as np
import matplotlib



#plt.style.use('thesis')


#t = np.arange(0.0, 1.0 + 0.01, 0.01)
#s = np.cos(4 * np.pi * t) + 2

#plt.plot(t, s)

#tikz_save('fig.tikz',
#           figureheight = '\\figureheight',
#           figurewidth = '\\figurewidth')


#%%



def dataset_summary(df):
    
    tasks = len(df.exercise.cat.categories)
    num_subm = len(df.student.cat.categories)
    avgchar = df.source_code.apply(lambda d: len(d)).mean()
    
    avgloc = df.source_code.apply(lambda d: len(d.split("\n"))).mean()
    maxloc = df.source_code.apply(lambda d: len(d.split("\n"))).max()
    minloc = df.source_code.apply(lambda d: len(d.split("\n"))).min()
    
    exprs = df.source_code.apply(lambda d: len(d.split(";"))).mean()
    
    exer_loc_avg = df.groupby(by="exercise").loc.mean().round(0)
    
    pass

#%%    
def acgloc_hist():
    plt.style.use('default')
    plt.figure()
    #ax = plt.gca()
    plt.ylim((0,45))
    #ax.yaxis.grid(True)
    ax = plt.subplot(111)  
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()  
    ax.get_yaxis().tick_left()
    #plt.xticks(range(0, 400, 10))  
    #plt.yticks(range(0, 70, 10))
    plt.xlabel("Average LOC")  
    plt.ylabel("Count")
    plt.hist(list(exer_loc_avg.dropna().values), bins=50, edgecolor="black")
    
    tikz_save('fig.tikz',
           figureheight = '\\figureheight',
           figurewidth = '\\figurewidth')
    
#%% AI OHPE results, students: [2, 4, 10, 20, 30, 40, 50]. Ng = 14 char


var_author_res_ohpe = [0.38, 0.18, 0.09, 0.10, 0.07, 0.07, 0.05]
var_author_res_ohja = [0.68, 0.10, 0.06, 0.01, 0.01, 0.02, 0.02]
x = [2,4,10,20,30,40,50]
plt.plot(x, var_author_res_ohpe, label="OHPE")
plt.plot(x, var_author_res_ohja, label="OHJA", linestyle=":")
plt.xticks(x, x)
plt.xlabel("Author pool")
plt.ylabel("Accuracy %")
plt.legend()
    
#tikz_save('fig.tikz',
#          figureheight = '\\figureheight',
#           figurewidth = '\\figurewidth')

#%% SD OHPE Cluster sizes (relative max/sum)

model = ["A", "B", "C", "D", "E"]
relative_size_a = [0.63, 0.89, 0.88,  0.92, 0.85]

relative_size_b = [0.13, 0.34, 0.59, 0.45, 0.23]

plt.bar(model, relative_size)
plt.xlabel("Model")
plt.ylabel("Relative size")

#tikz_save('fig.tikz',
#          figureheight = '\\figureheight',
#          figurewidth = '\\figurewidth')