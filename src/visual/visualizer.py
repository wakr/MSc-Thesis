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
   
    
