import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import progressbar
from time import sleep

def PlotHeatMap(data) :
    corr = data.corr()
    # plot the heatmap
    ax = sns.heatmap(corr, 
            square = True,
            vmin=-1, vmax=1, center=0,
            cmap=sns.diverging_palette(20, 220, n=200),
            xticklabels=corr.columns,
            yticklabels=corr.columns)

    ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right')

    plt.show()


def PlotScatter(data) :
    pd.plotting.scatter_matrix(data)
    plt.show()

def PlotHist(data) : 
    data.iloc[:,0:9].hist(layout=(3,3))
    data.iloc[:,9:18].hist(layout=(3,3))
    data.iloc[:,18:27].hist(layout=(3,3))
    data.iloc[:,27:30].hist(layout=(3,3))
    plt.show()

def PlotDensity(data) : 
    data.iloc[:,0:9].plot(kind='density', subplots=True, layout=(3,3))
    data.iloc[:,9:18].plot(kind='density', subplots=True, layout=(3,3))
    data.iloc[:,18:27].plot(kind='density', subplots=True, layout=(3,3))
    data.iloc[:,27:30].plot(kind='density', subplots=True, layout=(3,3))
    plt.show()

def PlotBoxWhisker(data) :
    data.iloc[:,0:9].plot(kind='box', subplots=True, layout=(3,3))
    data.iloc[:,9:18].plot(kind='box', subplots=True, layout=(3,3))
    data.iloc[:,18:27].plot(kind='box', subplots=True, layout=(3,3))
    data.iloc[:,27:30].plot(kind='box', subplots=True, layout=(3,3))
    plt.show()


def InitProgressBar():
    bar = progressbar.ProgressBar(maxval=20, \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()

    return bar
    
    

def UpdateProgressBar(bar, value):
    bar.update(value)
    sleep(0.1)
    if value == 100:
        bar.finish()
