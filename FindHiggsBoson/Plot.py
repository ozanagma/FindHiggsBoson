import sys
import re
import time
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
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig( 'Figures/HeatMap_' + timestamp + '.eps', format = 'eps')
    plt.show()


def PlotScatter(data) :
    pd.plotting.scatter_matrix(data)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig( 'Figures/Scatter_' + timestamp + '.eps', format = 'eps')
    plt.show()

def PlotHist(data) : 
    data.iloc[:,0:9].hist(layout=(3,3))
    data.iloc[:,9:18].hist(layout=(3,3))
    data.iloc[:,18:27].hist(layout=(3,3))
    data.iloc[:,27:30].hist(layout=(3,3))
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig( 'Figures/Hist_' + timestamp + '.eps', format = 'eps')
    plt.show()

def PlotDensity(data) : 
    data.iloc[:,0:9].plot(kind='density', subplots=True, layout=(3,3))
    data.iloc[:,9:18].plot(kind='density', subplots=True, layout=(3,3))
    data.iloc[:,18:27].plot(kind='density', subplots=True, layout=(3,3))
    data.iloc[:,27:30].plot(kind='density', subplots=True, layout=(3,3))
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig( 'Figures/Density_' + timestamp + '.eps', format = 'eps')
    plt.show()

def PlotBoxWhisker(data) :
    data.iloc[:,0:9].plot(kind='box', subplots=True, layout=(3,3))
    data.iloc[:,9:18].plot(kind='box', subplots=True, layout=(3,3))
    data.iloc[:,18:27].plot(kind='box', subplots=True, layout=(3,3))
    data.iloc[:,27:30].plot(kind='box', subplots=True, layout=(3,3))
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig( 'Figures/BoxWhisker_' + timestamp + '.eps', format = 'eps')
    plt.show()


def InitProgressBar():
    bar = progressbar.ProgressBar(maxval=100, \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    return bar
    

def UpdateProgressBar(bar, value):
    value = 100 if value >= 100 else value
    bar.update(value)
    if value == 100:
        bar.finish()
