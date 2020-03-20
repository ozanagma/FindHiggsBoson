import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


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
    data.hist()
    plt.show()