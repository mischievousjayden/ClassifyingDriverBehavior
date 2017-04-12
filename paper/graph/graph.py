import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def draw_plot_graph(df, xlabel="", ylabel="", title="", legend_loc="upper left"):
    columns = df.columns.tolist()
    for i, col in enumerate(columns):
        data = df[col].tolist()
        plt.plot(range(len(data)), data, label=col)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc=legend_loc)
    plt.show()


def draw_bar_graph(df, xlabel="", ylabel="", title="", legend_loc="upper left", outputfilename=None):
    columns = df.columns.tolist()
    rows = df.index.tolist()

    width = 0.3
    ind = np.arange(len(rows)) - width*(len(columns)-1)/2

    for i, col in enumerate(columns):
        plt.bar(ind+i*width, df[col].tolist(), width=width, label=col)

    plt.xticks(range(len(rows)), rows)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc=legend_loc)
    if outputfilename != None:
        plt.savefig(outputfilename)
    else:
        plt.show()

