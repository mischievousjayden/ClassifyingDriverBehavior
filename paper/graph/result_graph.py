import argparse

import numpy as np
import pandas as pd

import graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputfilename", help="result input filename")
    parser.add_argument("outputfilename", help="plot output filename")
    parser.add_argument("title", help="plot title")

    args = parser.parse_args()
    inputfilename = args.inputfilename
    outputfilename = args.outputfilename
    title = args.title

    df = pd.read_table(inputfilename, sep=',', header=0, index_col=0)
    df.index = df.index.map(str) 

    graph.draw_bar_graph(df, "The Number of Neurons", "Average Accuracy", title, "upper left", outputfilename)


if __name__ == "__main__":
    main()

