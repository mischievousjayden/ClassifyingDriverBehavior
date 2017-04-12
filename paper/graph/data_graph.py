import argparse

import numpy as np
import pandas as pd

import graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="filename")
    parser.add_argument("column_num", help="column number", type=int)

    args = parser.parse_args()
    filename = args.filename
    column_num = args.column_num

    filters = ("last", "mean", "gaussian")
    df = pd.DataFrame()
    columnname = "Value"
    for filter_name in filters:
        fullname = "../../data/" + filter_name + "/" + filename
        data = pd.read_table(fullname, sep=' ', header=0)
        columnname = data.columns[column_num]
        df[filter_name] = data.iloc[:,[column_num]]

    graph.draw_plot_graph(df, "Time", columnname, "Data Graph", "upper right")


if __name__ == "__main__":
    main()

