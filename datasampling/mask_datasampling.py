import argparse
from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd

import maskfilter


def sample_data(period, mask, inputdata):
    """The function samples data. 
    Args:
        period (int): sampling period
        mask (List<int>): mask filter
        inputdata (pandas dataframe): input data
    Returns:
        outputdata (pandas dataframe): sampled and filtered data
    """
    mask_size = len(mask)
    num_rows = inputdata.shape[0]
    outputdata = list()
    for i in range(mask_size//2, num_rows, period):
        max_row_index = num_rows - 1
        indices = np.arange(mask_size) - mask_size//2 + i
        indices = max_row_index - abs(max_row_index - abs(indices))
        outputdata.append(np.matmul(mask, inputdata.iloc[indices, :].as_matrix()))
    return pd.DataFrame(outputdata, columns=inputdata.columns)


def sample_data_set(period, mask, inputpath, outputpath):
    suffix = ".dat"
    file_list = [f for f in listdir(inputpath) if isfile(join(inputpath, f)) and f.endswith(".dat")]
    for f in file_list:
        inputfile = join(inputpath, f)
        outputfile = join(outputpath, "sample_" + f)
        with open(inputfile) as f:
            inputdata = pd.read_table(f, sep=' ', header=0)
            outputdata = sample_data(period, mask, inputdata)
            outputdata.to_csv(outputfile, sep=' ', index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputpath", help="input path")
    parser.add_argument("outputpath", help="output path")
    parser.add_argument("-p", "--period", help="period", type=int, default=10)
    parser.add_argument("-m", "--mask", help="""select mask
            1. last
            2. mean
            3. gaussian
            """, type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("-s", "--size", help="mask size", type=int, default=11)

    args = parser.parse_args()
    inputpath = args.inputpath
    outputpath = args.outputpath
    period = args.period
    mask_size = args.size
    if(args.mask == 3):
        mask = maskfilter.get_gaussian_mask(mask_size)
    elif(args.mask == 2):
        mask = maskfilter.get_mean_mask(mask_size)
    else:
        mask = maskfilter.get_delta_last(mask_size)

    sample_data_set(period, mask, inputpath, outputpath)


if __name__ == "__main__":
    main()

