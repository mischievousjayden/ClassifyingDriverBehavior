#!/usr/bin/python

import sys
from os import listdir
from os.path import isfile, join


def sampleData(period, inputfile, outputfile):
    """The function samples data. 
    Example: sampleData(3, inputfile, outputfile)
        create outputfile with every 3rd line in inputfile
        
    Args:
        period (int): sampling period
        inputfile (string): input file name
        outputfile (string): output file name

    Returns:
        void
    """

    with open(inputfile, 'r') as fread, open(outputfile, 'w') as fwrite:
        for i, line in enumerate(fread):
            if (i % period) == 0:
                fwrite.write(line)


def sampleDataSet(period, inputpath, outputpath):
    suffix = ".dat"
    file_list = [f for f in listdir(inputpath) if isfile(join(inputpath, f)) and f.endswith(".dat")]
    for f in file_list:
        inputfile = join(inputpath, f)
        outputfile = join(outputpath, "sample_" + f)
        sampleData(period, inputfile, outputfile)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        message = "Usag: datasampling.py <period> <input_path> <output_path>"
        sys.exit(message)

    period = int(sys.argv[1])
    inputpath = sys.argv[2]
    outputpath = sys.argv[3]
    sampleDataSet(period, inputpath, outputpath)

