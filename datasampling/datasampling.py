#!/usr/bin/python

import sys

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


if __name__ == "__main__":
    if len(sys.argv) != 4:
        message = "Usag: datafilter.py <period> <input_file> <output_file>"
        sys.exit(message)

    period = int(sys.argv[1])
    inputfile = sys.argv[2]
    outputfile = sys.argv[3]
    sampleData(period, inputfile, outputfile)

