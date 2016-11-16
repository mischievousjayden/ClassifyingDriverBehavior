from os import listdir
from os.path import isfile, join
import csv

def getAllFileNames(path, suffix=""):
    return [join(path, f) for f in listdir(path) if isfile(join(path, f)) and f.endswith(suffix)]

def readDataFile(filename, delimiter=","):
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=delimiter)
        data = list(reader)
    return data

def readDataFiles(filenames, delimiter=","):
    data_set = [readDataFile(filename, delimiter) for filename in filenames ]
    return data_set

