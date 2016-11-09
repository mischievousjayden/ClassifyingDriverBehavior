#!/bin/bash

for inputfile in $(find $2 -name *.dat)
do
    outputfile=$3/sample_$(basename $inputfile)
    python datasampling.py $1 $inputfile $outputfile
done

exit 0

