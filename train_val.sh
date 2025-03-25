#!/bin/bash

#Vinzent Aschir 18.01.205: 

#this fucntion splits data dependent on the given arguments into train and validation data.
#should the validationset contain different data? 

directory=$1
gen_val=$2 # bool if the validation set should contain generated data 
percentage_val=$3 # percentage of data that should contain the validation set

if [ $percentage_val -le 0 ] or [ $percentage_val -ge 1 ] ; then 
    echo "percentage_val should be be a value in between 0 and 1"
    exit 1
fi

ac_dir=$(pwd)

cd $directory

n=$(ls -f | wc -l) # n stores the amount of files in this directory 

v= $(echo "$n * $percentage_val" | bc) # v stores the amount of files that should be in the validation set

for i in $(seq 1 $n); do
    if [ $i -le $(($n/10)) ] ; then
        mv $(ls | head -n 1) $ac_dir/train/
    else
        mv $(ls | head -n 1) $ac_dir/val/
    fi
done 