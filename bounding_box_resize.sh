#!/bin/bash

#Vinzent Aschir 18.10.2024: file to change all the dirs 

#this script needs the generated data to be in the format: [number].txt" to give it a different bounding box size
#take care that the bounding box values are given relatively to the image size. 

#compile in bash shell: 
# ./bounding_box_resize.sh data/custom_philo_generated/labels 0.08 0.06 

#To do: 
# split this in an editing and a copying function. to not make it to complicated. 

# Check if the correct number of arguments is provided
if [ "$#" -ne 3 ] ; then
    echo "Usage: working_dir bounding_box_custom_size bounding_box_generated_data_size"
    printf "Your input was: 1. (working_dir) = $1 \n 2. (bounding_box_custom_size) = $2 \n 3. (bounding_box_generated_data_size) = $3 \n"
    exit 1
fi
#echo "you are compiling out of the dir:"
#pwd

working_dir=$1
bounding_box_custom=$2 #suggested 0.06
bounding_box_generated=$3 #suggested 0.05

#for the parallelisation we want to use the tool limited ressource that every paralell process can increment the counter. 
#for using the tool we have to add it to the PAth variable.
# dirs=$(pwd) #save current path 
# our_dir=$(find $dirs -type f -name "limit_ressource")
# 
# if [ -z $our_dir ] ; then 
#     printf "limit_ressource not found check if u have this file installed and not removed. \n"
#     exit 1
# fi
# 
# our_dir=$(dirname $our_dir)
# i=$PATH 
# PATH=$i:$our_dir

ac_dir=$(pwd)
cd $working_dir

#checks if the bounding box parameter is in relative and not in absolute size
if [ $(echo "$bounding_box_custom < 1" | bc -l) -eq 0 ] | [ $(echo "$bounding_box_generated < 1" | bc -l) -eq 0 ] ; then 
    printf "bounding box size must be smaller than 1 \n"
    exit 1
fi

generated=$(ls | grep -E "^[[:digit:]]*.txt") # list of all the files that are generated 
custom=$(ls | grep -E --invert-match "^[[:digit:]]*.txt") # list of all the files that are custom

#resizing of the custom data: 
echo "custom files:" 
i=0 
j=0
for file in $custom; do   
    # takes the 4th and 5th column and sets them to the bounding box size
    #sed -Ei "s/^(([0-9,\.]*\s){3})[0-9,\.]*[0-9,\.]*\s[0-9,\.]*/\1\ $bounding_box_custom\ $bounding_box_custom/g" $file &
    sed -Ei "s/^([0-9]*\s[0-9,\.]*\s[0-9,\.]*\s)[0-9,\.]*\s[0-9,\.]*/\1\ $bounding_box_custom\ $bounding_box_custom/g" $file &
    #awk -v bbox="$bounding_box_custom" '{ $4 = $5 = bbox }1' $file > $file 
    #some file couldn't be resized?
    if [ $? -ne 0 ] ; then 
        printf "resizing failed at file $file is not readable \n"
        j=$(($j+1))
        continue; 
    fi 
    echo -ne "already editet: $i and $j failed. \r" 
    sleep 0.000001;
    i=$(($i+1))
    #mv temp $file
    #rm -rf temp  
done
echo -ne "\n";
echo "all $i custom files editet and $j failed."

#resizing of the generated data:
k=0
v=0 
echo "generated_files:"
for file in $generated; do

    # takes the 4th and 5th column and sets them to the bounding box size
    sed -Ei "s/^(([0-9,\.]*\s){3})[0-9,\.]*[0-9,\.]*\s[0-9,\.]*/\1\ $bounding_box_generated\ $bounding_box_generated/g" $file & 
    #some file couldn't be resized?
    if [ $? -ne 0 ] ; then 
        echo "resizing failed at file $file"
        v=$(($v+1))
        continue; 
    fi 
    echo -ne "already editet: $k and $v failed. \r"
    sleep 0.000001; 
    k=$(($k+1))
    #mv temp $file
    #rm -rf temp  
done
echo -ne "\n"
echo "all $k generated files editet and $v failed."


