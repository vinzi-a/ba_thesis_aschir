#!/bin/bash

#Vinzent Aschir 18.10.2024: 
#file to change all the content from the origin directory to the future directory with deleting the old directory if wished"   

# ./change_dir.sh true data/custom_philo_generated data/custom_philo_generated_resized false
if [ $# -ne 4 ] ; then 
    echo "your input should look like: delete_dir=(true/false) orgin_dir=/ future_dir=/ copy_already_exisiting_files=(true/false)"
    printf "Your input was: 1. $1 = (delete_dir) \n 2. $2 = (origin_dir) \n 3. $3 = (future_dir) \n 4. $4 = (copy_already_existing_files) \n"
    exit 1; 
fi 
delete_dir=$1
origin_dir=$2
future_dir=$3 
copy_already_existing_files=$4

ac_dir=$(pwd)
cd $origin_dir
all_files=$(ls)
cd $ac_dir
chmod -R 755 $origin_dir

#creates the future dir if not existant: 
if ! [ -d "$future_dir" ] ; then
  echo "$future_dir doesn_t exist yet."
  mkdir $future_dir
fi

i=0

#all=$(find "$ac_dir" -type f | wc -l)
#sleep 1;
#print $all

#echo -ne  
for file in $all_files; do
    echo -ne "already copied: $i \r" #/ $all 
    sleep 0.000001; 
    #checks if the copied file is readable in the original dir
    if ! [ -r $origin_dir/$file ] ; then
        printf "error: $origin_dir/$file is not readable \n"
        delete_dir=false
        continue; 
    fi
    
    #if the file is already in the future dir and we want to copy it anyway, we delete it first in the future dir.
    if [ -r $future_dir/$file ] && $copy_already_existing_files ; then
        rm -rf $future_dir/$file
    fi
    if [ -r $future_dir/$file ] && [ "$copy_already_existing_files" = false ] ; then
        i=$(($i+1)) 
        continue; 
    fi
    #lets do the copying
    scp -p $origin_dir/$file $future_dir/$file &

    if ! [ $? ] ; then 
        printf "the copying from $file went wrong please check its content. \n"
        exit 1
    fi 
    #if ! [ -r $future_dir/$file ] ; then # since the copying is parallelised the copying process hasn't finished before checking this. 
    #    printf "error: $future_dir/$file is not readable \n"
    #    continue;
    #fi
    
    i=$(($i+1)) 
done 

wait
chmod -R 777 $future_dir
echo -ne "\n"
echo "all $i files copied from $origin_dir to $future_dir"; 

if $delete_dir ; then
    echo "content deleted at: $origin_dir"
    rm -rf $origin_dir/*
fi
