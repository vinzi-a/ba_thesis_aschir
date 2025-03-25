#!/bin/bash

# this file is for fixing the str(1)-file in their coordinates
for file in $all ; do
sed -i "s/0\.06 0\.06/ 0\.06 0\.06/g" $file
done