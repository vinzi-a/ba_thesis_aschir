#!/bin/bash

for file in $all ; do
sed -i "s/0\.06 0\.06/ 0\.06 0\.06/g" $file
done