#!/bin/bash
while IFS='' read -r line || [[ -n "$line" ]]; do
    scp $2 $line:$3
done < "$1"
