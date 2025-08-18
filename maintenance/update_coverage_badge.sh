#!/bin/bash

TOTAL=$(coverage report | awk 'END{print $4}' | sed 's/%//')

if (( $(echo "$TOTAL <= 50" | bc -l) )) ; then
    COLOR=red
elif (( $(echo "$TOTAL > 80" | bc -l) )); then
    COLOR=green
else
    COLOR=orange
fi

curl "https://img.shields.io/badge/coverage-$TOTAL%25-$COLOR" > badge.svg