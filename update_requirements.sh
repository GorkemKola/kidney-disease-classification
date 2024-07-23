#!/bin/bash

# Generate new requirements file
pipreqs . --force --savepath /tmp/requirements_new.txt

# Merge the existing and new requirements
cat requirements.txt /tmp/requirements_new.txt | sort | uniq > /tmp/requirements_merged.txt

# Replace the original requirements.txt with the merged one
mv /tmp/requirements_merged.txt requirements.txt
