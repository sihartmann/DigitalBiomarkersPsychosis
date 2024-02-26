#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated processing pipeline to extract acoustic features from audio
recordings using openSMILE


Created on Mon Feb 12 11:16:50 2024

@author: Simon Hartmann
"""

import opensmile
import csv
import os

# Define openSMILE extractor, here eGeMAPS v02, set as low level descriptors
# to get features for every 100 milliseconds
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
)

# Print all features
features = smile.feature_names

# Set path with folders of all subjects
path = '../Interviews'

if os.path.isdir(path):
    subject_list = next(os.walk(path))[1]
else:
    print('Path does not exist!')
      
# Create csv file where acoustic features will be stored
f = open('../Interviews/AudioFeatures_Output.csv', 'w')

# create the csv writer
writer = csv.writer(f)
# Add the header name row into csv
writer.writerow([feature for feature in features])

# Loop through all subjects, check if audio file exists, extract acoustic
# features and write them to a csv file
for subject in subject_list:
    # Check if folder contains audio file 
    if not any(fname.endswith('.mp4') for fname in os.listdir(os.path.join(path, subject))):
        print("No audio file exists for subject:" + subject)
    else:
        features = smile.process_file(os.path.join(path, subject))
        averageFeatures = features.mean()
        writer.writerow(averageFeatures)

# Close csv file        
f.close()        
        
        
