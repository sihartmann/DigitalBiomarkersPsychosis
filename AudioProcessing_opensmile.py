#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated processing pipeline to extract acoustic features from audio
recordings using openSMILE


Created on Mon Feb 12 11:16:50 2024

@author: Simon Hartmann
"""

import opensmile

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
)

smile.feature_names

y = smile.process_file('../Interviews/Test_FreeSpeech_JamesSimon_16_01_2024/Audio_Participant_16012024.mp4')