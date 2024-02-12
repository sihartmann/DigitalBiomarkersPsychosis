#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 14:51:16 2024

@author: simon
"""

import whisper

model = whisper.load_model("base")

text = model.transcribe('../Interviews/Test_FreeSpeech_JamesSimon_16_01_2024/Audio_Participant_16012024.mp4')

#printing the transcribe

text['text']

with open('../Interviews/SpeechToText_example.txt', 'w') as f:
    f.write(text['text'])

f.close()