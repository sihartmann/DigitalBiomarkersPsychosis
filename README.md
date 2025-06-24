# DIGITAL BIOMARKER PIPELINE

## DESCRIPTION
The presented pipeline is a small-scale automated end-to-end solution that can be used to extract facial, audio, and linguistic features from recorded video interviews. It is designed to analyse video/audio automatically recorded during HIPAA zoom interviews as part of clinical trials or research studies but can also be used in other situations.

It automatically extracts acoustic, linguistic, and facial movement information. It has the [Silero VAD](https://github.com/snakers4/silero-vad/) Voice Activity Detection system built in. Acoustic features such as pitch and jitter are extracted using [openSMILE](https://audeering.github.io/opensmile/). Automated transcription is performed by [Whisper](https://openai.com/index/whisper). Linguistic features such as part-of-speech tagging, dependency tagging, semantic coherence,  and sentiment scoring is done using [The Natural Language Toolkit](https://www.nltk.org/) and [spaCy](https://spacy.io/). Visual features such as gaze, head position, and [action units](https://www.cs.cmu.edu/~face/facs.htm) are extracted using [OpenFace](https://cmusatyalab.github.io/openface/).

A summary file is generated for each participant listing all extracted data. For cross-sectional studies, a summary file containing data from all participants is also generated.

## How To Use The Pipeline

See [wiki](https://github.com/sihartmann/DigitalBiomarkersPsychosis/wiki)
