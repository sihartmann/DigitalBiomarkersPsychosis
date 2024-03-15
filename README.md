# DigitalBiomarkersPsychosis
Automated pipeline to extract and process digital biomarkers from audio and video recordings from mental health interviews

## File structure
There must be a folder called 'Interviews' (or something else) which contains subfolders, which contain the patient data.
e.g. Interviews -> patient_1 -> patient_1_audio.mp3, patient_1_video.mp3

## Installation

* Install anaconda with `pip install anaconda`
* Create anaconda environment with `conda env create -n [name]`
* Activate your environment with `conda activate [name]`
* Install opensmile using `pip install opensmile`
* Install ffmpeg on your system with `sudo apt install ffmpeg`
* Download and extract openSMILE from https://github.com/audeering/opensmile/releases
* Install cmake and g++, if not already present
* run `bash build.sh`
* Install nltk by running `pip install nltk`
* Install the following nltk data packages by running:
    <!-- # import nltk
    # nltk.download('maxent_ne_chunker')
    # nltk.download('words')
    # nltk.download('vader_lexicon')
    # nltk.download('averaged_perceptron_tagger')
    # nltk.download('wordnet')
    # nltk.download('punkt')
    # nltk.download('stopwords') -->
    in a python script.

## Running the pipeline
To run the program with test data, run:
`python3 biomarker_pipeline.py --file config.txt`

