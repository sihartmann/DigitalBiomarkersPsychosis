# DIGITAL BIOMARKER PIPELINE

## DESCRIPTION
The presented pipeline is a small-scale automated end-to-end solution that can be used to extract facial, audio, and linguistic features from recorded video interviews. It is designed to analyse video/audio automatically recorded during HIPAA zoom interviews as part of clinical trials or research studies but can also be used in other situations.

It automatically extracts acoustic, linguistic, and facial movement information. It has the [Silero VAD](https://github.com/snakers4/silero-vad/) Voice Activity Detection system built in. Acoustic features such as pitch and jitter are extracted using [openSMILE](https://audeering.github.io/opensmile/). Automated transcription is performed by [Whisper](https://openai.com/index/whisper). Linguistic features such as part-of-speech tagging, dependency tagging, semantic coherence,  and sentiment scoring is done using [The Natural Language Toolkit](https://www.nltk.org/) and [spaCy](https://spacy.io/). Visual features such as gaze, head position, and [action units](https://www.cs.cmu.edu/~face/facs.htm) are extracted using [OpenFace](https://cmusatyalab.github.io/openface/).

A summary file is generated for each participant listing all extracted data. For cross-sectional studies, a summary file containing data from all participants is also generated.

## Versions

* 1.0
Initial version including:
    - Parallel processing using subprocesses
    - Automated video cropping to right half of the video (if needed)
    - Silence removal prior to audio processing (silent periods > 2 seconds)
    - Acoustic feature extraction using openSMILE
    - Automated transcription using Whisper
    - Semantic coherence calculation using lemmatization
    - Sentiment score calculation
    - Frequency of POS tags (Universal tags) and dependeny tags calculation
    - Punctuation and contraction removal
    - Facial movement analysis using OpenFace
    - Facial movement analysis for speaking vs non-speaking periods
    - Automated writing of output data into summary file

* 1.0 with GUI:
    - Added Graphical User Interface to start pipeline

* 1.1 with GUI
    - All POS features are normalized according to the total number of sentences
    - Added Penn Treebank POS Tags
    - Added frequency of first, second, and third person personal pronouns
    - Added min, max, and average OpenFace confidence score to summary output
    - Changed semantic similarity calculation using BERT
    - Modified silenceremoval parameter for more aggressive silence removal (> 2 seconds) from audio file for opensmile
    - Updated ffmpeg call to prevent error messages in Whisper log file (ffmpeg needs to be installed now instead of * - added to the path -> see documentation)
    - Updated UI file to version 1.1
    - Updated dictionary

* 1.2.1 with GUI
    - Added Silero Voice Activity Detection to pipeline
    - Updated GUI to integrate VAD option and new design
    - Added max sentence length as feature
    - Added min, max, and variance of semantic similarity as features

* 1.2.2 with GUI
        - Whisper timestamps extracting start and end timestamps for segments and/or words in transcripts using [whisper-timestamped](https://github.com/linto-ai/whisper-timestamped)

## INSTALLATION
- Download latest release ([here](https://github.com/sihartmann/DigitalBiomarkersPsychosis/releases/tag/v1.2_GUI)) and extract content to your preferred destination.

## REQUIREMENTS
- OpenFace (tested with v2.2.0): Download [here](https://github.com/TadasBaltrusaitis/OpenFace/releases/tag/OpenFace_2.2.0)
- Whisper (tested with r136): Download the standalone version [here](https://github.com/Purfview/whisper-standalone-win/releases/tag/Whisper-OpenAI)
- ffmpeg (tested with version 2024-03-11-git-3d1860ec8d-full_build-www.gyan.dev): Download [here](https://www.gyan.dev/ffmpeg/builds/)

## SET-UP
- Place OpenFace and Whisper folders into the extracted folder.
- Follow installation of ffmpeg by adding ffmpeg to the Windows PATH (see ‘Advanced system settings’ -> ‘Environment Variables’ -> ‘System Variables’ -> ‘Path’)
- Create a folder for every participant. The name of this folder will be the subject's ID. Each folder must contain the following files, e.g. downloaded from HIPAA zoom.
    - A file ending with ‘1.mp4’ or ‘1.m4a’ indicating the interviewer audio stream, e.g. when downloaded from HIPAA Zoom [date and time of the interview]_Recording_separate1.mp4
    - A file ending with ‘2.mp4’ or ‘2.m4a’ indicating the participant audio stream, e.g. when downloaded from HIPAA Zoom [date and time of the interview]_Recording_separate2.mp4
    - A file ending including ‘gvo’ and ending on ‘.mp4’ indicating the recorded video, e.g. when downloaded from HIPAA Zoom [date and time of the interview]_Recording_gvo_1280x720.mp4

- Do not rename these files. If you don't have either video or audio from the recording, you can still run the pipeline, however the output will be reduced.
- If your recordings are not from HIPAA zoom, you will need to rename the files to match the above format.
- Place the folders of all subjects in the same directory

## USAGE

Start interface either by using PowerShell
```bash
.\digital_biomarkers.exe
```
or by double-clicking the biomarker_pipe.exe file.

![GUI design](/figures/DigBio_GUI.png)

Select path to interviews folder and specify all other parameters with the dropdown menu.
- Mode: Run only audio, video or both
- Verbosity: 4 for more detail, 1 and 2 for minimal detail.
- Overwrite old results: Start pipeline from scratch. Will delete all previously generated files, so use with caution.
- Skip video cropping: Check if not using HIPAA zoom. You may need to crop videos manually to only show one person.
- Whisper model: Select different sized models for transcription.
- VAD: Select Voice Activity Detection mode - 'None' for no detection, 'VAD' for Voice Activity Detection, or 'Both' for high level audio results on only speech parts and low level information on whole audio.
- Whisper timestamps: Select Whisper timestamp mode - 'None' for no timestamps, 'Segment' for start and end timestamps of segments such as sentences or phrases, or 'Words' for start and end timestamps of words.

The popup window will close once the pipeline has finished.

## OUTPUT
-	**logs**: Contains logs generated by ffmpeg, Whisper and OpenFace (depending on the ‘Verbosity’ that was selected). If there are any errors, the log files are a good starting point for troubleshooting.
-	**[participant name]_aligned**: Generated by OpenFace. Contains every frame from the video recording. The size of the folder increases with the length of the video.
-	**[participant name].mp4**: The processed OpenFace video showing gaze and posture.
-	**[participant name].csv**: Generated by OpenFace. Contains data on gaze, posture and action units for each video frame.
-	**[participant name].hog**: Generated by OpenFace.
-	**[participant name]_openface_out.csv**: OpenFace summary file. Contains rate of binary action unit activation and rate for no-speaking vs. speaking.
-	**[participant name]_of_details.txt**: Generated by OpenFace. General information on configuration and parameters.
-	**[participant name]_nltk_results.txt**: Detailed information on tokenisation, part of speech and dependency tagging.
-	**[participant name]_nltk_results.csv**: Summary file on semantic analysis step. Contains sentiment scores, POS and dependency tagging counts, average similarity score between neighbours and interviewer/participant speech ratio
-	**[participant name]_sim_scores.csv**: Generated during semantic analysis. Full matrix showing similarity between each sentence.
-	**[participant name]_opensmile.csv**: Generated by openSMILE. Contains information on various acoustic markers every 10 milliseconds.
-	**[participant name]_summary_opensmile**: Summary of openSMILE results. Shows average of every data point.
-	**[participant name]_transcript.txt**: Transcript generated by Whisper for the participant.
-	**[participant name]_interviewer_transcript.txt**: Transcript generated by Whisper for the interviewer.
-	**[participant name]_video.txt**: Video cropped to only show one person.
-	**[participant name]_silences.txt**: Generated by ffmpeg. Contains on the detected periods of non-speaking in the participant’s audio.
-	**[participant name]_summary.csv**: Summary file per participant of all pipeline steps.
-	**all_summary.csv**: Summary file for all participants. This file can be found in the parent folder.
