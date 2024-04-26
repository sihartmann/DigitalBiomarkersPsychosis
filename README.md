# PIPELINE TO ANALYZE DIGITAL BIOMARKERS

## DESCRIPTION
## STEPS
- ffmpeg: Parse periods of silence from participant audio\
AUDIO:
- opensmile: Extract audio features of participant.
- whisper: Transcripe both interviewer and participant audio
- nltk/spacey: Extract linguistic features\
VIDEO:
- OpenFace: analyze participant video for gaze, posture and action units

## INSTALLATION
- Download latest release and extract.

## REQUIREMENTS
- OpenFace (tested with v2.2.0): Download [here](https://github.com/TadasBaltrusaitis/OpenFace/releases/tag/OpenFace_2.2.0) 
- Whisper (tested with r136): Download the standalone version [here](https://github.com/Purfview/whisper-standalone-win/releases/tag/Whisper-OpenAI)
- ffmpeg (tested with version 2024-03-11-git-3d1860ec8d-full_build-www.gyan.dev): Download [here](https://www.gyan.dev/ffmpeg/builds/)

## SET-UP
- Place all required packages into the DigBio folder. Place a copy of the ffmpeg executable into the folder of the whisper installation that contains the file 'whisper.exe'
- Create a folder for every participant. The name of this folder will be the participant's ID. Each folder must contain the following files, downloaded from HIPAA zoom.
    - [date and time of the interview]_Recording_separate1.mp4
    - [date and time of the interview]_Recording_separate1.mp4
    - [date and time of the interview]_Recording_gvo_1280x720.mp4
- Do not rename these files. If you don't have either video or audio from the recording, you can still run the pipeline, however the output will be reduced.
- Place all of your participants folders into a folder called e.g. 'my_interviews'.

## USAGE
```bash
.\digital_biomarkers.exe --interviews <path to my_interviews>
```
## Options:
`--interviews` [path to participant folder] This is the only required argument. Make sure you use the full path.\
`--mode [audio,video,all]` Run only part of the pipeline. See description for more details on what will be run in these modes. The default is all, i.e. the complete pipeline.\
`--verbosity [1,2,3,4]` Change amount of detail in logging output. Recommended (and default mode) is 3. Use 4 if you need to debug something, and 1 or 2 if you want very limited output.\
`--overwrite` Use this if you want to start the entire pipeline again. This will delete all files other than the recordings, so use with caution.\
`--no_cut` Will skip the video cutting step. This option should be used if your video either only shows one person or the format is not 1280x720. You may need to manually trim your videos to only show one person.\
`--whisper_model [tiny,small,base,medium,large]` Change the model that whisper uses for transcription. Default is base.\