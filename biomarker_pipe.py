"""
Pipeline for extraction of facial, acoustic and linguistic features from HIPAA Zoom recordings.
@author: Oliver Williams, Simon Hartmann
"""
import logging
import multiprocessing.process
import os
import subprocess
import sys
import opensmile
import csv
import shutil
import multiprocessing
import nltk
import string
from nltk.sentiment import SentimentIntensityAnalyzer
import contractions
import spacy
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from argparse import Namespace
import PyQt5
from PyQt5.QtWidgets import *
from PyQt5.uic import *
import silero_vad
from scipy.io.wavfile import read as read_wav
from scipy.io.wavfile import write as write_wav
import numpy as np
from scipy.signal import resample
import whisper_timestamped
import whisper_timestamped.make_subtitles
from whisper_timestamped.transcribe import write_csv


VERSION = '1.3 with GUI'

## Main processing class.
class pipe:

	# Parse arguments and set up logging.
	def parse_args(self, args):
		self.audio = args[0].replace('/','\\')
		self.video = args[1].replace('/', '\\')
		self.audio_interviewer = args[2].replace('/', '\\')
		self.video_interviewer = ""
		self.loglevel = args[3]
		self.run_mode = args[4]
		self.vad = args[5]
		self.whisper_time = args[6]
		self.no_cut = args[7]
		self.whisper_model = args[8]
		self.interviewer_analysis = args[9]
		if self.audio is not None:
			self.participant_dir = os.path.dirname(self.audio)
		elif self.video is not None:
			self.participant_dir = os.path.dirname(self.video)
		else:
			raise Exception("You must provide video and/or audio files.")
		self.participant_name = os.path.basename(self.participant_dir)
		self.nltk_out = f'{self.participant_dir}\\{self.participant_name}_nltk_results.csv'
		self.nltk_out_interviewer = f'{self.participant_dir}\\{self.participant_name}_interviewer_nltk_results.csv'
		self.summary =  f'{self.participant_dir}\\..\\all_summary.csv'
		if self.interviewer_analysis:
			self.interviewer_summary = f'{self.participant_dir}\\..\\all_interviewer_summary.csv'
		stderrhandler = logging.StreamHandler()
		filehandler = logging.FileHandler(f"{os.path.dirname(self.participant_dir)}\\DigBio.log")
		filehandler.setLevel(int(self.loglevel))
		filehandler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
		stderrhandler.setLevel(int(self.loglevel))
		stderrhandler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
		self.logger = logging.getLogger('DigBio')
		self.logger.addHandler(stderrhandler)
		self.logger.addHandler(filehandler)
		match self.loglevel:
			case '1':
				self.logger.setLevel(logging.ERROR)
			case '2':
				self.logger.setLevel(logging.WARNING)
			case '3':
				self.logger.setLevel(logging.INFO)
			case '4':
				self.logger.setLevel(logging.DEBUG)

		self.logger.info("\tDigBio version: {0}.".format(VERSION))
		self.logger.debug("\tRunning DigBio with command: {}.".format(' '.join(sys.argv)))
		self.whisper_path = r"Whisper-OpenAI_r136\Whisper-OpenAI\whisper.exe"
		if not os.path.isfile(self.whisper_path):
			self.logger.error("\tMissing dependency: whisper.")
			#sys.exit(1)
		self.feat_detect =  r"OpenFace_2.2.0_win_x64\FeatureExtraction.exe"
		if not os.path.isfile(self.feat_detect):
			self.logger.error("\tMissing dependency: openface.")
			#sys.exit(1)

	# Parse through Interviews directory and return list of all participant names.
	def get_participants(self, directory):
		participants = list()
		for entry in os.listdir(directory):
			if os.path.isdir(os.path.join(directory, entry)):
				participants.append(entry)
		return participants

	# Run openSMILE, generates both detailed and summary output.
	def run_opensmile(self):
		opensmile_path = f"{self.participant_dir}\\{self.participant_name}"
		self.f_summary = f"{opensmile_path}_summary_opensmile.csv"
		f_individual = f"{opensmile_path}_opensmile.csv"
		if os.path.isfile(f_individual) and os.path.isfile(self.f_summary):
			self.logger.info("\tOutput files already exist. Skipping OpenSmile...")
		else:
			self.logger.info(f"\tStarting opensmile for participant {self.participant_name}.")
			smile = opensmile.Smile(
				feature_set = opensmile.FeatureSet.eGeMAPSv02,
				feature_level = opensmile.FeatureLevel.LowLevelDescriptors,
			)
			if self.vad == 'VAD':
				features = smile.process_file(self.opensmile_audio)
			else:
				features = smile.process_file(self.audio)
			df = pd.DataFrame(features, columns=smile.feature_names)
			df.to_csv(f_individual, index=False)

			smile_summary = opensmile.Smile(
				feature_set = opensmile.FeatureSet.eGeMAPSv02
			)
			if self.vad != 'None':
				features = smile_summary.process_file(self.opensmile_audio)
			else:
				features = smile_summary.process_file(self.audio)
			df = pd.DataFrame(features, columns=smile_summary.feature_names)
			df.to_csv(self.f_summary, index=False)
			self.logger.info("\tOpensmile has completed successfully.")
		
		if self.interviewer_analysis:
			opensmile_path = f"{self.participant_dir}\\{self.participant_name}"
			self.f_interviewer_summary = f"{opensmile_path}_summary_interviewer_opensmile.csv"
			f_individual = f"{opensmile_path}_interviewer_opensmile.csv"
			if os.path.isfile(f_individual) and os.path.isfile(self.f_interviewer_summary):
				self.logger.info("\tOutput files already exist. Skipping interviewer OpenSmile...")
			else:
				self.logger.info(f"\tStarting opensmile for interviewer of {self.participant_name}.")
				smile = opensmile.Smile(
					feature_set = opensmile.FeatureSet.eGeMAPSv02,
					feature_level = opensmile.FeatureLevel.LowLevelDescriptors,
				)
				if self.vad == 'VAD':
					features = smile.process_file(self.interviewer_cleaned_audio)
				else:
					features = smile.process_file(self.audio_interviewer)
				df = pd.DataFrame(features, columns=smile.feature_names)
				df.to_csv(f_individual, index=False)

				smile_summary = opensmile.Smile(
					feature_set = opensmile.FeatureSet.eGeMAPSv02
				)
				if self.vad != 'None':
					features = smile_summary.process_file(self.interviewer_cleaned_audio)
				else:
					features = smile_summary.process_file(self.audio_interviewer)
				df = pd.DataFrame(features, columns=smile_summary.feature_names)
				df.to_csv(self.f_interviewer_summary, index=False)
				self.logger.info("\tOpensmile has completed successfully.")			

	# Transcribe both interviewer and participant audio using specified model.
	def run_whisper(self, model):
		self.logger.info(f"\tStarting whisper for participant {self.participant_name}. This may take a while.")
		try:
			os.mkdir(self.log_dir)
		except OSError:
			self.logger.debug("\tThe log directory already exists. Existing logs will be overwritten.")
		log_file = self.participant_name + "_whisper.log"
		self.transcript = f"{self.participant_dir}\\{self.participant_name}_transcript.txt"
		self.transcript_int = f"{self.participant_dir}\\{self.participant_name}_interviewer_transcript.txt"
		self.transcript_clean = f"{self.participant_name}_transcript.txt"
		self.transcript_int_clean = f"{self.participant_name}_interviewer_transcript.txt"
		if self.vad != 'None':
			whisper_audio = self.opensmile_audio
		else:
			whisper_audio = self.audio
		if os.path.isfile(self.transcript):
			self.logger.info("\tTranscript for this participant already exists. Skipping transcription...")
		else:
			linux_audio = whisper_audio.replace('\\','/')
			command = f'{self.whisper_path} {linux_audio} -f txt --model {model} --language English --hallucination_silence_threshold 1 --word_timestamps True --output_dir {self.participant_dir}> {self.log_dir}\\{log_file} 2>&1'
			self.logger.debug("\tRunning whisper on participant file using {}.".format(command))
			subprocess.check_output(command, shell=True)
			command = f'ren {whisper_audio[:-4]}.txt {self.transcript_clean}'
			subprocess.check_output(command, shell=True)
			self.logger.info(f"\twhisper (participant {self.participant_name}) has finished successfully.")
		if self.audio_interviewer is not None:
			if self.vad != 'None':
				whisper_interviewer_audio = self.interviewer_cleaned_audio
			else:
				whisper_interviewer_audio = self.audio_interviewer
			if os.path.isfile(self.transcript_int):
				self.logger.info("\tTranscript for this interviewer already exists. Skipping transcription...")
			else:
				linux_audio = whisper_interviewer_audio.replace('\\','/')
				command = f'{self.whisper_path} {linux_audio} -f txt --model {model} --language English --hallucination_silence_threshold 1 --word_timestamps True --output_dir {self.participant_dir}> {self.log_dir}\\{log_file} 2>&1'
				self.logger.debug("\tRunning whisper on interviewer file using {}.".format(command))
				subprocess.check_output(command, shell=True)
				command = f'ren {whisper_interviewer_audio[:-4]}.txt {self.transcript_int_clean}'
				subprocess.check_output(command, shell=True)
				self.logger.info(f"\twhisper (interviewer) {self.participant_name} has finished successfully.")
		else:
			if self.interviewer_analysis:
				self.logger.warning("\tNo interviewer audio found. Please provide interviewer audio or deactivate interviewer analysis.")
	
	def flatten(self, list_of_lists, key = None):
		for sublist in list_of_lists:
			for item in sublist.get(key, []) if key else sublist:
				yield item
		
	def run_whisper_timestamped(self, model_type):
		outname = os.path.splitext(self.transcript)[0] + '_timestamps'
		if os.path.isfile(outname + '.csv') or os.path.isfile(outname + '.srt'):
			self.logger.info("\tTimestamps for the transcript of this participant already exists. Skipping timestamps...")
		else:
			audio = whisper_timestamped.load_audio(self.audio_convert)
			model = whisper_timestamped.load_model(model_type, device="cpu")
			self.logger.debug(f"\tRunning whisper timestamps on participant {self.participant_name}.")
			result = whisper_timestamped.transcribe(model, audio, language="en")

			if self.whisper_time == "Segments":
				with open(outname + '.csv', 'w', newline='') as csvfile:
					write_csv(result["segments"], csvfile)
			else:
				with open(outname + '.srt', "w", encoding="utf-8") as srt:
					whisper_timestamped.make_subtitles.write_srt(self.flatten(result["segments"], "words"), file=srt)
			self.logger.info(f"\tWhisper timestamps (participant) {self.participant_name} has finished successfully.")

		if self.audio_interviewer is not None:
			outname = os.path.splitext(self.transcript)[0] + '_interviewer_timestamps'
			if os.path.isfile(outname + '.csv') or os.path.isfile(outname + '.srt'):
				self.logger.info("\tTimestamps for the transcript of this interviewer already exists. Skipping timestamps...")
			else:
				audio = whisper_timestamped.load_audio(self.audio_interviewer_convert)
				self.logger.debug(f"\tRunning whisper timestamps on interviewer {self.participant_name}.")
				result = whisper_timestamped.transcribe(model, audio, language="en")

				if self.whisper_time == "Segments":
					with open(outname + '.csv', 'w', newline='') as csvfile:
						write_csv(result["segments"], csvfile)
				else:
					with open(outname + '.srt', "w", encoding="utf-8") as srt:
						whisper_timestamped.make_subtitles.write_srt(self.flatten(result["segments"], "words"), file=srt)
			self.logger.info(f"\tWhisper timestamps (interviewer) {self.participant_name} has finished successfully.")
		else:
			if self.interviewer_analysis:
				self.logger.warning("\tNo interviewer audio found. Please provide interviewer audio or deactivate interviewer analysis.")


	def run_VAD(self):
		self.opensmile_audio = self.participant_dir + '\\' + self.participant_name + "_cleaned.wav"
		self.logger.debug(f"\tRunning VAD on participant {self.participant_name}.")
		self.interviewer_cleaned_audio = self.participant_dir + '\\' + self.participant_name + "_interviewer_cleaned.wav"
		self.part_timestamps, self.audio_convert = self.voice_activity_detection(self.audio)
		self.int_timestamps, self.audio_interviewer_convert  = self.voice_activity_detection(self.audio_interviewer)
		self.strip_audio(self.part_timestamps, self.audio_convert, self.opensmile_audio, False)
		self.strip_audio(self.int_timestamps, self.audio_interviewer_convert, self.interviewer_cleaned_audio, False)
		self.logger.info(f"\tVAD for participant {self.participant_name} and interviewer has finished successfully.")

	# Audio domain of pipeline: openSMILE, Whisper, semantic analysis.	
	def run_audio(self):
		self.log_dir = self.participant_dir + r'\logs'
		self.run_opensmile()
		self.run_whisper(self.whisper_model)
		if self.whisper_time != "None" and self.vad != "None":
			self.run_whisper_timestamped(self.whisper_model)
		self.run_nltk(self.transcript, f'{self.participant_dir}\\{self.participant_name}_nltk_results.txt', 
				f'{self.participant_dir}\\{self.participant_name}_sim_scores.csv',self.nltk_out, True)
		self.run_nltk(self.transcript_int, f'{self.participant_dir}\\{self.participant_name}_nltk_interviewer_results.txt', 
				f'{self.participant_dir}\\{self.participant_name}_interviewer_sim_scores.csv',
				self.nltk_out_interviewer, False)

	# Video domain of pipeline: OpenFace (including downstream analysis).
	def run_video(self, openface_out, video):
		if os.path.isfile(openface_out):
			self.logger.info("\tOutput files already exist. Skipping OpenFace...")
		else:
			if self.audio is None:
				self.logger.warning("\tNo audio files found. Video analysis will run without silence detection.")
			self.logger.info(f'\tStarting openface for participant {self.participant_name}. This may take a while.')
			self.log_dir = self.participant_dir + r'\logs'
			log_file = self.participant_name + "_openface.log"
			try:
				os.mkdir(self.log_dir)
			except OSError:
				self.logger.debug("\tThe log directory already exists. Existing logs will be overwritten.")
			log_path = f"{self.log_dir}\\{log_file}"
			command = f'{self.feat_detect} -f {video} -out_dir {self.participant_dir} -of {openface_out} -gaze -pose -aus -3Dfp > {log_path} 2>&1'
			self.logger.debug(f"\tRunning openface with command {command}")
			exit_c, output = subprocess.getstatusoutput(command)
			if exit_c != 0:
				self.logger.error(f'\tOpenface returned exit code {exit_c}. See log file for detailed error message.'.format(exit_c))
				sys.exit(1)

	# Facial analysis domain of pipeline: Analyse OpenFace output
	def facial_analysis(self, openface_output, int_flag):

		with open(openface_output, 'r') as file:
			reader = csv.reader(file)
			rows = list(reader)
			self.video_len = float(rows[-1][2].strip())
		if self.audio is not None and self.vad != 'None':
			if int_flag:
				edited_rows = self.check_silence_periods(self.int_timestamps, self.part_timestamps, rows)
			else:
				edited_rows = self.check_silence_periods(self.part_timestamps, self.int_timestamps, rows)
			audio_off = 0
		else:
			edited_rows = rows
			audio_off = 1
		au_count = self.count_binary_aus(edited_rows, audio_off)
		conf_scores = self.confidence_calculation(rows)
		if int_flag:
			self.openface_out_interviewer = f"{self.participant_dir}\\{self.participant_name}_interviewr_openface_out.csv"
			openface_out = self.openface_out_interviewer
		else:
			self.openface_out = f"{self.participant_dir}\\{self.participant_name}_openface_out.csv"
			openface_out = self.openface_out

		with open(openface_output, 'w', newline='') as file:
			writer = csv.writer(file)
			for row in edited_rows:
				writer.writerow(row)
		with open(openface_out, 'w', newline='') as f:
			writer = csv.writer(f)
			if not audio_off:
				writer.writerow(["Max confidence", "Min confidence", "Avg confidence", "AU01_c", "AU02_c",  "AU04_c" ,"AU05_c", "AU06_c", "AU07_c", "AU09_c", "AU10_c", "AU12_c",
						"AU14_c", "AU15_c", "AU17_c", "AU20_c", "AU23_c", "AU25_c", "AU26_c", "AU28_c", "AU45_c", "AU01_c_sil", "AU02_c_sil",  "AU04_c_sil" ,"AU05_c_sil", "AU06_c_sil", "AU07_c_sil", "AU09_c_sil", "AU10_c_sil", "AU12_c_sil",
						"AU14_c_sil", "AU15_c_sil", "AU17_c_sil", "AU20_c_sil", "AU23_c_sil", "AU25_c_sil", "AU26_c_sil", "AU28_c_sil", "AU45_c_sil", "AU01_c_sp", "AU02_c_sp",  "AU04_c_sp" ,"AU05_c_sp", "AU06_c_sp", "AU07_c_sp", "AU09_c_sp", "AU10_c_sp", "AU12_c_sp",
						"AU14_c_sp", "AU15_c_sp", "AU17_c_sp", "AU20_c_sp", "AU23_c_sp", "AU25_c_sp", "AU26_c_sp", "AU28_c_sp", "AU45_c_sp"])
				writer.writerow(conf_scores + au_count[0]+au_count[1]+au_count[2])
			else:
				writer.writerow(["Max confidence", "Min confidence", "Avg confidence", "AU01_c", "AU02_c",  "AU04_c" ,"AU05_c", "AU06_c", "AU07_c", "AU09_c", "AU10_c", "AU12_c",
						"AU14_c", "AU15_c", "AU17_c", "AU20_c", "AU23_c", "AU25_c", "AU26_c", "AU28_c", "AU45_c"])
				writer.writerow(conf_scores + au_count[0])

	# Participant sumary file of all results generated.
	def write_own_summary(self, int_flag):
		if int_flag:
			own_summary = f"{self.participant_dir}\\{self.participant_name}_interviewer_summary.csv"
			nltk_out = self.nltk_out_interviewer
			f_summary = self.f_interviewer_summary
			openface_out = self.openface_out_interviewer
		else:
			own_summary = f"{self.participant_dir}\\{self.participant_name}_summary.csv"
			nltk_out = self.nltk_out
			f_summary = self.f_summary
			openface_out = self.openface_out

		if os.path.isfile(own_summary):
			self.logger.info("\tParticipant's summary file alredy exists. Skipping...")
		header_list = []
		content_list = []
		if self.run_mode != "Video":
			with open(nltk_out, 'r') as f:
				reader = list(csv.reader(f))
				header_list += reader[0]
				content_list += (reader[1])
			with open(f_summary, 'r') as f:
				reader = list(csv.reader(f))
				header_list += (reader[0])
				content_list += (reader[1])
		if self.run_mode != "Audio":
			with open(openface_out, 'r') as f:
				reader = list(csv.reader(f))
				header_list += (reader[0])
				content_list += (reader[1])
		with open(own_summary, 'w', newline='') as f:
			writer = csv.writer(f)
			writer.writerow(header_list)
			writer.writerow(content_list)
		if int_flag:
			self.header_list_int = header_list
			self.content_list_int = content_list
		else:
			self.header_list = header_list
			self.content_list = content_list

	# Parse OpenFace output file and count activation of all binary action units. Counts for silent, non-silent and all periods.
	def count_binary_aus(self, rows, audio_off):
		au_count = [0]*18
		au_count_sil = [0]*18
		au_count_speech = [0]*18
		last_val = [0]*18
		for row in rows[1:]:
			for i in range(520, 537):
				if int(float(row[i].strip())) == 1:
					if last_val[i-520] == 0:
						au_count[i-520] += 1
						if not audio_off:
							# Get second to last element indicating participant speech 
							if int(float(row[-2].strip())) == 1:
								au_count_speech[i-520] += 1
							elif int(float(row[-2].strip())) == 0:
								au_count_sil[i-520] += 1
				last_val[i-520] = int(float(row[i].strip()))
		if not audio_off:
			return_cts = [au_count, au_count_sil, au_count_speech]
		else:
			return_cts = [au_count]
		for cts in return_cts:
				for i in range(len(cts)):
					cts[i] /= self.video_len
		return return_cts

	def confidence_calculation(self, rows):
		max_confidence = 0
		min_confidence = 100
		avg_confidence = 0
		count = 0
		for row in rows[1:]:
			confidence = int(float(row[3].strip())*100)
			if confidence > 0:
				if confidence > max_confidence:
					max_confidence = confidence
				if confidence < min_confidence:
					min_confidence = confidence
				avg_confidence += confidence
				count += 1

		avg_confidence = avg_confidence/count
		return [max_confidence, min_confidence, avg_confidence]

	# Helper function fo AU count. Updates each frame with participant speech value (0 for silent, 1 for speech).
	def check_silence_periods(self, part_timestamps, int_timestamps, rows):
		part_start = part_timestamps[0]['start']
		part_end = part_timestamps[0]['end']
		int_start = int_timestamps[0]['start']
		int_end = int_timestamps[0]['end']
		part_index = 1
		int_index = 1
		modified_rows = []
		rows[0].extend(["participant speech"])
		rows[0].extend(["interviewer speech"])
		modified_rows.append(rows[0])
		part_end_flag = False
		int_end_flag = False
		for row in rows[1:]:
			timestamp = float(row[2])
			if part_start <= timestamp < part_end and part_end_flag is not True:
				speaking_flag = 1
			else:
				speaking_flag = 0
			if int_start <= timestamp < int_end and int_end_flag is not True:
				int_flag = 1
			else:
				int_flag = 0
			row.append(str(speaking_flag))
			row.append(str(int_flag))
			modified_rows.append(row)

			if timestamp > part_end:
				if part_timestamps[-1]['end'] == part_end:
					part_end_flag = True
				else:
					part_start = part_timestamps[part_index]['start']
					part_end = part_timestamps[part_index]['end']
					part_index += 1
			if timestamp > int_end:
				if int_timestamps[-1]['end'] == int_end:
					int_end_flag = True
				else:
					int_start = int_timestamps[int_index]['start']
					int_end = int_timestamps[int_index]['end']
					int_index += 1
		return modified_rows

	# Convert all text to lower case.
	def preprocess_text(self, text, nlp):     
		doc = nlp(text.lower())
		tokens = [token.lemma_ for token in doc]
		return ' '.join(tokens)
	
	# Semantic analysis usng spaCy, nltk, ...
	def run_nltk(self, transcript, output_file, lsa_output, csv_output, ratio_flag):
		if not(os.path.isfile(transcript)):
			self.logger.error("\tTranscript could not be found. Do not rename or move it.")
			sys.exit(1)
		file = transcript
		with open(file, 'r') as f:
			all_text = f.read()
		self.logger.info(f"\tStarting semantic analysis for participant {self.participant_name}.")

		# Extract all sentences
		nlp = spacy.load("en_core_web_sm")
		sentences = [sent.text.strip() for sent in nlp(all_text).sents]

		# Initializing the Sentence Transformer model
		model = SentenceTransformer('all-mpnet-base-v2')

		# Encoding the sentences to obtain their embeddings
		X = model.encode(sentences, convert_to_tensor=True)
		similarity_matrix = cosine_similarity(X)
		with open(lsa_output, "w", newline="") as f:
			writer = csv.writer(f)
			head = ["Sentence"] + [sentence for sentence in sentences]
			writer.writerow(head)
			for i, sentence in enumerate(sentences):
				row = [sentence] + [similarity_matrix[i][j] for j in range(len(sentences))]
				writer.writerow(row)
		
		# Fragment into sentences, calculate part/interviewer speech ratio.
		sentences = nltk.tokenize.sent_tokenize(all_text)
		num_sent = len(sentences)
		words_num_part = len(nltk.tokenize.word_tokenize(all_text))
		file = transcript
		if ratio_flag:
			if not(os.path.isfile(self.transcript_int)):
				self.logger.error("\tTranscript could not be found. Do not rename or move it.")
				sys.exit(1)
			if self.audio_interviewer is not None:
				with open(self.transcript_int, 'r') as f:
					int_text = f.read()
				words_num_int = len(nltk.tokenize.word_tokenize(int_text))
				total_words = words_num_int + words_num_part
				word_ratio = words_num_part/total_words
			else:
				word_ratio = "N/A"
		else:
			word_ratio = "N/A"
				
		neighbour_scores = []
		for i in range(num_sent-1):
			for j in range(num_sent -1):
				if abs(i-j) == 1:
					neighbour_scores.append(similarity_matrix[i][j])
		avg_sim_score = 0
		if neighbour_scores:
			max_sim_score = max(neighbour_scores)
			min_sim_score = min(neighbour_scores)
			avg_sim_score = sum(neighbour_scores)/len(neighbour_scores)
			var_sim_score = np.var(neighbour_scores)

		# Do sentiment scores per sentence
		sia = SentimentIntensityAnalyzer()
		sentiment_scores = [sia.polarity_scores(sentence) for sentence in sentences]
		overall_sentiment = {
		'neg': sum(score['neg'] for score in sentiment_scores) / len(sentiment_scores),
		'neu': sum(score['neu'] for score in sentiment_scores) / len(sentiment_scores),
		'pos': sum(score['pos'] for score in sentiment_scores) / len(sentiment_scores),
		'compound': sum(score['compound'] for score in sentiment_scores) / len(sentiment_scores)}
		overall_sentiment_list = [val for val in overall_sentiment.values()]

		# Remove punctuation
		translator = str.maketrans('','', string.punctuation)
		fixed_text = all_text.translate(translator)

		# Get rid of contractions like I'm, it's, can't ...
		words = nltk.tokenize.word_tokenize(fixed_text)
		expanded_text = [contractions.fix(word) for word in words]

		merged_text = ' '.join(expanded_text)

		# Do POS and dependency tagging.
		nlp = spacy.load('en_core_web_sm')
		doc = nlp(merged_text)
		with open(output_file, 'w') as f:
			pos_counter = {}
			dep_counter = {}
			tag_counter = {}
			for i, token in enumerate(doc):
				f.write(f"Token: {token.text}\tPOS: {token.pos_}\tTags: {token.tag_}\tDep: {token.dep_}\n")
				pos= token.pos_
				if pos == "PRON":
					match token.morph.get("Person"):
						case ['1']:
							pos = "p1"
						case ['2']:
							pos = "p2"
						case ['3']:
							pos = "p3"
				if pos in pos_counter:
					pos_counter[pos] += 1
				else:
					pos_counter[pos] = 1
				tag= token.tag_
				if tag in tag_counter:
					tag_counter[tag] += 1
				else:
					tag_counter[tag] = 1
				dep = token.dep_
				if dep in dep_counter:
					dep_counter[dep] += 1
				else:
					dep_counter[dep] = 1
			f.write(f"\nPOS counts: {pos_counter}")
			f.write(f"\nTAG counts: {tag_counter}")
			f.write(f"\nDependency counter: {dep_counter}")
		with open(csv_output, 'w', newline='') as f:
			writer = csv.writer(f)
			POS_tags = ["ADJ", "ADP", "ADV", "AUX", "CONJ", "CCONJ", "DET", "INTJ", "NOUN" , "NUM", "PART", "PRON", "p1", "p2", "p3", "PROPN", "PUNCT", "SCONJ", "SYM", "X"]
			TAG_tags = ["AFX","CC", "CD", "DT", "EX", "FW", "HYPH", "IN", "JJ", "JJR", "JJS",
            "LS", "MD", "NIL", "NN", "NNP", "NNPS", "NNS", "PDT",
            "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "TO", "UH", "VB", "VBD", "VBG",
            "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB", "SP", "ADD",
            "NFP", "GW", "XX"]
			DEP_tags = ["ROOT", "acl", "acomp", "advcl", "advmod", "agent", "amod", "appos", "attr",
			"aux", "auxpass", "case", "cc", "ccomp", "compound", "conj", "cop", "csubj",
			"csubjpass", "dative", "dep", "det", "dobj", "expl", "intj", "mark", "meta",
			"neg", "nmod", "npadvmod", "nsubj", "nsubjpass", "nummod", "oprd", "parataxis",
			"punct", "quantmod", "relcl"]

			POS_val_list = []
			for key in POS_tags:
				val = pos_counter.get(key,0)
				POS_val_list.append(val)

			TAG_val_list = []
			for key in TAG_tags:
				val = tag_counter.get(key,0)
				TAG_val_list.append(val)
			
			DEP_val_list = []
			for key in DEP_tags:
				val = dep_counter.get(key,0)
				DEP_val_list.append(val)

			data = ["max sentence len", "avg sentence len", "total num sent", "neg sent", "neu sent",
		   "pos sent", "comp sent","max sim score", "min sim score", "avg sim score", 
		   "var sim score", "part_words/total_words"]
			col_names = data + POS_tags + TAG_tags + DEP_tags
			head = [col for col in col_names]
			avg_sentence_length = sum(len(sent.split()) for sent in sentences) / len(sentences)
			max_sentence_length = max(len(sent.split()) for sent in sentences)
			writer.writerow(head)
			POS_val_list_norm = [x / len(sentences) for x in POS_val_list]
			TAG_val_list_norm = [x / len(sentences) for x in TAG_val_list]
			DEP_val_list_norm = [x / len(sentences) for x in DEP_val_list]
			body = [max_sentence_length] + [avg_sentence_length] + [len(sentences)] + overall_sentiment_list + \
			[max_sim_score]  + [min_sim_score]  + [avg_sim_score]  + [var_sim_score] + \
			[word_ratio]+ POS_val_list_norm + TAG_val_list_norm + DEP_val_list_norm
			writer.writerow(body)
			
		self.logger.info("\tSemantic analysis completed.")

	def convert_video_audio(self, path, save_path):
		command = f'ffmpeg -y -i "{path}" -vn "{save_path}"'
		exit_c, output = subprocess.getstatusoutput(command)
		if exit_c != 0:
			print(f'\tffmpeg returned exit code {exit_c}. See log file for detailed error message.'.format(exit_c))
			sys.exit(1)

	def minimum_silences(self, timestamps):
		timestamps_new = [timestamps[0]]
		for index in range(1,len(timestamps)):
			timestamp = timestamps[index]
			if timestamp['start'] - timestamps_new[-1]['end'] < 1.0:
				timestamps_new[-1] = {'start':timestamps_new[-1]['start'], 'end':timestamp['end']}
			else:
				timestamps_new.append(timestamp)
		return timestamps_new
	
	def voice_activity_detection(self, filepath):
		model = silero_vad.load_silero_vad()
		audio_codec = "_converted.wav"
		convert_path = os.path.splitext(filepath)[0] + audio_codec
		if os.path.isfile(f"{convert_path}") == False:
			self.convert_video_audio(filepath, convert_path)
			self.sampling_rate, data=read_wav(convert_path)
			if self.sampling_rate not in [8000, 16000, 32000, 48000]:
				self.logger.info("\tSample rate not supported by voice activity detection. " \
				"Automatic resampling to 16kHz.")
				origin_num_samples, _  = data.shape
				new_samps = int(origin_num_samples * 16000/self.sampling_rate)
				# resampling
				target_audio_scipy = resample(data[:,0], new_samps).astype(int)
				target_audio_scipy = np.array(target_audio_scipy, np.int16)
				write_wav(convert_path, 16000, target_audio_scipy)
				self.sampling_rate = 16000
		else:
			self.sampling_rate, data=read_wav(convert_path)

		wav = silero_vad.read_audio(convert_path, sampling_rate=self.sampling_rate)
		speech_timestamps = silero_vad.get_speech_timestamps(wav, model, return_seconds=True,
													   sampling_rate=self.sampling_rate,
													   min_silence_duration_ms = 1000,
													   min_speech_duration_ms = 500)
		
		speech_timestamps = self.minimum_silences(speech_timestamps)
		return [speech_timestamps, convert_path]

	def strip_audio(self, speech_timestamps, convert_path, save_path, drop_flag):
		wav = silero_vad.read_audio(convert_path, sampling_rate=self.sampling_rate)
		timestamp_samples = self.seconds_to_samples(speech_timestamps, self.sampling_rate)
		if drop_flag:
			silero_vad.save_audio(save_path, silero_vad.utils_vad.drop_chunks(timestamp_samples, wav), sampling_rate=self.sampling_rate)
		else:
			silero_vad.save_audio(save_path, silero_vad.collect_chunks(timestamp_samples, wav), sampling_rate=self.sampling_rate)

	def seconds_to_samples(self, timestamps: list[dict], sampling_rate: int) -> list[dict]:
		"""Convert coordinates expressed in seconds to sample coordinates.
		"""
		return [{
			'start': round(stamp['start']) * sampling_rate,
			'end': round(stamp['end']) * sampling_rate
			} for stamp in timestamps]

	# Crop the video. This removes the interviewer's face so that OpenFace reliably works on the participant only.
	def cut_video(self):
		self.log_dir = self.participant_dir + r'\logs'
		log_file = self.participant_name + "_ffmpeg.log"
		cropped_video = f'{self.participant_dir}\\{self.participant_name}_video.mp4'
		try:
			os.mkdir(self.log_dir)
		except OSError:
			self.logger.debug("\tThe log directory already exists. Existing logs will be overwritten.")
		log_path = f"{self.log_dir}\\{log_file}"
		if os.path.isfile(cropped_video):
			self.logger.info("\tVideo has already been cropped. Skipping step and continuing...")
		else:
			#command = f'{self.ffmpeg_path} -i {self.video} -vf "crop=in_w/2:in_h:in_w/2:0"  {cropped_video}>> {log_path} 2>&1'
			command = f'ffmpeg -i {self.video} -vf "crop=in_w/2:in_h:in_w/2:0"  {cropped_video}>> {log_path} 2>&1'
			self.logger.debug("\tRunning ffmpeg cropping on participant file using {}.".format(command))
			exit_c, output = subprocess.getstatusoutput(command)
			if exit_c != 0:
				self.logger.error(f'\tffmpeg returned exit code {exit_c}. See log file for detailed error message.'.format(exit_c))
				sys.exit(1)
		self.video = cropped_video
		# Crop interviewer when flag is set
		if self.interviewer_analysis:
			cropped_video = f'{self.participant_dir}\\{self.participant_name}_interviewer_video.mp4'
		log_path = f"{self.log_dir}\\{log_file}"
		if os.path.isfile(cropped_video):
			self.logger.info("\tInterviewer video has already been cropped. Skipping step and continuing...")
		else:
			#command = f'{self.ffmpeg_path} -i {self.video} -vf "crop=in_w/2:in_h:in_w/2:0"  {cropped_video}>> {log_path} 2>&1'
			command = f'ffmpeg -i {self.video} -vf "crop=in_w/2:in_h:0:0"  {cropped_video}>> {log_path} 2>&1'
			self.logger.debug("\tRunning ffmpeg cropping on interviewer file using {}.".format(command))
			exit_c, output = subprocess.getstatusoutput(command)
			if exit_c != 0:
				self.logger.error(f'\tffmpeg returned exit code {exit_c}. See log file for detailed error message.'.format(exit_c))
				sys.exit(1)
		self.interviewer_video = cropped_video


	# Run either/both audio or/and video domains, acll summary generation function.
	def run_pipe(self):
		if self.vad != 'None':
			self.run_VAD()
		if self.run_mode != "Video":
			self.run_audio()
		if self.run_mode != "Audio":
			if not self.no_cut:
				self.cut_video()
			
			openface_out = f'{self.participant_dir}\\{self.participant_name}.csv'
			self.run_video(openface_out, self.video)
			self.facial_analysis(openface_out, False)
			if self.interviewer_analysis:
				openface_out = f'{self.participant_dir}\\{self.participant_name}_interviewer.csv'
				self.run_video(openface_out, self.interviewer_video)
				self.facial_analysis(openface_out, True)

		self.write_own_summary(False)
		if self.interviewer_analysis:
			self.write_own_summary(True)
		self.logger.info(f"\tPipeline completed for {self.participant_name}.")
		if self.interviewer_analysis:
			return [self.summary, self.interviewer_summary, self.header_list, self.content_list, self.header_list_int, self.content_list_int, self.participant_name]
		else:
			return [self.summary, self.header_list, self.content_list, self.participant_name]

## Class used only for initial parsing. Splits input up so it can be divided among processes.
class pipeParser:
	def parse_args(self, args):
		if not os.path.isdir(args.interviews):
			raise Exception(f"\tFolder {args.interviews} does not exist. Check that you provided the correct path.")
		if args.overwrite:
			clear_data(args.interviews)
		all_args = []
		video = None
		audio = None
		int_audio = None
		for participant_dir in os.listdir(args.interviews):	
			participant_dir_path = os.path.join(args.interviews, participant_dir)
			if not participant_dir.endswith(".csv"):
				for f in os.listdir(participant_dir_path):
					file = os.path.join(participant_dir_path, f)
					if (f.endswith('2.mp4') or f.endswith('2.m4a')):
						audio = file
					elif (f.endswith('1.mp4') or f.endswith('1.m4a')):
						int_audio = file
					elif f.find('gvo') != -1 and f.endswith('.mp4'):
						video = file
				if args.mode != video and audio == None:
					raise Exception(f"\t{participant_dir} does not have an audio file. Either remove this folder or provide a video file. See README for information on naming files.")
				if args.mode != audio and video == None:
					raise Exception(f"\t{participant_dir} does not have a video file. Either remove this folder or provide an audio file. See README for information on naming files.")
				if args.mode != video and int_audio == None:
					print(f"{participant_dir} does not have an interviewer audio file. The pipeline will skip all analysis requiring this file.")
				all_args.append([audio, video, int_audio, args.verbosity, args.mode, args.vad, args.whisper_time, args.no_cut, args.whisper_model, args.interviewer_analysis])
		return len(all_args), all_args

# Function executed by each process, returns summary list, added to shared queue.
def process_func(queue, my_pipe, args):
	my_pipe.parse_args(args)
	result = my_pipe.run_pipe()
	queue.put(result)

# If 'overwrite' functionality specified, deletes all non-necessary files.
def clear_data(path):
    try:
        for root, dirs, files in os.walk(path):
            if 'logs' in dirs:
                log_dir_path = os.path.join(root, 'logs')
                for log_root, log_dirs, log_files in os.walk(log_dir_path):
                    for log_file in log_files:
                        log_file_path = os.path.join(log_root, log_file)
                        os.remove(log_file_path)
                    for log_dir in log_dirs:
                        log_subdir_path = os.path.join(log_root, log_dir)
                        print(f"Deleting directory: {log_subdir_path}")
                        shutil.rmtree(log_subdir_path)
                os.rmdir(log_dir_path)
            for filename in files:
                file_path = os.path.join(root, filename)
                if (not (filename.endswith('mp4')) and not (filename.endswith('m4a'))):
                    os.remove(file_path)
    except Exception as e:
        print(f"An error occurred: {e}")\

# Function only executed by 'reader' process, gets items from queue as long as pipeline is not over, writes out to file.
def make_summary(queue):
		while True:
			first = queue.get()
			if first == None:
				return False
			else:
				if len(first) == 4:
					file_path = first[0]
					int_flag = False
				else:
					file_path = first[0]
					file_int_path = first[1]
					int_flag = True
				break
		if int_flag:
			with (open(file_path, 'w', newline='') as f, open(file_int_path, 'w', newline='') as f_int):
				writer = csv.writer(f)
				writer_int = csv.writer(f_int)
				writer.writerow(["ID"] + first[2])
				writer.writerow([first[6]] + first[3])
				writer_int.writerow(["ID"] + first[4])
				writer_int.writerow([first[6]] + first[5])
				while True:
					item = queue.get()
					if item == None:
						break
					else:
						writer.writerow([item[6]] + item[3])
						writer_int.writerow([item[6]] + item[5])
		else:
			with open(file_path, 'w', newline='') as f:
				writer = csv.writer(f)
				writer.writerow(["ID"] + first[1])
				writer.writerow([first[3]] + first[2])
				while True:
					item = queue.get()
					if item == None:
						break
					else:
						writer.writerow([item[3]] + item[2])

## GUI class.
class DigBioWindow(QMainWindow):
	def __init__(self):
		super(DigBioWindow, self).__init__() # Call the inherited classes __init__ method
		loadUi('DigBio.ui', self) # Load the .ui file
		self.show() # Show the GUI
		self.setWindowTitle('DigBio 1.3')
		self.setFixedSize(1100, 550)
		
		# Set default values
		self.pathFolder.setText(os.path.expanduser("~"))
		self.modeBox.setCurrentText("All")
		self.verbosityBox.setCurrentText("3")
		self.whisperBox.setCurrentText("base")
		self.vadBox.setCurrentText("Both")
		self.timestampBox.setCurrentText("None")
		self.nocutBox.setChecked(False)
		self.overwriteBox.setChecked(False)
		self.interviewer.setChecked(False)

        # Set GUI signals
		self.pathButton.clicked.connect(self.pathButtonClicked)
		self.startButton.clicked.connect(self.startButtonClicked)

	def pathButtonClicked(self):
		self.pathFolder.setText(str(QFileDialog.getExistingDirectory(
            self,"Select Directory",self.pathFolder.text(),
            QFileDialog.ShowDirsOnly)))

	def startButtonClicked(self):
		all_args = Namespace(interviews=self.pathFolder.text(), 
                             mode=self.modeBox.currentText(),
                             verbosity=self.verbosityBox.currentText(),
                             overwrite=self.overwriteBox.isChecked(),
							 vad = self.vadBox.currentText(),
							 whisper_time = self.timestampBox.currentText(), 
                             no_cut=self.nocutBox.isChecked(),
                             whisper_model=self.whisperBox.currentText(),
							 interviewer_analysis=self.interviewer.isChecked())
		
		pipeParserInt = pipeParser()
		num_procs, parsed_all_args = pipeParserInt.parse_args(all_args)
		my_pipes = [pipe() for _ in range(num_procs)]
		summary_queue = multiprocessing.Queue()
		processes = []
		summary_process = multiprocessing.Process(target=make_summary, 
                                                  args=(summary_queue,))
		summary_process.start() # Reader process
		for i in range(num_procs):
			process = multiprocessing.Process(target=process_func,
                                              args=(summary_queue, my_pipes[i], 
                                              parsed_all_args[i])) # Writer processes
			processes.append(process)
			process.start()
		for process in processes:
			process.join()
		summary_queue.put(None)
		summary_process.join()
		print("INFO:\tAll participants complete. You can close the windows now.")
		self.close()


if __name__ == '__main__':
    multiprocessing.freeze_support()
    app = QApplication(sys.argv)
    window = DigBioWindow()
    app.exec_()