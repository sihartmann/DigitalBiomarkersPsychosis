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
import regex as re
from nltk.sentiment import SentimentIntensityAnalyzer
import contractions
import spacy
import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from argparse import Namespace
import PyQt5
from PyQt5.QtWidgets import *
from PyQt5.uic import *
import cv2
import numpy as np
from ffpyplayer.player import MediaPlayer
import plotly.express as px
import io 
from PIL import Image

VERSION = '1.0 with GUI'

## Main processing class.
class pipe:

	# Parse arguments and set up logging.
	def parse_args(self, args):
		self.audio = args[0].replace('/','\\')
		self.video = args[1].replace('/', '\\')
		self.audio_interviewer = args[2].replace('/', '\\')
		self.loglevel = args[3]
		self.run_mode = args[4]
		self.no_cut = args[5]
		self.whisper_model = args[6]
		if self.audio is not None:
			self.participant_dir = os.path.dirname(self.audio)
		elif self.video is not None:
			self.participant_dir = os.path.dirname(self.video)
		else:
			raise Exception("You must provide video and/or audio files.")
		self.participant_name = os.path.basename(self.participant_dir)
		self.nltk_out = f'{self.participant_dir}\\{self.participant_name}_nltk_results.csv'
		self.summary =  f'{self.participant_dir}\\..\\all_summary.csv'
		stderrhandler = logging.StreamHandler()
		filehandler = logging.FileHandler("DigBio.log")
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
		self.ffmpeg_path =  r"Whisper-OpenAI_r136\Whisper-OpenAI\ffmpeg.exe"
		if not os.path.isfile(self.ffmpeg_path):
			self.logger.error("\tMissing dependency: ffmpeg.")
			sys.exit(1)
		self.whisper_path = r"Whisper-OpenAI_r136\Whisper-OpenAI\whisper.exe"
		if not os.path.isfile(self.whisper_path):
			self.logger.error("\tMissing dependency: whisper.")
			sys.exit(1)
		self.feat_detect =  r"OpenFace_2.2.0_win_x64\FeatureExtraction.exe"
		if not os.path.isfile(self.feat_detect):
			self.logger.error("\tMissing dependency: openface.")
			sys.exit(1)

	# Parse through Interviews directory and return list of all participant names.
	def get_participants(self, directory):
		participants = list()
		for entry in os.listdir(directory):
			if os.path.isdir(os.path.join(directory, entry)):
				participants.append(entry)
		return participants
	
	# Cut silences from audio using ffmpeg. This improves openSMILE results.
	def clean_opensmile_audio(self):
		self.opensmile_audio = self.participant_dir + '\\' + self.participant_name + "_cleaned.wav"
		log_file = self.participant_name + "_ffmpeg.log"
		try:
			os.mkdir(self.log_dir)
		except OSError:
			self.logger.info("The log directory already exists. Existing logs will be overwritten.")
		if os.path.isfile(f"{self.opensmile_audio}") == False:
			command = f'{self.ffmpeg_path}  -i {self.audio} -af "silenceremove=start_periods=0:start_duration=1:stop_periods=-1:stop_duration=5" {self.opensmile_audio} > {self.log_dir}\\{log_file} 2>&1'
			self.logger.debug("\tRunning ffmpeg on participant file using {}.".format(command))
			exit_c, output = subprocess.getstatusoutput(command)
			if exit_c != 0:
				self.logger.error(f'\tffmpeg returned exit code {exit_c}. See log file for detailed error message.'.format(exit_c))
				sys.exit(1)
			self.logger.info("\tffmpeg has finished successfully.")
		else:
			self.logger.info("\tFiles have already been cleaned. Skipping step and continuing...")

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
			features = smile.process_file(self.opensmile_audio)
			df = pd.DataFrame(features, columns=smile.feature_names)
			df.to_csv(f_individual, index=False)

			smile_summary = opensmile.Smile(
				feature_set = opensmile.FeatureSet.eGeMAPSv02
			)
			features = smile_summary.process_file(self.opensmile_audio)
			df = pd.DataFrame(features, columns=smile_summary.feature_names)
			df.to_csv(self.f_summary, index=False)
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
		if os.path.isfile(self.transcript):
			self.logger.info("\tTranscript for this participant already exists. Skipping transcription...")
		else:
			linux_audio = self.audio.replace('\\','/')
			command = f'{self.whisper_path} -f txt --model {model} audio {linux_audio} --hallucination_silence_threshold 1 --word_timestamps True --output_dir {self.participant_dir}> {self.log_dir}\\{log_file} 2>&1'
			self.logger.debug("\tRunning whisper on participant file using {}.".format(command))
			subprocess.check_output(command, shell=True)
			command = f'ren {self.audio[:-4]}.txt {self.transcript_clean}'
			subprocess.check_output(command, shell=True)
			self.logger.info(f"\twhisper (participant {self.participant_name}) has finished successfully.")
		if self.audio_interviewer is not None:
			if os.path.isfile(self.transcript_int):
				self.logger.info("\tTranscript for this interviewer already exists. Skipping transcription...")
			else:
				linux_audio = self.audio_interviewer.replace('\\','/')
				command = f'{self.whisper_path} -f txt --model {model} audio {linux_audio} --hallucination_silence_threshold 1 --word_timestamps True --output_dir {self.participant_dir}> {self.log_dir}\\{log_file} 2>&1'
				self.logger.debug("\tRunning whisper on interviewer file using {}.".format(command))
				subprocess.check_output(command, shell=True)
				command = f'ren {self.audio_interviewer[:-4]}.txt {self.transcript_int_clean}'
				subprocess.check_output(command, shell=True)
				self.logger.info(f"\twhisper (interviewer) {self.participant_name} has finished successfully.")
	
	# Audio domain of pipeline: openSMILE, Whisper, semantic analysis.
	def run_audio(self):
		self.clean_opensmile_audio()
		self.run_opensmile() 
		self.run_whisper(self.whisper_model)
		self.run_nltk(f'{self.participant_dir}\\{self.participant_name}_nltk_results.txt', f'{self.participant_dir}\\{self.participant_name}_sim_scores.csv',self.nltk_out)

	# Video domain of pipeline: OpenFace (including downstream analysis).
	def run_video(self):
		openface_out = f'{self.participant_dir}\\{self.participant_name}.csv'
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
			command = f'{self.feat_detect} -f {self.video} -out_dir {self.participant_dir} -of {self.participant_name} > {log_path} 2>&1'
			self.logger.debug(f"\tRunning openface with command {command}")
			exit_c, output = subprocess.getstatusoutput(command)
			if exit_c != 0:
				self.logger.error(f'\tOpenface returned exit code {exit_c}. See log file for detailed error message.'.format(exit_c))
				sys.exit(1)
		with open(openface_out, 'r') as file:
			reader = csv.reader(file)
			rows = list(reader)
			self.video_len = float(rows[-1][2].strip())
		if self.audio is not None:
			part_silence_list = self.parse_silence_file(self.output_sr)
			edited_rows = self.check_silence_periods(part_silence_list, rows)
			audio_off = 0
		else:
			edited_rows = rows
			audio_off = 1
		au_count = self.count_binary_aus(edited_rows, audio_off)
		self.openface_out = f"{self.participant_dir}\\{self.participant_name}_openface_out.csv"
		with open(f'{self.participant_dir}\\{self.participant_name}.csv', 'w', newline='') as file:
			writer = csv.writer(file)
			for row in edited_rows:
				writer.writerow(row)
		with open(self.openface_out, 'w', newline='') as f:
			writer = csv.writer(f)
			if not audio_off:
				writer.writerow([ "AU01_c", "AU02_c",  "AU04_c" ,"AU05_c", "AU06_c", "AU07_c", "AU09_c", "AU10_c", "AU12_c",
						"AU14_c", "AU15_c", "AU17_c", "AU20_c", "AU23_c", "AU25_c", "AU26_c", "AU28_c", "AU45_c", "AU01_c_sil", "AU02_c_sil",  "AU04_c_sil" ,"AU05_c_sil", "AU06_c_sil", "AU07_c_sil", "AU09_c_sil", "AU10_c_sil", "AU12_c_sil",
						"AU14_c_sil", "AU15_c_sil", "AU17_c_sil", "AU20_c_sil", "AU23_c_sil", "AU25_c_sil", "AU26_c_sil", "AU28_c_sil", "AU45_c_sil", "AU01_c_sp", "AU02_c_sp",  "AU04_c_sp" ,"AU05_c_sp", "AU06_c_sp", "AU07_c_sp", "AU09_c_sp", "AU10_c_sp", "AU12_c_sp",
						"AU14_c_sp", "AU15_c_sp", "AU17_c_sp", "AU20_c_sp", "AU23_c_sp", "AU25_c_sp", "AU26_c_sp", "AU28_c_sp", "AU45_c_sp"])
				writer.writerow(au_count[0]+au_count[1]+au_count[2])
			else:
				writer.writerow([ "AU01_c", "AU02_c",  "AU04_c" ,"AU05_c", "AU06_c", "AU07_c", "AU09_c", "AU10_c", "AU12_c",
						"AU14_c", "AU15_c", "AU17_c", "AU20_c", "AU23_c", "AU25_c", "AU26_c", "AU28_c", "AU45_c"])
				writer.writerow(au_count[0])

	# Participant sumary file of all results generated.
	def write_own_summary(self):
		own_summary = f"{self.participant_dir}\\{self.participant_name}_summary.csv"
		if os.path.isfile(own_summary):
			self.logger.info("\tParticipant's summary file alredy exists. Skipping...")
		self.header_list = []
		self.content_list = []
		if self.run_mode != "video":
			with open(self.nltk_out, 'r') as f:
				reader = list(csv.reader(f))
				self.header_list += reader[0]
				self.content_list += (reader[1])
			with open(self.f_summary, 'r') as f:
				reader = list(csv.reader(f))
				self.header_list += (reader[0])
				self.content_list += (reader[1])
		if self.run_mode != "audio":
			with open(self.openface_out, 'r') as f:
				reader = list(csv.reader(f))
				self.header_list += (reader[0])
				self.content_list += (reader[1])
		with open(own_summary, 'w', newline='') as f:
			writer = csv.writer(f)
			writer.writerow(self.header_list)
			writer.writerow(self.content_list)

	# Create list of silent periods from participant audio. Used for OpenFace analysis.
	def parse_silence_file(self, file_path):
		silence_list = []
		with open(file_path, 'r') as file:
			for line in file:
					match = re.search(r'silence_end: (\d+\.\d+).*silence_duration: (\d+\.\d+)', line)
					if match:
						end = float(match.group(1))
						dur = float(match.group(2))
						start = end - dur
						silence_list += [start,end]
		return silence_list

	# Parse OpenFace output file and count activation of all binary action units. Counts for silent, non-silent and all periods.
	def count_binary_aus(self, rows, audio_off):
		au_count = [0]*18
		au_count_sil = [0]*18
		au_count_speech = [0]*18
		last_val = [0]*18
		for row in rows[1:]:
			for i in range(696, 714):
				if int(float(row[i].strip())) == 1:
					if last_val[i-696] == 0:
						au_count[i-696] += 1
						if not audio_off:
							if int(float(row[-1].strip())) == 1:
								au_count_speech[i-696] += 1
							elif int(float(row[-1].strip())) == 0:
								au_count_sil[i-696] += 1
				last_val[i-696] = int(float(row[i].strip()))
		if not audio_off:
			return_cts = [au_count, au_count_sil, au_count_speech]
		else:
			return_cts = [au_count]
		for cts in return_cts:
				for i in range(len(cts)):
					cts[i] /= self.video_len
		return return_cts

	# Helper function fo AU count. Updates each frame with participant speech value (0 for silent, 1 for speech).
	def check_silence_periods(self, silence_list, rows):
		current_start = silence_list[0]
		current_end = silence_list[1]
		period_index = 2 
		modified_rows = []
		rows[0].extend(["participant speech"])
		modified_rows.append(rows[0])
		for row in rows[1:]:
			timestamp = float(row[2])
			end_flag = False
			if current_start <= timestamp < current_end and end_flag is not True:
				silence_flag = 0
			else:
				silence_flag = 1
			row.append(str(silence_flag))
			modified_rows.append(row)
			if timestamp > current_end:
				if silence_list[-1] == current_end:
					end_flag = True
				else:
					current_start = silence_list[period_index]
					current_end = silence_list[period_index + 1]
					period_index += 2
		return modified_rows

	# Convert all text to lower case.
	def preprocess_text(self, text, nlp):     
		doc = nlp(text.lower())
		tokens = [token.lemma_ for token in doc]
		return ' '.join(tokens)
	
	# Smenatic analysis usng spaCy, nltk, ...
	def run_nltk(self, output_file, lsa_output, csv_output):
		if not(os.path.isfile(self.transcript) and os.path.isfile(self.transcript_int)):
			self.logger.error("\tTranscript could not be found. Do not rename or move it.")
			sys.exit(1)
		file = self.transcript
		with open(file, 'r') as f:
			all_text = f.read()
		self.logger.info(f"\tStarting semantic analysis for participant {self.participant_name}.")

		# Text to lowercase conversion
		nlp = spacy.load("en_core_web_sm")
		sentences = [self.preprocess_text(sent.text,nlp) for sent in nlp(all_text).sents]

		# lsa analysis
		vectorizer = TfidfVectorizer()
		X = vectorizer.fit_transform(sentences)
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
		file = self.transcript
		if self.audio_interviewer is not None:
			with open(self.transcript_int, 'r') as f:
				int_text = f.read()
			words_num_int = len(nltk.tokenize.word_tokenize(int_text))
			total_words = words_num_int + words_num_part
			word_ratio = words_num_part/total_words
		else:
			word_ratio = "N/A"
		neighbour_scores = []
		for i in range(num_sent-1):
			for j in range(num_sent -1):
				if abs(i-j) == 1:
					neighbour_scores.append(similarity_matrix[i][j])
		avg_sim_score = 0
		if neighbour_scores:
			avg_sim_score = sum(neighbour_scores)/len(neighbour_scores)

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
			for i, token in enumerate(doc):
				f.write(f"Token: {token.text}\tPOS: {token.pos_}\tDep: {token.dep_}\n")
				pos= token.pos_
				if pos in pos_counter:
					pos_counter[pos] += 1
				else:
					pos_counter[pos] = 1
				dep = token.dep_
				if dep in dep_counter:
					dep_counter[dep] += 1
				else:
					dep_counter[dep] = 1
			f.write(f"\nPOS counts: {pos_counter}")
			f.write(f"\nDependency counter: {dep_counter}")
		with open(csv_output, 'w', newline='') as f:
			writer = csv.writer(f)
			POS_tags = ["ADJ", "ADP", "ADV", "AUX", "CONJ", "CCONJ", "DET", "INTJ", "NOUN" , "NUM", "PART", "PROUN", "PROPN", "PUNCT", "SCONJ", "SYM", "X"]
			DEP_tags = ["ROOT", "acl", "acomp", "advcl", "advmod", "agent", "amod", "appos", "attr",
			"aux", "auxpass", "case", "cc", "ccomp", "compound", "conj", "cop", "csubj",
			"csubjpass", "dative", "dep", "det", "dobj", "expl", "intj", "mark", "meta",
			"neg", "nmod", "npadvmod", "nsubj", "nsubjpass", "nummod", "oprd", "parataxis",
			"punct", "quantmod", "relcl"]

			POS_val_list = []
			for key in POS_tags:
				val = pos_counter.get(key,0)
				POS_val_list.append(val)
			
			DEP_val_list = []
			for key in DEP_tags:
				val = dep_counter.get(key,0)
				DEP_val_list.append(val)

			data = ["avg sentence len", "neg sent", "neu sent", "pos sent", "comp sent","avg sim score", "part_words/total_words"]
			col_names = data + POS_tags + DEP_tags
			head = [col for col in col_names]
			avg_sentence_length = sum(len(sent.split()) for sent in sentences) / len(sentences)
			writer.writerow(head)
			body = [avg_sentence_length] + overall_sentiment_list + [avg_sim_score] + [word_ratio]+ POS_val_list + DEP_val_list
			writer.writerow(body)
			
		self.logger.info("\tSemantic analysis completed.")

	# Output silent periods from participant audio to file (used for OpenFace analysis).
	def silence_detect(self, stop_d):
		self.output_name = self.participant_name + "_audio"
		self.log_dir = self.participant_dir + r'\logs'
		try:
			os.mkdir(self.log_dir)
		except OSError:
			self.logger.debug("\tThe log directory already exists. Existing logs will be overwritten.")
		self.output_sr = f"{self.participant_dir}\\{self.participant_name}_silences.txt"
		if self.audio == None:
			self.logger.warning("\tNo audio files provided.")
		else:
			command = f'{self.ffmpeg_path} -i "{self.audio}" -af "silencedetect=d={stop_d}" -f null - > {self.output_sr} 2>&1'
			self.logger.debug("\tRunning ffmpeg detection on participant file using {}.".format(command))
			exit_c, output = subprocess.getstatusoutput(command)
			if exit_c != 0:
				self.logger.error(f'\tffmpeg returned exit code {exit_c}. See {self.output_sr} for detailed error message.'.format(exit_c))
				sys.exit(1)

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
			command = f'{self.ffmpeg_path} -i {self.video} -vf "crop=in_w/2:in_h:in_w/2:0"  {cropped_video}>> {log_path} 2>&1'
			self.logger.debug("\tRunning ffmpeg cropping on participant file using {}.".format(command))
			exit_c, output = subprocess.getstatusoutput(command)
			if exit_c != 0:
				self.logger.error(f'\tffmpeg returned exit code {exit_c}. See log file for detailed error message.'.format(exit_c))
				sys.exit(1)
		self.video = cropped_video

	# Run either/both audio or/and video domains, acll summary generation function.
	def run_pipe(self):
		self.silence_detect(5)
		if self.run_mode != "video":
			self.run_audio()
		if self.run_mode != "audio":
			if not self.no_cut:
				self.cut_video()
			self.run_video()
		self.write_own_summary()
		self.logger.info(f"\tPipeline completed for {self.participant_name}.")
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
					if (f.endswith('2.mp4')):
						audio = file
					elif (f.endswith('1.mp4')):
						int_audio = file
					elif f.find('gvo') != -1 and f.endswith('.mp4'):
						video = file
				if args.mode != video and audio == None:
					raise Exception(f"\t{participant_dir} does not have an audio file. Either remove this folder or provide a video file. See README for information on naming files.")
				if args.mode != audio and video == None:
					raise Exception(f"\t{participant_dir} does not have a video file. Either remove this folder or provide an audio file. See README for information on naming files.")
				if args.mode != video and int_audio == None:
					print(f"{participant_dir} does not have an interviewer audio file. The pipeline will skip all analysis requiring this file.")
				all_args.append([audio, video, int_audio, args.verbosity, args.mode, args.no_cut, args.whisper_model])
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
                if (not (filename.endswith('mp4'))):
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
				file_path = first[0]
				break
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

def replay_video(path):
		
		subject = os.path.basename(path)
		df = load_features(path, subject)
		video = cv2.VideoCapture(f"{path}/{subject}_video.mp4")
		fps = video.get(cv2.CAP_PROP_FPS)
		sleep_ms = int(np.round((1/fps)*1000))
		count = 1
		#player = MediaPlayer(f"{path}/{subject}_video.mp4")
		while True:
			grabbed, frame=video.read()
			# Resize the image frames 
			resize = cv2.resize(frame, (900, 1100))
			fig = create_chart(count, df, subject, path)
			fig = fig[:,:,0:3]
			rows,cols,channels = fig.shape
			overlay=cv2.addWeighted(resize[0:0+rows, 0:0+cols],0.5,fig,0.5,0)
			resize[0:0+rows, 0:0+cols ] = overlay
			count += 1
			#audio_frame, val = player.get_frame()
			if not grabbed:
				print("End of video")
				break
			if cv2.waitKey(sleep_ms) & 0xFF == ord("q"):
				break
			cv2.imshow("Video", resize)
			#if val != 'eof' and audio_frame is not None:
				#audio
				#img, t = audio_frame
		video.release()
		cv2.destroyAllWindows()

def load_features(path, subject):

	df_face = pd.read_csv(f"{path}/{subject}.csv")
	df_face = df_face[["frame"," AU02_c"," AU09_c"," AU10_c"," AU28_c"," AU45_c"]]
	for column in [" AU02_c"," AU09_c"," AU10_c"," AU28_c"," AU45_c"]:
		tmp = np.diff(df_face[column])
		tmp = tmp > 0
		tmp = np.append(np.zeros(1), tmp.astype(int))
		df_face[column] = np.convolve(tmp, np.ones(260)/260, mode='same')*26*6
	return(df_face)

def create_chart(count, df, subject, path):
	df = df[df["frame"] == count]
	df = df[[" AU02_c"," AU09_c"," AU10_c"," AU28_c"," AU45_c"]]
	df_plot = pd.DataFrame(dict(
    r=df.iloc[0],
    theta=["Outer brow raiser","Nose wrinkler","Upper lip raiser","Lip suck rate","Blink rate"]))
	fig = px.line_polar(df_plot, r='r', theta='theta', line_close=True)
	fig.update_polars(
    radialaxis_tickvals=[1, 2, 3, 4, 5],
    radialaxis_tickmode="array",
    radialaxis_range=[0, 5])  # Set the range of radial axis to always go up to 5
	fig.update_traces(fill='toself')
	fig_bytes = fig.to_image(format="png",  width=400, height=400, scale=0.75)
	buf = io.BytesIO(fig_bytes)
	img = Image.open(buf)
	return np.asarray(img)

## GUI class.
class DigBioWindow(QMainWindow):
    def __init__(self):
        super(DigBioWindow, self).__init__() # Call the inherited classes __init__ method
        loadUi('DigBio.ui', self) # Load the .ui file
        self.show() # Show the GUI
        self.setWindowTitle('DigBio 1.0')
        self.setFixedSize(1000, 300)

        # Set default values
        self.pathFolder.setText(os.path.expanduser("~"))
        self.modeBox.setCurrentText("all")
        self.verbosityBox.setCurrentText("3")
        self.whisperBox.setCurrentText("base")
        self.nocutBox.setChecked(False)

        # Set GUI signals
        self.pathButton.clicked.connect(self.pathButtonClicked)
        self.replayButton.clicked.connect(self.replayButtonClicked)
        self.startButton.clicked.connect(self.startButtonClicked)

    def pathButtonClicked(self):
        self.pathFolder.setText(str(QFileDialog.getExistingDirectory(
            self,"Select Directory",self.pathFolder.text(),
            QFileDialog.ShowDirsOnly)))
		
    def replayButtonClicked(self):
        subject_path = str(QFileDialog.getExistingDirectory(
            self,"Select processed subject",self.pathFolder.text(),
            QFileDialog.ShowDirsOnly))
        replay_video(subject_path)

    def startButtonClicked(self):
        all_args = Namespace(interviews=self.pathFolder.text(), 
                             mode=self.modeBox.currentText(),
                             verbosity=self.verbosityBox.currentText(),
                             overwrite=self.overwriteBox.isChecked(),
                             no_cut=self.nocutBox.isChecked(),
                             whisper_model=self.whisperBox.currentText())

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
        print("INFO:\t All participants complete. You can close the windows now.")
        self.close()


if __name__ == '__main__':
    multiprocessing.freeze_support()
    app = QApplication(sys.argv)
    window = DigBioWindow()
    app.exec_()