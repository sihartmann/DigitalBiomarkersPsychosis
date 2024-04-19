"""
Pipeline for extraction of facial, acoustic and linguistic features from HIPAA Zoom recordings.
@author: Oliver Williams
"""
import argparse
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

VERSION = '1.0'

## Main processing class.
class pipe:
	# Parse arguments and set up logging.
	def parse_args(self, args):
		self.audio = args[0]
		self.video = args[1]
		self.loglevel = args[2]
		self.run_mode = args[3]
		self.audio_interviewer = args[4]
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
		stderrhandler.setLevel(int(self.loglevel))
		stderrhandler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
		self.logger = logging.getLogger('DigBio')
		self.logger.addHandler(stderrhandler)
		match self.loglevel:
			case '1':
				self.logger.setLevel(logging.ERROR)
			case '2':
				self.logger.setLevel(logging.WARNING)
			case '3':
				self.logger.setLevel(logging.INFO)
			case '4':
				self.logger.setLevel(logging.DEBUG)

		self.logger.info("\tInitializing DigBio")
		self.logger.info("\tDigBio version: {0}.".format(VERSION))
		self.logger.debug("\tRunning DigBio with command: {}.".format(' '.join(sys.argv)))
		self.ffmpeg_path =  r"ffmpeg.exe"
		self.whisper_path = r"whisper.exe"
		self.feat_detect =  r"OpenFace_2.2.0_win_x64\FeatureExtraction.exe"

	def get_participants(self, directory):
		participants = list()
		for entry in os.listdir(directory):
			if os.path.isdir(os.path.join(directory, entry)):
				participants.append(entry)
		return participants

	def shutdown(self, exitcode, msg, files=None):
		if exitcode == 1:
			if files is not None:
				self.logger.warning(f'\tThe following files may be corrupted and will be deleted: {files}'.format(files))
				self.delete_files(files)
			self.logger.error(msg)
			sys.exit(1)
		self.logger.info(msg)

	def check_filepaths(self):
		if self.run_mode != "audio" and self.video is None:
			self.shutdown(1, "\tThe video file could not found.")
		if self.run_mode != "video" and self.audio is  None:
			self.shutdown(1, "\tThe audio file could not be found.")

	def run_opensmile(self):
		opensmile_path = f"{self.participant_dir}\\{self.participant_name}"
		self.f_summary = f"{opensmile_path}_summary_opensmile.csv"
		f_individual = f"{opensmile_path}_opensmile.csv"
		if os.path.isfile(f_individual) and os.path.isfile(self.f_summary):
			self.logger.info("\tOutput files already exist. Skipping OpenSmile...")
		else:
			self.logger.debug("\tStarting opensmile.")
			smile = opensmile.Smile(
				feature_set = opensmile.FeatureSet.eGeMAPSv02,
				feature_level = opensmile.FeatureLevel.LowLevelDescriptors,
			)
			features = smile.process_file(self.audio)
			df = pd.DataFrame(features, columns=smile.feature_names)
			df.to_csv(f_individual, index=False)

			smile_summary = opensmile.Smile(
				feature_set = opensmile.FeatureSet.eGeMAPSv02
			)
			features = smile_summary.process_file(self.audio)
			df = pd.DataFrame(features, columns=smile_summary.feature_names)
			df.to_csv(self.f_summary, index=False)
			self.logger.debug("\tOpensmile has completed successfully.")

	def run_whisper(self, model):
		self.logger.info("Starting whisper.")
		try:
			os.mkdir(self.log_dir)
		except OSError:
			self.logger.warning("The log directory already exists. Existing logs will be overwritten.")
		log_file = self.participant_name + "_whisper.log"
		self.transcript = f"{self.participant_dir}\\{self.participant_name}_transcript.txt"
		self.transcript_int = f"{self.participant_dir}\\{self.participant_name}_interviewer_transcript.txt"
		self.transcript_clean = f"{self.participant_name}_transcript.txt"
		self.transcript_int_clean = f"{self.participant_name}_interviewer_transcript.txt"
		if os.path.isfile(self.transcript):
			self.logger.info("Transcript for this participant already exists. Skipping transcription...")
		else:
			linux_audio = self.audio.replace('\\','/')
			command = f'{self.whisper_path} -f txt --model {model} audio {linux_audio} --hallucination_silence_threshold 1 --word_timestamps True --output_dir {self.participant_dir}> {self.log_dir}\\{log_file} 2>&1'
			self.logger.debug("\tRunning whisper on cleaned participant file using {}.".format(command))
			subprocess.check_output(command, shell=True)
			command = f'ren {self.audio[:-4]}.txt {self.transcript_clean}'
			subprocess.check_output(command, shell=True)
			self.logger.info("\twhisper (participant) has finished successfully.")
		
		if os.path.isfile(self.transcript_int):
			self.logger.info("Transcript for this interviewer already exists. Skipping transcription...")
		else:
			linux_audio = self.audio_interviewer.replace('\\','/')
			command = f'{self.whisper_path} -f txt --model {model} audio {linux_audio} --hallucination_silence_threshold 1 --word_timestamps True --output_dir {self.participant_dir}> {self.log_dir}\\{log_file} 2>&1'
			self.logger.debug("\tRunning whisper on cleaned interviewer file using {}.".format(command))
			subprocess.check_output(command, shell=True)
			command = f'ren {self.audio_interviewer[:-4]}.txt {self.transcript_int_clean}'
			subprocess.check_output(command, shell=True)
			self.logger.info("\twhisper (interviewer) has finished successfully.")
		
	def run_audio(self):
		self.run_opensmile() 
		self.run_whisper("base")
		self.run_nltk(f'{self.participant_dir}\\{self.participant_name}_nltk_results.txt', f'{self.participant_dir}\\{self.participant_name}_sim_scores.csv',self.nltk_out)

	def run_video(self):
		openface_out = f'{self.video[:-10]}.csv'
		if os.path.isfile(openface_out):
			self.logger.info("Output files already exist. Skipping OpenFace...")
		else:
			if self.audio is None:
				self.logger.warning("No audio files found. Video analysis will run without silence detection.")
			self.logger.info('Starting openface.')
			self.log_dir = self.participant_dir + r'\logs'
			log_file = self.participant_name + "_openface.log"
			try:
				os.mkdir(self.log_dir)
			except OSError:
				self.logger.warning("The log directory already exists. Existing logs will be overwritten.")
			log_path = f"{self.log_dir}\\{log_file}"
			if not os.path.isfile(self.video):
				self.logger.error("No video file was found.")
			command = f'{self.feat_detect} -f {self.video} -out_dir {self.participant_dir} -of {self.participant_name} > {log_path} 2>&1'
			self.logger.info(f"Running openface with command {command}")
			exit_c, output = subprocess.getstatusoutput(command)
			if exit_c != 0:
				self.logger.error(f'\tOpenface returned exit code {exit_c}. See log file for detailed error message.'.format(exit_c))
				self.shutdown(1, "\tExecution cancelled due to error in openface.")
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
		with open(f'{self.video[:-4]}.csv', 'w', newline='') as file:
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


	def write_own_summary(self):
		own_summary = f"{self.participant_dir}\\{self.participant_name}_summary.csv"
		if os.path.isfile(own_summary):
			self.logger.info("Participant's summary file alredy exists. Skipping...")
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
		with open(self.summary, 'w', newline='') as f:
			writer = csv.writer(f)
			writer.writerow(self.header_list)
			writer.writerow(self.content_list)

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
						if audio_off:
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

	def delete_files(self, files):
		for file in files:
			try:
				os.remove(file)
				self.logger.debug(f"{file} deleted successfully")
			except OSError as e:
				self.logger.error(f"Error deleting {file}: {e}")

	def preprocess_text(self, text, nlp):     
		doc = nlp(text.lower())
		tokens = [token.lemma_ for token in doc]
		return ' '.join(tokens)
	
	def run_nltk(self, output_file, lsa_output, csv_output):
		# Open output file
		if not(os.path.isfile(self.transcript) and os.path.isfile(self.transcript_int)):
			self.shutdown(1, "Transcript could not be found. Do not rename or move it.", [])
		file = self.transcript
		with open(file, 'r') as f:
			all_text = f.read()
		self.logger.info("\tStarting semantic analysis.")
		#LSA
		nlp = spacy.load("en_core_web_sm")
		sentences = [self.preprocess_text(sent.text,nlp) for sent in nlp(all_text).sents]
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
		
		# Fragment into sentences
		sentences = nltk.tokenize.sent_tokenize(all_text)
		num_sent = len(sentences)
		words_num_part = len(nltk.tokenize.word_tokenize(all_text))
		file = self.transcript
		with open(self.transcript_int, 'r') as f:
			int_text = f.read()
		words_num_int = len(nltk.tokenize.word_tokenize(int_text))
		total_words = words_num_int + words_num_part
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
			body = [avg_sentence_length] + overall_sentiment_list + [avg_sim_score] + [words_num_int/total_words]+ POS_val_list + DEP_val_list
			writer.writerow(body)
			
		self.logger.info("\tSemantic analysis completed.")

	def clean_audio(self, stop_d):
		self.output_name = self.participant_name + "_audio"
		self.log_dir = self.participant_dir + r'\logs'
		log_file = self.participant_name + "_ffmpeg.log"
		try:
			os.mkdir(self.log_dir)
		except OSError:
			self.logger.warning("The log directory already exists. Existing logs will be overwritten.")
		log_path = f"{self.log_dir}\\{log_file}"
		self.output_path = f"{self.participant_dir}\\{self.output_name}.mp3"
		self.output_path_int = f'{self.output_path[:-4]}_int.mp3'
		self.output_sr = f"{self.participant_dir}\\silenceremove.txt"
		if os.path.isfile(self.output_path) and os.path.isfile(self.output_sr) and os.path.isfile(self.output_sr_int) and os.path.isfile(self.output_path_int):
			self.logger.info("\tFiles have already been cleaned. Skipping step and continuing...")
		elif self.audio == None:
			self.logger.warning("No audio files provided.")
		else:
			command = f'{self.ffmpeg_path} -i "{self.audio}" -af "silencedetect=d={stop_d}" -f null - > {self.output_sr} 2>&1'
			self.logger.debug("\tRunning ffmpeg detection on participant file using {}.".format(command))
			exit_c, output = subprocess.getstatusoutput(command)
			if exit_c != 0:
				self.logger.error(f'\tffmpeg returned exit code {exit_c}. See log file for detailed error message.'.format(exit_c))
				self.shutdown(1, "\tExecution cancelled due to error in ffmpeg.", [])

	def cut_video(self):
		self.log_dir = self.participant_dir + r'\logs'
		log_file = self.participant_name + "_ffmpeg.log"
		cropped_video = f'{self.participant_dir}\\{self.participant_name}_video.mp4'
		try:
			os.mkdir(self.log_dir)
		except OSError:
			self.logger.warning("The log directory already exists. Existing logs will be overwritten.")
		log_path = f"{self.log_dir}\\{log_file}"
		if os.path.isfile(cropped_video):
			self.logger.info("\tVideo has already been cropped. Skipping step and continuing...")
		else:
			command = f'{self.ffmpeg_path} -i {self.video} -vf "crop=in_w/2:in_h:in_w/2:0"  {cropped_video}>> {log_path} 2>&1'
			self.logger.debug("\tRunning ffmpeg detection on participant file using {}.".format(command))
			exit_c, output = subprocess.getstatusoutput(command)
			if exit_c != 0:
				self.logger.error(f'\tffmpeg returned exit code {exit_c}. See log file for detailed error message.'.format(exit_c))
				self.shutdown(1, "\tExecution cancelled due to error in ffmpeg.", [])
		self.video = cropped_video

	def run_pipe(self):
		self.check_filepaths()
		self.clean_audio(5)
		if self.run_mode != "video":
			self.run_audio()
		if self.run_mode != "audio":
			self.cut_video()
			self.run_video()
		self.write_own_summary()
		return [self.summary, self.header_list, self.content_list, self.participant_name]

class pipeParser:
	def parse_args(self, args):
		self.config_name = args.file
		if not os.path.isfile(self.config_name):
			raise Exception("Provided config file is not valid.")
		with open(self.config_name, 'r') as file:
			lines = file.readlines()
			if len(lines) == 3:
				interviews_path = lines[0].strip()
				log_level = lines[1].strip()
				run_mode = lines[2].strip()
				if os.path.isdir(interviews_path):
					print(f"Directory path '{interviews_path}' is valid.")
					if args.overwrite:
						clear_data(interviews_path)
					all_args = []
					video = None
					audio = None
					int_audio = None
					for participant_dir in os.listdir(interviews_path):
						participant_dir_path = os.path.join(interviews_path, participant_dir)
						if not participant_dir.endswith(".csv"):
							for f in os.listdir(participant_dir_path):
								file = os.path.join(participant_dir_path, f)
								if (f.endswith('2.mp4')):
									audio = file
								elif (f.endswith('1.mp4')):
									int_audio = file
								elif f.find('gvo') != -1 and f.endswith('.mp4'):
									video = file
							all_args.append([audio, video, log_level, run_mode, int_audio])
					return len(all_args), all_args
				else:
					raise Exception(f"Directory path '{interviews_path}' is not valid.")
			else:
				raise Exception("Invalid configuration file format.")

def process_func(queue, my_pipe, args):
	my_pipe.parse_args(args)
	result = my_pipe.run_pipe()
	queue.put(result)

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
	
if __name__ == '__main__':
	multiprocessing.freeze_support()
	parser = argparse.ArgumentParser()
	parser.add_argument("--file", "-f", help = "The name of your file containing the arguments you want to use.", default=r"C:\Users\a1863615\INT\config_win.txt")
	parser.add_argument("--overwrite", "-o", help = "Use this option if you want to run the pipeline again. This will overwrite old data!", action="store_true")
	all_args = parser.parse_args()
	pipeParser = pipeParser()
	num_procs, parsed_all_args = pipeParser.parse_args(all_args)
	my_pipes = [pipe() for _ in range(num_procs)]
	summary_queue = multiprocessing.Queue()
	processes = []
	summary_process = multiprocessing.Process(target=make_summary, args=(summary_queue,))
	summary_process.start()
	for i in range(num_procs):
		process = multiprocessing.Process(target=process_func, args=(summary_queue, my_pipes[i], parsed_all_args[i]))
		processes.append(process)
		process.start()
	for process in processes:
		process.join()
	summary_queue.put(None)
	summary_process.join()
