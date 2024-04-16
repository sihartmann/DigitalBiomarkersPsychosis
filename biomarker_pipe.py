import argparse
import logging
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
import numpy as np
import string
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Necessary data downloads:
# nltk.download('maxent_ne_chunker')
# nltk.download('words')
# nltk.download('vader_lexicon')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('stopwords')

VERSION = '1.0'

# To do:
# Fix naming.
# Create opnsmile/face/nltk summary file for all patients.
# python wrapper

class pipe:
	def parse_args(self, args):
		self.audio = args[0]
		self.video = args[1]
		if self.audio is not None:
			self.participant_dir = os.path.dirname(self.audio)
			print(self.participant_dir)
		elif self.video is not None:
			self.participant_dir = os.path.dirname(self.video)
			print(self.participant_dir)
		else:
			sys.exit(1)
		self.participant_name = os.path.basename(self.participant_dir)
		self.loglevel = args[2]
		self.run_mode = args[3]
		self.audio_interviewer = args[4]
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
		self.ffmpeg_path = r"C:\Users\a1863615\Downloads\ffmpeg-2024-03-11-git-3d1860ec8d-full_build\bin\ffmpeg.exe"
		self.whisper_path = r"C:\Users\a1863615\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\LocalCache\local-packages\Python312\Scripts\whisper.exe"
		self.feat_detect = r"C:\Users\a1863615\Downloads\OpenFace_2.2.0_win_x64\OpenFace_2.2.0_win_x64\FeatureExtraction.exe"

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
			self.shutdown(1, "\tThe video file could not found")
		if self.run_mode != "video" and self.audio is  None:
			self.shutdown(1, "\tThe audio file could not be found")

	def run_opensmile(self):
		opensmile_path = f"{self.participant_dir}\\{self.participant_name}"
		f_summary = f"{opensmile_path}_summary_opensmile.csv"
		f_individual = f"{opensmile_path}_opensmile.csv"
		if os.path.isfile(f_individual) and os.path.isfile(f_summary):
			self.logger.info("\tOutput files already exist. Skipping OpenSmile...")
		else:
			self.logger.debug("\tStarting opensmile.")
			smile = opensmile.Smile(
				feature_set = opensmile.FeatureSet.eGeMAPSv02,
				# feature_level = opensmile.FeatureLevel.LowLevelDescriptors,
			)
			features = smile.process_file(self.audio)
			with open(f_summary, 'w', newline='') as f:
				writer = csv.writer(f)
				writer.writerow(features.keys())
				averageFeatures = [np.mean(features[key]) for key in features.keys()]
				writer.writerow(averageFeatures)
			with open(f_individual, 'w', newline='') as file:
				writer = csv.writer(file)
				writer.writerow(features.keys())
				for key in features.keys():
					writer.writerow(features[key])
			self.logger.debug("\tOpensmile has completed successfully.")

	def run_whisper(self, model):
		if os.path.isfile(f"{self.participant_dir}\\{self.output_name}.mp3") == False:
			self.shutdown(1, "Cleaned mp3 files could not be found. Please do not move or delete them.")
		self.logger.info("Starting whisper.")
		try:
			os.mkdir(self.log_dir)
		except OSError:
			self.logger.warning("The log directory already exists. Existing logs will be overwritten.")
		log_file = self.participant_name + "_whisper.log"
		self.transcript = f"{self.participant_dir}\\{self.output_name}.txt"
		if os.path.isfile(self.transcript):
			self.logger.info("Transcript for this participant already exists. Skipping transcription...")
		else:
			self.linux_audio = f"{self.participant_dir}\\{self.output_name}.mp3"
			linux_audio = self.linux_audio.replace('\\','/')
			command = f'{self.whisper_path} -f txt --model {model} audio {linux_audio} --output_dir {self.participant_dir}> {self.log_dir}\\{log_file} 2>&1'
			self.logger.debug("\tRunning whisper on cleaned participant file using {}.".format(command))
			subprocess.check_output(command, shell=True)
			self.logger.info("\twhisper (participant) has finished successfully.")
		
	def run_audio(self):
		if self.audio.endswith('mp4') and not os.path.isfile(f'{self.audio[:-4]}.mp3'):
			self.logger.info('Found mp4 audio file. Converting to mp3...')
			command = f'{self.ffmpeg_path} -i {self.audio} {self.audio[:-4]}.mp3'
			exit_c, output = subprocess.getstatusoutput(command)
			if exit_c != 0:
				self.logger.error(f'\tffmpeg returned exit code {exit_c}. See log file for detailed error message.'.format(exit_c))
				self.shutdown(1, "\tAudio file participant could not be converted to .mp3. If this issue persists, manually convert and reupload it.")
			self.logger.info("\tAudio file participant has been converted to mp3")

		if self.audio_interviewer.endswith('mp4') and not os.path.isfile(f'{self.audio_interviewer[:-4]}.mp3'):
			self.logger.info('Found mp4 audio file. Converting to mp3...')
			command = f'{self.ffmpeg_path} -i {self.audio_interviewer} {self.audio_interviewer[:-4]}.mp3'
			exit_c, output = subprocess.getstatusoutput(command)
			if exit_c != 0:
				self.logger.error(f'\tffmpeg returned exit code {exit_c}. See log file for detailed error message.'.format(exit_c))
				self.shutdown(1, "\tAudio file interviewer could not be converted to .mp3. If this issue persists, manually convert and reupload it.")
			self.logger.info("\tAudio file interviewer has been converted to mp3")
		
		self.audio_interviewer = f'{self.audio_interviewer[:-4]}.mp3'
		self.run_opensmile() 
		self.run_whisper("base")
		self.run_nltk(f'{self.participant_dir}\\{self.participant_name}_nltk_results.txt', f'{self.participant_dir}\\{self.participant_name}_sim_scores.csv', f'{self.participant_dir}\\{self.participant_name}_nltk_results.csv')

	def run_video(self):
		openface_out = f'{self.video[:-10]}.csv'
		if os.path.isfile(openface_out):
			self.logger.info("Output files already exist. Skipping OpenFace...")
		else:
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
		part_silence_list = self.parse_silence_file(self.output_sr)
		int_silence_list = self.parse_silence_file(self.output_sr_int)
		edited_rows = self.check_silence_periods(part_silence_list, int_silence_list, rows)
		au_count = self.count_binary_aus(rows)
		print(self.participant_name, au_count)
		with open(f'{self.video[:-4]}.csv', 'w', newline='') as file:
			writer = csv.writer(file)
			for row in edited_rows:
				writer.writerow(row)

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

	def count_binary_aus(self, rows):
		au_count = [0]*18
		last_val = [0]*18
		for row in rows[1:]:
			for i in range(696, 714):
				if int(float(row[i].strip())) == 1:
					if last_val[i-696] == 0:
						au_count[i-696] += 1
				last_val[i-696] = int(float(row[i].strip()))
		return au_count

	def check_silence_periods(self, silence_list, silence_list_int, rows):
		current_start = silence_list[0]
		current_end = silence_list[1]
		period_index = 2 
		modified_rows = []
		rows[0].extend(["participant speech", "interviewer speech"])
		modified_rows.append(rows[0])
		current_start_int = silence_list_int[0]
		current_end_int = silence_list_int[1]
		period_index_int = 2 

		for row in rows[1:]:
			timestamp = float(row[2])
			end_flag = False
			end_flag_int = False

			if current_start <= timestamp < current_end and end_flag is not True:
				silence_flag = 0
			else:
				silence_flag = 1
			row.append(str(silence_flag))

			if timestamp > current_end:
				if silence_list[-1] == current_end:
					end_flag = True
				else:
					current_start = silence_list[period_index]
					current_end = silence_list[period_index + 1]
					period_index += 2

			if current_start_int <= timestamp < current_end_int and end_flag_int is not True:
				silence_flag_int = 0
			else:
				silence_flag_int = 1
			row.append(str(silence_flag_int))
			modified_rows.append(row)
			if timestamp > current_end_int:
				if silence_list_int[-1] == current_end_int:
					end_flag_int = True
				else:
					current_start_int = silence_list_int[period_index_int]
					current_end_int = silence_list_int[period_index_int + 1]
					period_index_int += 2
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

			data = ["avg sentence len", "neg sent", "neu sent", "pos sent", "comp sent","avg sim score"]
			col_names = data + POS_tags + DEP_tags
			head = ["participant"] + [col for col in col_names]
			avg_sentence_length = sum(len(sent.split()) for sent in sentences) / len(sentences)
			writer.writerow(head)
			body = [self.participant_name, avg_sentence_length] + overall_sentiment_list + [avg_sim_score] + POS_val_list + DEP_val_list
			writer.writerow(body)
			
		self.logger.info("\tSemantic analysis completed.")

	def clean_audio(self, stop_d):
		self.output_name = self.participant_name + "_cleaned"
		self.log_dir = self.participant_dir + r'\logs'
		log_file = self.participant_name + "_ffmpeg.log"
		try:
			os.mkdir(self.log_dir)
		except OSError:
			self.logger.warning("The log directory already exists. Existing logs will be overwritten.")
		log_path = f"{self.log_dir}\\{log_file}"
		output_path = f"{self.participant_dir}\\{self.output_name}.mp3"
		output_path_int = f'{output_path[:-4]}_int.mp3'
		self.output_sr = f"{self.participant_dir}\\silenceremove.txt"
		self.output_sr_int = f"{self.participant_dir}\\silenceremove_int.txt"
		if os.path.isfile(output_path) and os.path.isfile(self.output_sr) and os.path.isfile(self.output_sr_int) and os.path.isfile(output_path_int):
			self.logger.info("\tFiles have already been cleaned. Skipping step and continuing...")
		else:
			command = f'{self.ffmpeg_path} -i "{self.audio}" -af "silencedetect=d={stop_d}" -f null - > {self.output_sr} 2>&1'
			self.logger.debug("\tRunning ffmpeg detection on participant file using {}.".format(command))
			exit_c, output = subprocess.getstatusoutput(command)
			if exit_c != 0:
				self.logger.error(f'\tffmpeg returned exit code {exit_c}. See log file for detailed error message.'.format(exit_c))
				self.shutdown(1, "\tExecution cancelled due to error in ffmpeg.", [])

			command = f'{self.ffmpeg_path} -i "{self.audio_interviewer}" -af "silencedetect=d={stop_d}" -f null - > {self.output_sr_int} 2>&1'
			self.logger.debug("\tRunning ffmpeg detection on interviewer file using {}.".format(command))
			exit_c, output = subprocess.getstatusoutput(command)
			if exit_c != 0:
				self.logger.error(f'\tffmpeg returned exit code {exit_c}. See log file for detailed error message.'.format(exit_c))
				self.shutdown(1, "\tExecution cancelled due to error in ffmpeg.", [])

			command = f'{self.ffmpeg_path} -i "{self.audio}" -af "silenceremove=start_periods=0:start_duration=1:stop_periods=-1:stop_duration={stop_d}" {output_path} >> {log_path} 2>&1'
			self.logger.debug("\tRunning ffmpeg removal on participant file using {}.".format(command))
			exit_c, output = subprocess.getstatusoutput(command)
			if exit_c != 0:
				self.logger.error(f'\tffmpeg returned exit code {exit_c}. See log file for detailed error message.'.format(exit_c))
				self.shutdown(1, "\tExecution cancelled due to error in ffmpeg.", [self.output_name + ".mp3"])
			self.logger.info("\tffmpeg has finished successfully.")
			
			command = f'{self.ffmpeg_path} -i "{self.audio_interviewer}" -af "silenceremove=start_periods=0:start_duration=1:stop_periods=-1:stop_duration={stop_d}" {output_path_int} >> {log_path} 2>&1'
			self.logger.debug("\tRunning ffmpeg removal on interviewer file using {}.".format(command))
			exit_c, output = subprocess.getstatusoutput(command)
			if exit_c != 0:
				self.logger.error(f'\tffmpeg returned exit code {exit_c}. See log file for detailed error message.'.format(exit_c))
				self.shutdown(1, "\tExecution cancelled due to error in ffmpeg.", [self.output_name + ".mp3"])
			self.logger.info("\tffmpeg has finished successfully.")



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

class pipeParser:
	def parse_args(self, args):
		self.config_name = args.file
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
						for f in os.listdir(participant_dir_path):
							file = os.path.join(participant_dir_path, f)
							print(file)
							if (f.endswith('2.mp4')):
								audio = file
							elif (f.endswith('1.mp4')):
								int_audio = file
							elif f.find('gvo') != -1 and f.endswith('.mp4'):
								video = file
						all_args.append([audio, video, log_level, run_mode, int_audio])
					return len(all_args), all_args
				else:
					print(f"Directory path '{interviews_path}' is not valid.")
			else:
				print("Invalid configuration file format.")

def process_func(my_pipe, args):
	my_pipe.parse_args(args)
	my_pipe.run_pipe()

def clear_data(path):
    try:
        for root, dirs, files in os.walk(path):
            if 'logs' in dirs:
                log_dir_path = os.path.join(root, 'logs')
                print(f"Found logs directory: {log_dir_path}")
                for log_root, log_dirs, log_files in os.walk(log_dir_path):
                    for log_file in log_files:
                        log_file_path = os.path.join(log_root, log_file)
                        print(f"Deleting file: {log_file_path}")
                        os.remove(log_file_path)
                    for log_dir in log_dirs:
                        log_subdir_path = os.path.join(log_root, log_dir)
                        print(f"Deleting directory: {log_subdir_path}")
                        shutil.rmtree(log_subdir_path)
                print(f"Deleting log directory: {log_dir_path}")
                os.rmdir(log_dir_path)

            for filename in files:
                file_path = os.path.join(root, filename)
                if (not (filename.endswith('mp4'))):
                    print(f"Deleting file: {file_path}")
                    os.remove(file_path)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--file", "-f", help = "The name of your file containing the arguments you want to use.")
	parser.add_argument("--overwrite", "-o", help = "Use this option if you want to run the pipeline again. This will overwrite old data!", action="store_true")
	all_args = parser.parse_args()
	pipeParser = pipeParser()
	num_procs, parsed_all_args = pipeParser.parse_args(all_args)
	my_pipes = [pipe() for _ in range(num_procs)]
	processes = []
	for i in range(num_procs):
		process = multiprocessing.Process(target=process_func, args=(my_pipes[i], parsed_all_args[i]))
		processes.append(process)
		process.start()
	for process in processes:
		process.join()
	
	
	