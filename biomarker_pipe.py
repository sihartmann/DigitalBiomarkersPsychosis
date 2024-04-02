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

class pipe:
	def parse_args(self, args):
		self.audio = args[0]
		self.video = args[1]
		self.patient_dir = os.path.dirname(self.audio)
		match = re.search(r'\d+$', os.path.basename(self.audio[:-4]))
		self.patient_name = match.group()
		print(self.patient_name)
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



	def get_patients(self, directory):
		patients = list()
		for entry in os.listdir(directory):
			if os.path.isdir(os.path.join(directory, entry)):
				patients.append(entry)
		return patients

	def shutdown(self, exitcode, msg, files=None):
		if exitcode == 1:
			if files is not None:
				self.logger.warning(f'\tThe following files may be corrupted and will be deleted: {files}'.format(files))
				self.delete_files(files)
			self.logger.error(msg)
			sys.exit(1)
		self.logger.info(msg)

		
	def check_filepaths(self):
		if self.run_mode != "audio" and not os.path.isfile(self.video):
			self.shutdown(1, "\tThe provided path {0} is not valid".format(self.video))
		if self.run_mode != "video" and not os.path.isfile(self.audio):
			self.shutdown(1, "\tThe provided path {0} is not valid".format(self.audio))

	def run_opensmile(self):
		self.logger.debug("\tStarting opensmile.")
		smile = opensmile.Smile(
    	feature_set = opensmile.FeatureSet.eGeMAPSv02,
    	feature_level = opensmile.FeatureLevel.LowLevelDescriptors,
		)
		features = smile.feature_names
		opensmile_path = f"{self.patient_dir}\\{self.patient_name}"
		f = open(f"{opensmile_path}_summary_opensmile.csv", 'w')
		fI = open(f"{opensmile_path}_opensmile.csv", 'w')
		writer = csv.writer(f)
		writer.writerow([feature for feature in features])
		features = smile.process_file(self.audio)
		averageFeatures = features.mean()
		writer.writerow(averageFeatures)    
		f.close()
		self.logger.debug("\tOpensmile has completed successfully.")

	def run_whisper(self, model):
		if os.path.isfile(f"{self.patient_dir}\\{self.output_name}.mp3") == False:
			self.shutdown(1, "Cleaned mp3 files could not be found. Please do not move or delete them.")
		self.logger.info("Starting whisper.")
		try:
			os.mkdir(self.log_dir)
		except OSError:
			self.logger.warning("The log directory already exists. Existing logs will be overwritten.")
		log_file = self.patient_name + "_whisper.log"
		self.transcript = f"{self.patient_dir}\\{self.output_name}.txt"
		if os.path.isfile(self.transcript):
			self.logger.info("Transcript for this patient already exists. Skipping transcription...")
		else:
			self.linux_audio = f"{self.patient_dir}\\{self.output_name}.mp3"
			linux_audio = self.linux_audio.replace('\\','/')
			command = f'{self.whisper_path} -f txt --model {model} audio {linux_audio} --output_dir {self.patient_dir}> {self.log_dir}\\{log_file} 2>&1'
			self.logger.debug("\tRunning whisper on cleaned patient file using {}.".format(command))
			subprocess.check_output(command, shell=True)
			self.logger.info("\twhisper (patient) has finished successfully.")
		
		log_file = self.patient_name + "_whisper_interviewer.log"
		self.transcript_int = f"{self.audio_interviewer[:-4]}.txt"
		if os.path.isfile(self.transcript_int):
			self.logger.info("Transcript for this interviewer already exists. Skipping transcription...")
		else:
			self.linux_audio = f"{self.audio_interviewer}"
			linux_audio = self.linux_audio.replace('\\','/')
			command = f'{self.whisper_path} -f txt --model {model} audio {linux_audio} --output_dir {self.patient_dir}> {self.log_dir}\\{log_file} 2>&1'
			self.logger.debug("\tRunning whisper on cleaned interviewer file using {}.".format(command))
			subprocess.check_output(command, shell=True)
			self.logger.info("\twhisper (interviewer) has finished successfully.")

	def run_audio(self):
		if self.audio.endswith('mp4'):
			self.logger.info('Found mp4 audio file. Converting to mp3...')
			command = f'{self.ffmpeg_path} -i {self.audio} {self.audio[:-4]}.mp3'
			exit_c, output = subprocess.getstatusoutput(command)
			if exit_c != 0:
				self.logger.error(f'\twhisper returned exit code {exit_c}. See log file for detailed error message.'.format(exit_c))
				self.shutdown(1, "\tAudio file could not be converted to .mp3. If this issue persists, manually convert and reupload it.")
			self.logger.info("\tAudio file has been converted to mp3")
			self.audio =  f'{self.audio[:-4]}.mp3'
		self.clean_audio(1, 1, -50, -1, 5, -50)
		self.run_opensmile() 
		self.run_whisper("base")
		self.run_nltk(f'{self.patient_dir}\\{self.patient_name}_nltk_results.txt', f'{self.patient_dir}\\{self.patient_name}_sim_scores.csv')

	def run_video(self):
		self.logger.info('Starting openface.')
		self.log_dir = self.patient_dir + r'\logs'
		log_file = self.patient_name + "_openface.log"
		try:
			os.mkdir(self.log_dir)
		except OSError:
			self.logger.warning("The log directory already exists. Existing logs will be overwritten.")
		log_path = f"{self.log_dir}\\{log_file}"
		if not os.path.isfile(self.video):
			self.logger.error("No video file was found.")
		command = f'{self.feat_detect} -f {self.video} --out_dir {self.patient_dir} -verbose > {log_path} 2>&1'
		self.logger.info(f"Running openface with command {command}")
		exit_c, output = subprocess.getstatusoutput(command)
		if exit_c != 0:
			self.logger.error(f'\tOpenface returned exit code {exit_c}. See log file for detailed error message.'.format(exit_c))
			self.shutdown(1, "\tExecution cancelled due to error in openface.")
		self.logger.info("\tOpenface has finished successfully.")


	def delete_files(self, files):
		for file in files:
			try:
				os.remove(file)
				self.logger.debug(f"{file} deleted successfully")
			except OSError as e:
				print(f"Error deleting {file}: {e}")

	def preprocess_text(self, text, nlp):     
		doc = nlp(text.lower())
		tokens = [token.lemma_ for token in doc]
		return ' '.join(tokens)
	
	def run_nltk(self, output_file, lsa_output):
		# Open output file
		file = self.transcript
		with open(file, 'r') as f:
			all_text = f.read()

		with open(self.transcript_int, 'r') as f_i:
			all_int_text = f_i.read()
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
		int_sentences = nltk.tokenize.sent_tokenize(all_int_text)
		num_sent = len(sentences)
		num_int_sent = len(int_sentences)
		# Do sentiment scores per sentence
		sia = SentimentIntensityAnalyzer()
		sentiment_scores = [sia.polarity_scores(sentence) for sentence in sentences]

		# Remove punctuation
		translator = str.maketrans('','', string.punctuation)
		fixed_text = all_text.translate(translator)

		# Get rid of contractions like I'm, it's, can't ...
		words = nltk.tokenize.word_tokenize(fixed_text)
		expanded_text = [contractions.fix(word) for word in words]
		# print(expanded_text)
		merged_text = ' '.join(expanded_text)
		# print(merged_text)
		# Do POS and dependency tagging.
		nlp = spacy.load('en_core_web_sm')
		doc = nlp(merged_text)
		with open(output_file, 'w') as f:
			pos_counter = {}
			dep_counter = {}
			for token in doc:
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

			avg_sentence_length = sum(len(sent.split()) for sent in sentences) / len(sentences)
			f.write("\nAverage Sentence Length: {}\n".format(avg_sentence_length))
			f.write("\nSentiment Scores: {}\n".format(sentiment_scores))
			f.write(f"\nPOS counts: {pos_counter}")
			f.write(f"\nDependency counter: {dep_counter}")
			f.write(f"\nNumber of patient sentences: {num_sent}")
			f.write(f"\nNumber of interviewer sentences: {num_int_sent}")

	def clean_audio(self, start_p, start_d, start_t, stop_p, stop_d, stop_t):
		self.output_name = self.patient_name + "_cleaned"
		self.log_dir = self.patient_dir + r'\logs'
		log_file = self.patient_name + "_ffmpeg.log"
		try:
			os.mkdir(self.log_dir)
		except OSError:
			self.logger.warning("The log directory already exists. Existing logs will be overwritten.")
		log_path = f"{self.log_dir}\\{log_file}"
		output_path = f"{self.patient_dir}\\{self.output_name}.mp3"
		print("Output file will be", output_path)
		print("Log file will be", log_path)
		if os.path.isfile(output_path) == False:
			command = f'{self.ffmpeg_path} -i "{self.audio}" -af "silenceremove=start_periods=0:start_duration=1:stop_periods=-1:stop_duration=5" {output_path} > {log_path} 2>&1'
			self.logger.debug("\tRunning ffmpeg on patient file using {}.".format(command))
			exit_c, output = subprocess.getstatusoutput(command)
			if exit_c != 0:
				self.logger.error(f'\tffmpeg returned exit code {exit_c}. See log file for detailed error message.'.format(exit_c))
				self.shutdown(1, "\tExecution cancelled due to error in ffmpeg.", [self.output_name + ".mp3"])
			self.logger.info("\tffmpeg has finished successfully.")
		else:
			self.logger.info("\tFiles have already been cleaned. Skipping step and continuing...")


	def run_pipe(self):
		self.check_filepaths()
		if self.run_mode != "video":
			self.run_audio()
		if self.run_mode != "audio":
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
					for root, dirs, files in os.walk(interviews_path):
						for file in files:
							print(file, dirs)
							if (file.startswith('Audio_Participant')):
								audio = os.path.join(root,file)
							elif (file.startswith('Audio_Interviewer')):
								int_audio = os.path.join(root, file)
							elif file.startswith('Video'):
								video = os.path.join(root,file)
						all_args.append([audio, video, log_level, run_mode, int_audio])
					print(all_args[1:])
					return len(all_args) - 1, all_args[1:]
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
                if (not ((filename.startswith('Audio') and (filename.endswith('mp4') or filename.endswith('mp3'))) or filename.startswith('Video')) or (filename.endswith('_cleaned.mp3'))):
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
	
	
	