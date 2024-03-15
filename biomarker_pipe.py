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
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from gensim import corpora, models
import contractions
import string

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
		self.patient_dir = os.path.dirname(self.audio)
		self.patient_name = os.path.splitext(os.path.basename(self.audio))[0]
		self.loglevel = args[1]
		self.run_mode = args[2]
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
		f = open(self.patient_dir + "/" + self.patient_name + "_whisper.csv", 'w')
		writer = csv.writer(f)
		writer.writerow([feature for feature in features])
		features = smile.process_file(self.audio)
		averageFeatures = features.mean()
		writer.writerow(averageFeatures)    
		f.close()
		self.logger.debug("\tOpensmile has completed successfully.")

	def run_whisper(self, model):
		if os.path.isfile(f"{self.patient_dir}/{self.output_name}.mp3") == False:
			self.shutdown(1, "Cleaned mp3 files could not be found. Please do not move or delete them.")
		self.logger.info("Starting whisper.")
		try:
			os.mkdir(self.log_dir)
		except OSError:
			self.logger.warning("The log directory already exists. Existing logs will be overwritten.")
		log_file = self.patient_name + "_whisper.log"
		self.transcript = f"{self.patient_dir}/{self.output_name}_whisper.txt"
		if os.path.isfile(self.transcript):
			self.logger.info("Transcript for this patient already exists. Skipping transcription...")
		else:
			command = f'whisper  -o {self.patient_dir} -f txt --model {model} {self.patient_dir}/{self.output_name}.mp3 > {self.log_dir}/{log_file} && mv {self.patient_dir}/{self.output_name}.txt {self.transcript}'
			self.logger.debug("\tRunning whisper on cleaned patient file using {}.".format(command))
			exit_c, output = subprocess.getstatusoutput(command)
			if exit_c != 0:
				self.logger.error(f'\twhisper returned exit code {exit_c}. See log file for detailed error message.'.format(exit_c))
				self.shutdown(1, "\tExecution cancelled due to error in whisper.")
			self.logger.info("\twhisper has finished successfully.")

	def run_audio(self):
		if self.audio.endswith('mp4'):
			self.logger.info('Found mp4 audio file. Converting to mp3...')
			command = f'ffmpeg -i {self.audio} {self.audio[:-4]}.mp3'
			exit_c, output = subprocess.getstatusoutput(command)
			if exit_c != 0:
				self.logger.error(f'\twhisper returned exit code {exit_c}. See log file for detailed error message.'.format(exit_c))
				self.shutdown(1, "\tAudio file could not be converted to .mp3. If this issue persists, manually convert and reupload it.")
			self.logger.info("\tAudio file has been converted to mp3")
		self.clean_audio(1, 1, -50, -1, 5, -50)
		self.run_opensmile()
		self.run_whisper("base")
		self.run_nltk(f'{self.patient_dir}/{self.patient_name}_nltk_results.txt')

	def delete_files(self, files):
		for file in files:
			try:
				os.remove(file)
				self.logger.debug(f"{file} deleted successfully")
			except OSError as e:
				print(f"Error deleting {file}: {e}")


	def run_nltk(self, output_file):
		file = self.transcript
		with open(file, 'r') as f:
			all_text = f.read()

		sentences = nltk.sent_tokenize(all_text)
		fixed_sentences = [contractions.fix(sentence) for sentence in sentences]
		all_text_fixed = ' '.join(fixed_sentences)
		tokens = nltk.word_tokenize(all_text_fixed)
		table = str.maketrans('', '', string.punctuation)
		tokens = [word.translate(table).lower() for word in tokens if word.translate(table)]
		lemmatizer = nltk.stem.WordNetLemmatizer()
		lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]

		pos_tags = nltk.pos_tag(lemmatized_tokens)
		ner_tags = nltk.ne_chunk(pos_tags)
		sia = SentimentIntensityAnalyzer()
		sentiment_scores = [sia.polarity_scores(sentence) for sentence in lemmatized_tokens]
		dictionary = corpora.Dictionary([lemmatized_tokens])
		corpus = [dictionary.doc2bow(lemmatized_tokens)]
		lda_model = models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=10)
		topic_modeling_results = lda_model.print_topics()
		stop_words = set(stopwords.words('english'))
		filtered_text = [word for word in lemmatized_tokens if word.lower() not in stop_words]
		pos_tags = nltk.pos_tag(filtered_text)
		ner_tags = nltk.ne_chunk(pos_tags)
		sia = SentimentIntensityAnalyzer()
		sentiment_scores = [sia.polarity_scores(sentence) for sentence in filtered_text]
		dictionary = corpora.Dictionary([filtered_text])
		corpus = [dictionary.doc2bow(filtered_text)]
		lda_model = models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=10)
		topic_modeling_results = lda_model.print_topics()

		with open(output_file, 'w') as f:
			f.write("POS Tags: {}\n".format(pos_tags))
			f.write("NER Tags: {}\n".format(ner_tags))
			f.write("Sentiment Scores: {}\n".format(sentiment_scores))
			f.write("Topic Modeling Results: {}\n".format(topic_modeling_results))

	def clean_audio(self, start_p, start_d, start_t, stop_p, stop_d, stop_t):
		self.output_name = self.patient_name + "_cleaned"
		self.log_dir = self.patient_dir + "/logs"
		log_file = self.patient_name + "_ffmpeg.log"
		try:
			os.mkdir(self.log_dir)
		except OSError:
			self.logger.warning("The log directory already exists. Existing logs will be overwritten.")
		if os.path.isfile(f"{self.patient_dir}/{self.output_name}.mp3") == False:
			command = f'ffmpeg -i "{self.audio}" -af "silenceremove=start_periods=0:start_duration=1:stop_periods=-1:stop_duration=5" {self.patient_dir}/{self.output_name}.mp3 > {self.log_dir}/{log_file} 2>&1 < /dev/null'
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
					ct = 0
					for root, dirs, files in os.walk(interviews_path):
						for file in files:
							if (file.endswith('.mp3')or file.endswith('.mp4')) and not file.endswith('cleaned.mp3'):
								ct += 1
								all_args.append([os.path.join(root, file), log_level, run_mode])
					return ct, all_args
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
                if (not (filename.endswith('.mp3') or filename.endswith('.mp4')) or (filename.endswith('_cleaned.mp3'))):
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
	
	
	