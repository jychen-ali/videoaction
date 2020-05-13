import re
import tensorflow as tf
import random
from tqdm import tqdm
import spacy
import ujson as json
from collections import Counter
import collections
import numpy as np
import h5py
import random


nlp = spacy.blank("en")

def word_tokenize(sent):
	doc = nlp(sent)
	return [token.text for token in doc]

def read_json(json_file):
	with open(json_file) as data_file:
		data = json.load(data_file)
	return data

def convert_idx(text, tokens):
	current = 0
	spans = []
	for token in tokens:
		current = text.find(token, current)
		if current < 0:
			print("Token {} cannot be found".format(token))
			raise Exception()
		spans.append((current, current+len(token)))
		current += len(token)

	return spans

def visualFeatureExtractor(filename):
	features_h5py = h5py.File(filename)
	features = {}
	for key in features_h5py.keys():
		features[key] = np.array(features_h5py[key],dtype="float32")
	features_h5py.close()
	return features

def process_file_test(filename, data_type, word_counter, char_counter):
	print("Generating {} examples...".format(data_type))
	examples = []
	total = 0

	data_orig = read_json(filename)
	for sample in data_orig:
		id = sample['video']
		description = filtering(sample['description'])
		description_tokens = word_tokenize(description)
		description_chars = [list(token) for token in description_tokens]
		for token in description_tokens:
			word_counter[token] += 1
			for char in token:
				char_counter[char] += 1
		times_annotations = sample['times']
		t1 = times_annotations[0][0]
		t2 = times_annotations[0][1]
		example = {"id":id, "des_tokens": description_tokens, "des_chars": description_chars, "t_s":t1, "t_e":t2, "times":times_annotations}
		examples.append(example)
	print("{} questions in total".format(data_type))
	return examples		

def filtering(seq):
	seq = seq.lower()
	seq = seq.replace(',', ' ')
	seq = seq.replace('\n', ' ').replace('"', ' ')
	seq = seq.replace('.', ' ').replace('?', ' ').replace('!', ' ')
	seq = seq.replace('``', ' ').replace('`', ' ').replace("''", ' ')
	seq = seq.replace(':', ' ').replace('-', ' ').replace('--', ' ')
	seq = seq.replace('...', ' ').replace(';', ' ')
	seq = seq.replace('(', ' ').replace(')', ' ').replace('[', ' ').replace(']', ' ')
	seq = seq.replace('@', ' ').replace('#', ' ').replace('$', ' ').replace('&', ' ').replace('*', ' ')
	seq = seq.replace('\\', ' ').replace('/', ' ')
	seq = seq.replace('1', ' ').replace('2', ' ').replace('3', ' ').replace('4', ' ').replace('5', ' ')
	seq = seq.replace('6', ' ').replace('7', ' ').replace('8', ' ').replace('9', ' ').replace('0', ' ')

	seq = re.sub(' +', ' ', seq)

	return seq

def process_file(filename, data_type, word_counter, char_counter):
	print("Generating {} examples...".format(data_type))
	examples = []
	total = 0
	
	data_orig = read_json(filename)
	
	for sample in data_orig:
		id = sample['video']
		description = filtering(sample['description'])
		description_tokens = word_tokenize(description)
		description_chars = [list(token) for token in description_tokens]
		for token in description_tokens:
			word_counter[token] += 1
			for char in token:
				char_counter[char] += 1
		times_annotations = sample['times']
	
		
		temp_count = 0
		temp_t = []
		for times_annotation in times_annotations:
			cc = times_annotations.count(times_annotation)
			if cc > temp_count:
				temp_t = times_annotation
				temp_count = cc
		[t1,t2] = temp_t
		example = {"id": id, "des_tokens": description_tokens, "des_chars": description_chars, "t_s": t1, "t_e": t2}
		examples.append(example)

	random.shuffle(examples)
	print("{} questions in total".format(len(examples)))
	return examples

def get_embedding(counter, data_type, limit=-1, emb_file=None, size=None, vec_size=None):
	print("Generating {} embedding ...".format(data_type))
	embedding_dict = {}
	filtered_elements = [k for k,v in counter.items() if v>limit]
	if emb_file is not None:
		assert size is not None
		assert vec_size is not None
		with open(emb_file, "r", encoding="utf-8") as fh:
			for line in tqdm(fh, total=size):
				array = line.split()
				word = "".join(array[0:-vec_size])
				vector = list(map(float, array[-vec_size:]))
				if word in counter and counter[word]>limit:
					embedding_dict[word] = vector
	
		
		print("{} / {} tokens have corresponding {} embedding vector".format(len(embedding_dict), len(filtered_elements),data_type))

	else:
		assert vec_size is not None
		for token in filtered_elements:
			embedding_dict[token] = [np.random.normal(scale=0.1) for _ in range(vec_size)]


		print("{} tokens have corresponding embedding vector".format(len(filtered_elements)))

	NULL = "--NULL--"
	OOV = "--OOV--"
	token2idx_dict = {token: idx for idx, token in enumerate(embedding_dict.keys(), 2)}
	token2idx_dict[NULL] = 0
	token2idx_dict[OOV] = 1
	idx2token_dict = {idx: token for idx, token in enumerate(embedding_dict.keys(), 2)}
	embedding_dict[NULL] = [0. for _ in range(vec_size)]
	embedding_dict[OOV] = [0. for _ in range(vec_size)]
	idx2emb_dict = {idx: embedding_dict[token] for token, idx in token2idx_dict.items()}
	emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]

	return emb_mat, token2idx_dict, idx2token_dict

def build_features_test(config, vfeat, examples, data_type, out_file, word2idx_dict, char2idx_dict, is_test=True):
	frame_limit = config.test_frame_limit
	word_limit = config.test_word_limit
	char_limit = config.char_limit

	def filter_func(examples, is_test=True):
		return len(example["des_tokens"]) > word_limit
	print("Processing {} examples...".format(data_type))
	writer = tf.python_io.TFRecordWriter(out_file)
	total = 0
	total_ = 0
	meta = {}
	for example in tqdm(examples):
		total_ += 1

		if filter_func(example, is_test):
			continue
		total += 1
		des_idxs = np.zeros([word_limit], dtype=np.int32)
		des_char_idxs = np.zeros([word_limit, char_limit], dtype=np.int32)

		y1 = np.zeros([frame_limit], dtype=np.float32)
		y2 = np.zeros([frame_limit], dtype=np.float32)
		v = np.zeros([frame_limit], dtype=np.int32)
		mmask = np.ones([frame_limit,frame_limit],dtype=np.int32)
		lmask = np.tril(mmask)
		rmask = np.triu(mmask)
		def _get_word(word):
			for each in (word, word.lower(), word.capitalize(), word.upper()):
				if each in word2idx_dict:
					return word2idx_dict[each]
			return 1
		def _get_char(char):
			for each in char2idx_dict:
				return char2idx_dict[char]
			return 1

		for i, token in enumerate(example["des_tokens"]):
			des_idxs[i] = _get_word(token)
		for i, token in enumerate(example["des_chars"]):
			for j, char in enumerate(token):
				if j == char_limit:
					break
				des_char_idxs[i,j] = _get_char(char)
		start, end = int(example["t_s"]), int(example["t_e"])
		y1[start], y2[end] = 1.0, 1.0
		times_annotations = example["times"]
		vfeat_temp = vfeat[example["id"]] #np.append(vfeat[example["id"]], np.zeros((1,config.frame_vec_size), dtype=np.float32), axis=0)
		assert vfeat_temp.shape == (6,1024)
		v[:vfeat_temp.shape[0]] = 1
		record = tf.train.Example(features=tf.train.Features(feature={
			"id": tf.train.Feature(int64_list=tf.train.Int64List(value=[total])),
			"des_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[des_idxs.tostring()])),
			"des_char_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[des_char_idxs.tostring()])),
			"y1": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y1.tostring()])),
			"y2": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y2.tostring()])), 
			"vfeat": tf.train.Feature(bytes_list=tf.train.BytesList(value=[vfeat_temp.tostring()])),
			"v": tf.train.Feature(bytes_list=tf.train.BytesList(value=[v.tostring()])),
			"lmask": tf.train.Feature(bytes_list=tf.train.BytesList(value=[lmask.tostring()])),
			"rmask": tf.train.Feature(bytes_list=tf.train.BytesList(value=[rmask.tostring()]))
		}))
		meta[total] = times_annotations
		writer.write(record.SerializeToString())
	print("Build {} / {} instances of features in total".format(total, total_))


#	meta["total"] = total
	writer.close()
	return meta		

def build_features(config, vfeat, examples, data_type, out_file, word2idx_dict, char2idx_dict, is_test=False):
	frame_limit = config.test_frame_limit if is_test else config.frame_limit
	word_limit = config.test_word_limit if is_test else config.word_limit
	char_limit = config.char_limit

	def filter_func(examples, is_test=False):
		return len(example["des_tokens"]) > word_limit
	print("Processing {} examples...".format(data_type))
	writer = tf.python_io.TFRecordWriter(out_file)
	total = 0
	total_ = 0
	meta = {}


	for example in tqdm(examples):
		total_ += 1
		
		if filter_func(example, is_test):
			continue

		total += 1
		des_idxs = np.zeros([word_limit], dtype=np.int32)
		des_char_idxs = np.zeros([word_limit, char_limit], dtype=np.int32)
		
		y1 = np.zeros([frame_limit],dtype=np.float32)
		y2 = np.zeros([frame_limit],dtype=np.float32)
		v = np.zeros([frame_limit],dtype=np.int32)
		mmask = np.ones([frame_limit,frame_limit],dtype=np.int32)
		lmask = np.tril(mmask)
		rmask = np.triu(mmask)
		def _get_word(word):
			for each in (word, word.lower(), word.capitalize(), word.upper()):
				if each in word2idx_dict:
					return word2idx_dict[each]
			return 1

		def _get_char(char):
			if char in char2idx_dict:
				return char2idx_dict[char]
			return 1

		for i, token in enumerate(example["des_tokens"]):
			des_idxs[i] = _get_word(token)
		for i, token in enumerate(example["des_chars"]):
			for j, char in enumerate(token):
				if j == char_limit:
					break
				des_char_idxs[i,j] = _get_char(char)
		start, end = int(example["t_s"]), int(example["t_e"])
		y1[start], y2[end] = 1.0, 1.0
		assert vfeat[example["id"]].shape == (6,1024)


		vfeat_temp = vfeat[example["id"]]#np.append(vfeat[example["id"]], np.zeros((1,config.frame_vec_size), dtype=np.float32), axis=0)
		#assert vfeat_temp.shape == (7,1024)
		v[:vfeat_temp.shape[0]] = 1
		record = tf.train.Example(features=tf.train.Features(feature={
					"des_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[des_idxs.tostring()])),
					"des_char_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[des_char_idxs.tostring()])),
					"y1": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y1.tostring()])),
					"y2": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y2.tostring()])),
					"vfeat": tf.train.Feature(bytes_list=tf.train.BytesList(value=[vfeat_temp.tostring()])),
					"v": tf.train.Feature(bytes_list=tf.train.BytesList(value=[v.tostring()])),
					"lmask": tf.train.Feature(bytes_list=tf.train.BytesList(value=[lmask.tostring()])),
					"rmask": tf.train.Feature(bytes_list=tf.train.BytesList(value=[rmask.tostring()]))
					}))
		writer.write(record.SerializeToString())
	print("Build {} / {} instances of features in total".format(total, total_))
	meta["total"] = total
	writer.close()
	return meta

def save(filename, obj, message=None):
	if message is not None:
		print("Saving {} ...".format(message))
		with open(filename, "w") as fh:
			json.dump(obj, fh)

def prepro(config):
	word_counter, char_counter = Counter(), Counter()
	

	

	train_examples = process_file(config.train_file, "train", word_counter, char_counter)
	dev_examples = process_file_test(config.dev_file, "dev", word_counter, char_counter)
	test_examples = process_file_test(config.test_file, "test", word_counter, char_counter)

	vfeat = visualFeatureExtractor(config.video_file)

	word_emb_file = config.fasttext_file if config.fasttext else config.glove_word_file
	char_emb_file = config.glove_char_file if config.pretrained_char else None
	char_emb_size = config.glove_char_size if config.pretrained_char else None
	char_emb_dim = config.glove_dim if config.pretrained_char else config.char_dim

	word_emb_mat, word2idx_dict, idx2word_dict = get_embedding(word_counter, "word", emb_file=word_emb_file, size=config.glove_word_size, vec_size=config.glove_dim)
	char_emb_mat, char2idx_dict, _ = get_embedding(char_counter, "char", emb_file=char_emb_file, size=char_emb_size, vec_size=char_emb_dim)

	build_features(config, vfeat, train_examples, "train", config.train_record_file, word2idx_dict, char2idx_dict)
	dev_meta = build_features_test(config, vfeat, dev_examples, "dev", config.dev_record_file, word2idx_dict, char2idx_dict)
	test_meta = build_features_test(config, vfeat, test_examples, "test", config.test_record_file, word2idx_dict, char2idx_dict)

	save(config.word_emb_file, word_emb_mat, message="word embedding")
	save(config.char_emb_file, char_emb_mat, message="char embedding")

	save(config.idx2word_file, idx2word_dict, message="idx to word")

	save(config.dev_meta, dev_meta, message="dev meta")
	save(config.test_meta, test_meta, message="test meta")
