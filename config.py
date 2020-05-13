import os
import tensorflow as tf

from prepro import prepro
from main import train, test

flags = tf.flags
dataset = 'didemo'
feature = 'rgb'
feature_file = 'average_global_flow.h5'
feature_dim = 1024


dataset_target ="no_gate_ddm_dropout_07_dim_75_band2"

home = os.path.expanduser("/data1/view/jychen")
train_file = os.path.join(home, "data", "eccv", dataset, "train_data.json")
dev_file = os.path.join(home, "data", "eccv", dataset, "val_data.json")
test_file = os.path.join(home, "data", "eccv", dataset, "test_data.json")
video_file = os.path.join(home, "data", "eccv", dataset, feature_file)

glove_word_file = os.path.join(home, "data", "glove", "glove.840B.300d.txt")

target_dir = "data/"+ dataset + "/" + feature
log_dir = "log/event/"+ dataset_target + "/" + feature 
save_dir = "log/model/" + dataset_target + "/" + feature
answer_dir = "log/answer/" + dataset_target + "/" + feature
train_record_file = os.path.join(target_dir, "flow_train.tfrecords")
dev_record_file = os.path.join(target_dir, "flow_dev.tfrecords")
test_record_file = os.path.join(target_dir, "flow_test.tfrecords")

meta_dir = "data/" + dataset
word_emb_file = os.path.join(meta_dir, "word_emb.json")
char_emb_file = os.path.join(meta_dir, "char_emb.json")
idx2word_file = os.path.join(meta_dir, "idx2word.json")
train_eval = os.path.join(meta_dir, "train_eval.json")
dev_eval = os.path.join(meta_dir, "dev_eval.json")
test_eval = os.path.join(meta_dir, "test_eval.json")
dev_meta = os.path.join(meta_dir, "dev_meta.json")
test_meta = os.path.join(meta_dir, "test_meta.json")
answer_file = os.path.join(answer_dir, "answer.json")

if not os.path.exists(target_dir):
	os.makedirs(target_dir)
if not os.path.exists(log_dir):
	os.makedirs(log_dir)
if not os.path.exists(save_dir):
	os.makedirs(save_dir)
if not os.path.exists(answer_dir):
	os.makedirs(answer_dir)

flags.DEFINE_string("mode","train","Running mode train/debug/test")

flags.DEFINE_string("target_dir", target_dir, "Target directory for out data")
flags.DEFINE_string("log_dir", log_dir, "Directory for tf event")
flags.DEFINE_string("save_dir", save_dir, "Directory for saving model")
flags.DEFINE_string("train_file", train_file, "Train source file")
flags.DEFINE_string("dev_file", dev_file, "Dev source file")
flags.DEFINE_string("test_file", test_file, "Test source file")
flags.DEFINE_string("glove_word_file", glove_word_file, "Glove word embedding source file")
flags.DEFINE_string("video_file", video_file, "Video feature file")

flags.DEFINE_string("train_record_file", train_record_file, "Out file for train data")
flags.DEFINE_string("dev_record_file", dev_record_file, "Out file for dev data")
flags.DEFINE_string("test_record_file", test_record_file, "Out file for test data")
flags.DEFINE_string("word_emb_file", word_emb_file, "Out file for word embedding")
flags.DEFINE_string("char_emb_file", char_emb_file, "Out file for char embedding")
flags.DEFINE_string("train_eval_file", train_eval, "Out file for train eval")
flags.DEFINE_string("dev_eval_file", dev_eval, "Out file for dev eval")
flags.DEFINE_string("test_eval_file", test_eval, "Out file for test eval")
flags.DEFINE_string("dev_meta", dev_meta, "Out file for dev meta")
flags.DEFINE_string("test_meta", test_meta, "Out file for test meta")
flags.DEFINE_string("answer_file", answer_file, "Out file for answer")

flags.DEFINE_string("glove_char_size", 94, "Corpus size for Glove")
flags.DEFINE_string("glove_word_size", int(2.2e6), "Corpus size for Glove")
flags.DEFINE_string("glove_dim", 300, "Embedding dimension for Glove")
flags.DEFINE_string("char_dim", 8, "Embedding dimension for char")

flags.DEFINE_string("des_limit", 30, "Limit length for description")
flags.DEFINE_string("frame_limit", 6, "Limit length for video")
flags.DEFINE_string("test_frame_limit", 6, "Limit Length for test video")
flags.DEFINE_string("char_limit", 16, "Limit length for character")
flags.DEFINE_string("test_char_limit", 16, "Limit length for character")
flags.DEFINE_string("word_limit", 15, "Limit length for word")
flags.DEFINE_string("test_word_limit", 15, "Limit length for word")
flags.DEFINE_string("word_count_limit", -1, "Min count for word")
flags.DEFINE_string("char_count_limit", -1, "Min count for char")

flags.DEFINE_string("frame_vec_size", feature_dim, "frame feature dimension")

flags.DEFINE_integer("capacity", 15000, "Batch size of dataset shuffle")
flags.DEFINE_integer("num_threads", 4, "Number of thread in input pipeline")
flags.DEFINE_boolean("use_cudnn", True, "Whether to use cudnn rnn")
flags.DEFINE_boolean("is_bucket", False, "build bucket batch iterator or not")

flags.DEFINE_integer("batch_size", 64, "Batch size")
flags.DEFINE_integer("num_steps", 50000, "Number of steps")
flags.DEFINE_integer("checkpoint", 515, "checkpoint to save and evaluate the model")
flags.DEFINE_integer("period", 100, "period to save batch loss")
flags.DEFINE_integer("val_num_batches", 150, "Number of batches to evaluate the model")
flags.DEFINE_integer("init_lr", 0.001, "Initial learning rate for Adadelta")
flags.DEFINE_integer("keep_prob", 0.7, "Dropout keep prob in rnn")
flags.DEFINE_integer("ptr_keep_prob", 0.7, "Dropout keep prob for pointer network")
flags.DEFINE_integer("grad_clip", 5.0, "Global Norm gradient clipping rate")
flags.DEFINE_integer("hidden", 75, "Hidden size")
flags.DEFINE_integer("char_hidden", 100, "GRU dimention for char")
flags.DEFINE_integer("patience", 3, "Patience for learning rate decay")

glove_char_file = os.path.join(home, "data", "glove", "glove.840B.300d-char.txt")
flags.DEFINE_string("glove_char_file", glove_char_file, "Glove character embedding source file")
flags.DEFINE_string("pretrained_char", False, "Whether to use pretrained character embedding")

fasttext_file = os.path.join(home, "data", "fasttext", "wiki-news-300d-1M.vec")
flags.DEFINE_string("fasttext_file", fasttext_file, "Fasttext word embedding source file")
flags.DEFINE_string("fasttext", False, "Whether to use fasttext")

flags.DEFINE_string("idx2word_file", idx2word_file, "Out file for idx2word")

def main(_):
	config = flags.FLAGS
	if config.mode == "train":
		train(config)
	elif config.mode == "prepro":
		prepro(config)
	elif config.mode == "debug":
		config.num_steps = 2
		config.val_num_batches = 1
		config.checkpoint = 1
		config.period = 1
		train(config)
	elif config.mode == "test":
		if config.use_cudnn:
			print("Warning: Due to a known bug in TF, the paras of CudnnGRU may not be properly restored.")
		test(config)
	else:
		print("Unknown mode")
		exit(0)

if __name__ == "__main__":


	tf.app.run()
