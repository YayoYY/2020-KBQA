'''
ner
命名实体识别模型与文本匹配模型的训练
@Author: You Yue
'''

import tensorflow as tf
from train import ner
from train import sim

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "task", None,
    "Task type, ner or sim.")
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .txt files (or other data files) ")
flags.DEFINE_string(
    "vocab_file", None,
    "The vocab.txt path.")
flags.DEFINE_string(
    "output_dir", None,
    "The pretrained BERT model dir.")

flags.DEFINE_float(
    "dropout_rate", 0.5,
    "Dropout rate.")
flags.DEFINE_integer(
    "lstm_size", 128,
    "Size of lstm units.")
flags.DEFINE_string(
    "cell", "lstm",
    "Which rnn cell used.")
flags.DEFINE_integer(
    "num_layers", 1,
    "Number of rnn layers.")
flags.DEFINE_integer(
    "save_summary_steps", "500",
    "Save summary steps.")
flags.DEFINE_boolean(
    "do_lower_case", "True",
    "Whether do lower case or not.")

# bert_config_file
# output_dir
# max_seq_length
# vocab_file
# do_lower_case
# cell
# num_layers
# init_checkpoint
# batch_size
# clean

def main(_):




if __name__ == '__main__':
    tf.app.run()