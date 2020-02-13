'''
ner
命名实体识别模型
@Author: You Yue
@Reference: Macan (https://github.com/macanv/BERT-BiLSTM-CRF-NER)
'''
import codecs
import os

import tensorflow as tf
from bert import tokenization

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .txt files (or other data files) "
    "for the task.")

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid=None, text=None, label=None):
        """Constructs a InputExample.
        Args:
          guid: Unique id for the example.
          text: string. The untokenized text of the sequence.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids, ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        # self.label_mask = label_mask


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

class NERProcesser(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def __init__(self, output_dir):
        self.labels = set()
        self.output_dir = output_dir

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_data(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_data(os.path.join(data_dir, "dev.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_data(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self):
        """See base class."""
        return self.labels

    def _read_data(self, input_file):
        """Reads a BIO data.
        Args:
            1. input_file: 输入的文件名
        Outputs:
            1. lines, e.g. [['机 械 设 计 基 础 的 作 者 是 谁 ？', 'B I I I I I O O O O O O O O'], [...], [...]]
        """
        with codecs.open(input_file, 'r', encoding='utf-8') as f:
            lines = []
            words = []
            labels = []
            for line in f:
                contends = line.strip()
                tokens = contends.split(' ')
                if len(tokens) == 2:
                    words.append(tokens[0])
                    labels.append(tokens[-1])
                else:
                    if len(contends) == 0 and len(words) > 0:
                        for l in labels:
                            if len(l) > 0:
                                self.labels.add(l)
                        lines.append([' '.join(labels), ' '.join(words)])
                        words = []
                        labels = []
                        continue
                if contends.startswith("-DOCSTART-"):
                    continue
        return lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[0])
            examples.append(InputExample(guid=guid, text=text, label=label))
        return examples

class NERModel(object):
    '''基于BERT的命名实体识别模型'''

    pass

def main(_):
    processor = NERProcesser(FLAGS.data_dir)


if __name__ == '__main__':
    tf.app.run()