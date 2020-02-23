'''
ner_model.py
命名实体识别模型
@Author: You Yue
@Reference: Macan (https://github.com/macanv/BERT-BiLSTM-CRF-NER)
'''

import sys
sys.path.append("..")

import codecs
import os
import pickle

import tensorflow as tf
from bert import tokenization
from ner.ner_helper import NERProcesser, InputExample, convert_single_example
from ner.ner_config import *

class NERModel(object):
    '''基于BERT的命名实体识别模型'''

    def __init__(self):

        self.label_list = NERProcesser().get_labels()
        self.tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file)

        # PROBLEM REMAIN: 模型名
        model_path = os.path.join(output_dir, model_file)
        if os.path.exists(model_path):
            self.predict_fn = tf.contrib.predictor.from_saved_model(model_path)

    def online_predict(self, question):

        predict_example = InputExample("id", ' '.join(list(question)), ' '.join(['O']*len(question)))
        feature = convert_single_example(0, predict_example, self.label_list,
                                         max_seq_length, self.tokenizer,
                                         output_dir, None)

        prediction = self.predict_fn({
            "input_ids": [feature.input_ids],
            "input_mask": [feature.input_mask],
            "segment_ids": [feature.segment_ids],
            "label_ids": [feature.label_ids],
        })["output"][0]

        with codecs.open(os.path.join(output_dir, 'label2id.pkl'), 'rb') as rf:
            label2id = pickle.load(rf)
            id2label = {value: key for key, value in label2id.items()}

        ids = [x for x in prediction if x != 0]
        labels = [id2label[id] for id in ids if id2label[id] not in ['[CLS]', '[SEP]']]
        labels = ' '.join(labels)

        return labels