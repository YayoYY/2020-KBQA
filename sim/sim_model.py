'''
sim_model.py
文本相似度模型
@Author: You Yue
'''

import sys
sys.path.append("..")

import codecs
import os
import pickle

import tensorflow as tf
from bert import tokenization
from sim.sim_helper import SimProcessor, InputExample, convert_single_example
from sim.sim_config import *

class SIMModel(object):
    '''基于BERT的文本相似度模型'''

    def __init__(self):

        self.label_list = SimProcessor().get_labels()
        self.tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file)

        # PROBLEM REMAIN: 模型名
        model_path = os.path.join(output_dir, model_file)
        if os.path.exists(model_path):
            self.predict_fn = tf.contrib.predictor.from_saved_model(model_path)

    def predict(self, question, attribute):

        predict_example = InputExample("id", question, attribute, "0")
        feature = convert_single_example(0, predict_example, self.label_list,
                                         max_seq_length, self.tokenizer)

        prediction = self.predict_fn({
            "input_ids": [feature.input_ids],
            "input_mask": [feature.input_mask],
            "segment_ids": [feature.segment_ids],
            "label_id": [feature.label_id]
        })['probabilities'][0][1]

        return prediction