'''
sim_data
构建用于文本匹配任务的数据集（train & dev）
@Author: You Yue
'''

import re
import random

from data_helper.data_config import *

# 1. 正样本
# (1) 读取数据
f = open(train_data_path, 'r')
contents = f.readlines()
f.close()

# (2) 提取question、attribute
questions = [question for question in contents if 'question' in question]
reg_q = r'(<question id=\d*>)\s(.*)'
questions = [re.match(reg_q, question).group(2) for question in questions]

triples = [triple for triple in contents if 'triple' in triple]
reg_t = r'(<triple id=\d*>)\s(.*)'
triples = [re.match(reg_t, triple).group(2) for triple in triples]

attributes = [triple.split('|||')[1].strip() for triple in triples]

pos_samples = list(zip(questions, attributes))

# 2. 负采样与保存
f = open(sim_train_data_path, 'w')
for i, question in enumerate(questions[:13000]):
    attributes_5 = random.sample(attributes, 5)
    neg_samples = [(question, attribute) for attribute in attributes_5]
    pos_sample = '{0} {1}\n'.format(' '.join(pos_samples[i]), '1')
    neg_samples = ['{0} {1}'.format(' '.join(sample), '0') for sample in neg_samples]
    neg_samples = '\n'.join(neg_samples) + '\n'
    f.write(pos_sample + neg_samples)
f.close()

f = open(sim_dev_data_path, 'w')
for i, question in enumerate(questions[13000:], 13000):
    attributes_5 = random.sample(attributes, 5)
    neg_samples = [(question, attribute) for attribute in attributes_5]
    pos_sample = '{0} {1}\n'.format(' '.join(pos_samples[i]), '1')
    neg_samples = ['{0} {1}'.format(' '.join(sample), '0') for sample in neg_samples]
    neg_samples = '\n'.join(neg_samples) + '\n'
    f.write(pos_sample + neg_samples)
f.close()