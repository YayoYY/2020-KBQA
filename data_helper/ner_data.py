'''
ner_data
构建用于命名实体识别任务的数据集（train & dev）
@Author: You Yue
'''

import re
from data_helper.data_config import *

# 1. 读取训练数据
f = open(train_data_path, 'r')
contents = f.readlines()
f.close()

# 2. 提取question、triple和answer
questions = [question for question in contents if 'question' in question]
triples = [triple for triple in contents if 'triple' in triple]
assert len(questions) == len(triples)

reg_q = r'(<question id=\d*>)\s(.*)'
questions = [re.match(reg_q, question).group(2) for question in questions]
reg_t = r'(<triple id=\d*>)\s(.*)'
triples = [re.match(reg_t, triple).group(2) for triple in triples]
assert len(questions) == len(triples)

# 3. BIO标注
entities = [triple.split('|||')[0].strip() for triple in triples]
tags = []
for question, entity in zip(questions, entities):
    if entity in question:
        e_len, q_len = len(entity), len(question)
        idx = question.index(entity)
        tag = 'O' * idx + 'B' + 'I'*(e_len-1)
        tag = tag + 'O' * (q_len-len(tag))
        tags.append([question, tag])

# 4. 保存
f = open(ner_train_data_path, 'w')
for question, tag in tags[:13000]:
    for token, type in zip(question, tag):
        f.write('{0} {1}\n'.format(token, type))
    f.write('\n')
f.close()

f = open(ner_dev_data_path, 'w')
for question, tag in tags[13000:]:
    for token, type in zip(question, tag):
        f.write('{0} {1}\n'.format(token, type))
    f.write('\n')
f.close()