'''
kbqa.py
基于知识库的问答程序
@Author: You Yue
'''

import os
import pymysql

from data_helper.data_config import *
from ner.ner_model import NERModel
from sim.sim_model import SIMModel

os.environ['TF_APP_MIN_LOG_LEVEL'] = '3' # 屏蔽INFO WARNING WRONG

def init_model():
    return NERModel(), SIMModel()

def entity_recognize(ner_model, question):
    '''命名实体识别
    Args:
        ner_model: 命名实体模型
        question: 问题, str
    Return:
        entity: 实体, str
    '''
    tags = ner_model.predict(question).split(" ")
    entity = ''
    # PROBLEM REMAIN: 只提取了第一个实体
    for i, item in enumerate(question):
        if entity == '' and tags[i] == 'B':
            entity += item
        elif entity != '' and tags[i] == 'I':
            entity += item
        elif entity != '' and tags[i] == 'O':
            break
    return entity

def attributes_extract(entity):
    '''属性与属性值抽取
    Args:
        entity: 实体, str
    Return:
        attributes: 属性及属性值, dict, {attribute1: attribute1_value, attribute2: attribute2_value...}
    '''
    db = pymysql.connect(db_ip, db_username, db_password, db_name)
    cursor = db.cursor()
    sql = "SELECT * FROM kb WHERE entity='{0}';".format(entity)
    cursor.execute(sql)
    results = cursor.fetchall()
    attributes = {}
    for row in results:
        attributes[row[1]] = row[2]
    return attributes

def rank(sim_model, question, attributes):
    '''属性值排序
    Args:
        sim_model: 相似度模型
        question: 问题
        attributes: 属性值
    Return:
        attr_sort: 排序后的属性值
        pos_sort: 排序后的概率值
    '''
    predict = lambda x,y :sim_model.predict(x, y)
    pos_attr = {k:v for k,v in zip([predict(question, y) for y in attributes.keys()], attributes.keys())}
    pos_sort = sorted(pos_attr.keys(), reverse=True)
    attr_sort = [pos_attr[pos] for pos in pos_sort]
    return attr_sort, pos_sort

def kbqa(ner_model, sim_model, question):
    '''
    Args:
        ner_model: 命名实体识别
        sim_model: 文本相似度模型
        question: 问题文本
    Return:
        answer: 最可能的答案
        pos: 答案对应的概率
    '''

    entity = entity_recognize(ner_model, question)
    attributes = attributes_extract(entity)
    attr_sort, pos_sort = rank(sim_model, question, attributes)
    answer, pos = attributes[attr_sort[0]], pos_sort[0]
    return answer, pos

if __name__ == '__main__':

    ner_model, sim_model = init_model()

    question = input("Type your question:")
    answer, pos = kbqa(ner_model, sim_model, question)

    msg = "Answers: {0}, ({1}%)".format(answer, str(round(pos*100, 2)))
    print(msg)
