'''
kb_data
构建知识库
@Author: You Yue
'''

from data_helper.data_config import *
import pymysql

db = pymysql.connect(db_ip, db_username, db_password, db_name)
cursor = db.cursor()

def preprocess(value):
    '''字符串预处理（主要去掉引号）'''
    value.replace('''"''', "")
    return value

with open(kb_train_data_path, 'r') as f:
    for line in f:
        try:
            line = line.strip()
            entity, attribute, value = line.split(" ||| ")
            value = preprocess(value)
            sql = "INSERT INTO kb VALUES ('{0}', '{1}', '{2}');".format(entity, attribute, value)
            cursor.execute(sql)
            db.commit()
        except pymysql.err.IntegrityError:
            continue

db.close()