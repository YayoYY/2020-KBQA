# 基于BERT的KBQA系统

## 项目架构

利用BERT+BiLSTM+CRF进行命名实体识别，用以识别实体。利用BERT训练句子相似度计模型，用以关系抽取。

## 运行说明

下载文件：

- Google官方BERT：https://github.com/google-research/bert
- BERT中文配置文件：[chinese_L-12_H-768_A-12](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)
- NIPCC数据集

训练命名实体识别模型（NER）：

```shell
$ cd ner
$ ./run_ner.sh
```

## 环境

- python 3.6.5
- tensorflow 1.14