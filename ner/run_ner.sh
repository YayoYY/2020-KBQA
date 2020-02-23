python ner/run_ner.py \
  --task=ner \
  --do_train=True \
  --do_eval=True \
  --do_test=True \
  --data_dir=data/NER \
  --bert_config_file==bert/chinese_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint==bert/chinese_L-12_H-768_A-12/bert_model.ckpt \
  --vocab_file==bert/chinese_L-12_H-768_A-12/vocab.txt \
  --output_dir=ner/output