python run_sim.py \
  --task=sim \
  --do_train=False \
  --do_eval=False \
  --do_test=True \
  --data_dir=../data/SIM \
  --output_dir=output \
  --bert_config_file=../bert/chinese_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=../bert/chinese_L-12_H-768_A-12/bert_model.ckpt \
  --vocab_file=../bert/chinese_L-12_H-768_A-12/vocab.txt \
