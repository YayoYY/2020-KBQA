'''
run_sim.py
文本相似度模型训练
@Author: You Yue
'''

import sys
sys.path.append("..")

import codecs
import os
import pickle

import tensorflow as tf
from bert import tokenization, modeling
from sim.sim_helper import *

flags = tf.flags
FLAGS = flags.FLAGS

# 1. 任务配置
flags.DEFINE_string(
    "task", None,
    "Task name.")
flags.DEFINE_boolean(
    "do_train", True,
    "Whether to run training.")
flags.DEFINE_boolean(
    "do_eval", True,
    "Whether to run evaluating.")
flags.DEFINE_boolean(
    "do_test", False,
    "Whether to run testing.")
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .txt files (or other data files) ")
flags.DEFINE_string(
    "output_dir", None,
    "The pretrained BERT model dir.")
flags.DEFINE_string(
    "bert_config_file", None,
    "Bert config file dir.")
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model)")
flags.DEFINE_string(
    "vocab_file", None,
    "The vocab.txt path.")

# 2. 训练参数
flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after tokenization.")
flags.DEFINE_string(
    "cell", "lstm",
    "Which rnn cell used.")
flags.DEFINE_integer(
    "lstm_size", 128,
    "Size of lstm units.")
flags.DEFINE_integer(
    "num_layers", 1,
    "Number of rnn layers.")
flags.DEFINE_float(
    "dropout_rate", 0.5,
    "Dropout rate.")
flags.DEFINE_float(
    "learning_rate", 1e-5,
    "The initial learning rate for Adam Optimizer.")
flags.DEFINE_integer(
    "batch_size", 64,
    "Batch size.")
flags.DEFINE_integer(
    "num_train_epochs", 10,
    "Total number of training epochs to perform.")
flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for.")
flags.DEFINE_integer(
    "save_summary_steps", "500",
    "Save summary steps.")
flags.DEFINE_string(
    "save_checkpoints_steps", "500",
    "Save checkpoints steps.")

def main(_):

    # 1. 配置、检查
    tf.logging.set_verbosity(tf.logging.INFO)

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_test:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    tf.gfile.MakeDirs(FLAGS.output_dir)

    # 2. 创建estimator
    processor = SimProcessor()

    label_list = processor.get_labels()
    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file)

    run_config = tf.estimator.RunConfig(
        model_dir=FLAGS.output_dir,
        save_summary_steps=500,
        save_checkpoints_steps=500,
        session_config=tf.ConfigProto(
            log_device_placement=False,
            inter_op_parallelism_threads=0,
            intra_op_parallelism_threads=0,
            allow_soft_placement=True))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None

    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        num_train_steps = int(
            len(train_examples) / FLAGS.batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps)

    estimator = tf.estimator.Estimator(
        model_fn,
        params={'batch_size': FLAGS.batch_size},
        config=run_config)

    # 3. 训练、验证、预测
    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        file_based_convert_examples_to_features(
            train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if FLAGS.do_eval:
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        num_actual_eval_examples = len(eval_examples)

        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        file_based_convert_examples_to_features(
            eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(eval_examples), num_actual_eval_examples,
                        len(eval_examples) - num_actual_eval_examples)
        tf.logging.info("  Batch size = %d", FLAGS.batch_size)

        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=False)

        result = estimator.evaluate(input_fn=eval_input_fn, steps=None)

        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    if FLAGS.do_test:
        predict_examples = processor.get_test_examples(FLAGS.data_dir)
        num_actual_predict_examples = len(predict_examples)

        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        file_based_convert_examples_to_features(predict_examples, label_list,
                                                FLAGS.max_seq_length, tokenizer,
                                                predict_file)

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(predict_examples), num_actual_predict_examples,
                        len(predict_examples) - num_actual_predict_examples)
        tf.logging.info("  Batch size = %d", FLAGS.batch_size)

        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=False)

        result = estimator.predict(input_fn=predict_input_fn)

        output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv")
        with tf.gfile.GFile(output_predict_file, "w") as writer:
            num_written_lines = 0
            tf.logging.info("***** Predict results *****")
            for (i, prediction) in enumerate(result):
                probabilities = prediction["probabilities"]
                if i >= num_actual_predict_examples:
                    break
                output_line = "\t".join(
                    str(class_probability)
                    for class_probability in probabilities) + "\n"
                writer.write(output_line)
                num_written_lines += 1
        assert num_written_lines == num_actual_predict_examples

    # 4. 将模型保存为pb格式
    def serving_input_fn():
        label_id = tf.placeholder(tf.int32, [None], name='label_id')
        input_ids = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name='input_ids')
        input_mask = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name='input_mask')
        segment_ids = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name='segment_ids')
        input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
            'label_id': label_id,
            'input_ids': input_ids,
            'input_mask': input_mask,
            'segment_ids': segment_ids,
        })()
        return input_fn

    estimator._export_to_tpu = False
    out_put_model_file = os.path.join(FLAGS.output_dir, "model")
    estimator.export_savedmodel(out_put_model_file, serving_input_fn)


if __name__ == '__main__':
    flags.mark_flag_as_required("task")
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("output_dir")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("init_checkpoint")
    flags.mark_flag_as_required("vocab_file")
    tf.app.run()