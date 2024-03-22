# CODing:utf-8

'''import bert_model.modeling as modeling
import bert_model.tokenization as tokenization
from bert_model.run_classifier import create_model, file_based_input_fn_builder, ColaProcessor, \
    file_based_convert_examples_to_features
import tensorflow as tf
import bert_model.optimization as optimization
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef, roc_auc_score, accuracy_score, \
    confusion_matrix, roc_curve
import os
import time
import sys
import copy
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


tf.logging.set_verbosity(tf.logging.INFO)
batch_size = 32
use_tpu = False
seq_length = 128
vocab_file = "./bert_model/vocab_1kmer.txt"
init_checkpoint = "./bert_model/model.ckpt"
bert_config = modeling.BertConfig.from_json_file("./bert_model/bert_config_1.json")
learning_rate = 2e-5
num_train_steps = 100
num_warmup_steps = 10

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.75

input_file = "predict.tf_record"
tokenizer = tokenization.FullTokenizer(
    vocab_file=vocab_file, do_lower_case=True)
input_ids = tf.placeholder(dtype=tf.int32, shape=(None, 128))
input_mask = tf.placeholder(dtype=tf.int32, shape=(None, 128))
segment_ids = tf.placeholder(dtype=tf.int32, shape=(None, 128))
label_ids = tf.placeholder(dtype=tf.int32, shape=(None,))
is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)
num_labels = 2
use_one_hot_embeddings = False
is_training = False

(total_loss, per_example_loss, logits, probabilities) = create_model(
    bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
    num_labels, use_one_hot_embeddings)
tvars = tf.trainable_variables()
initialized_variable_names = {}
scaffold_fn = None

if init_checkpoint:
    (assignment_map, initialized_variable_names
     ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

train_op = optimization.create_optimizer(
    total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

name_to_features = {
    "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
    "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
    "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
    "label_ids": tf.FixedLenFeature([], tf.int64),
    "is_real_example": tf.FixedLenFeature([], tf.int64),
}
drop_remainder = False
#sess=tf.Session()

#sess.run(tf.global_variables_initializer())


#default_g=tf.get_default_graph()


def fasta2record(data, output_file, vocab_file, step=1):
    # This function gets an input_file which is .fasta
    # This function returns the numbers of sequences in input_file
    # This function will check if the input_file is right
    #with open(input_file) as f:
        #lines = f.readlines()
    #print(lines)
    lines=['>s',data]
    for index, line in enumerate(lines):
        #print(line)
        if index % 2 == 0:
            if line[0] != ">":
                print("Row " + str(index + 1) + " is wrong!")
                exit()
        else:
            if line[0] == ">":
                print("Row " + str(index + 1) + " is wrong!")
                exit()
    seq_num = int(len(lines) / 2)

    processor = ColaProcessor()
    label_list = processor.get_labels()
    examples = processor.ljy_get_dev_examples(data)
    train_file = "predict.tf_record"
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=True)
    file_based_convert_examples_to_features(
        examples, label_list, 128, tokenizer, train_file)
    return seq_num

def cal_activity(data,sess):
    t1 = time.time()
    ss = ''
    for i in data:
        ss = ss + i
    samples_num = fasta2record(ss, "predict.tf_record", vocab_file, step=1)
    batch_num = math.ceil(samples_num / batch_size)

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        batch_size = params["batch_size"]
        d = tf.data.TFRecordDataset(input_file)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,))

        return d

    #sess.graph.finalize()

    predict_data = input_fn({"batch_size": batch_size})
    iterator = predict_data.make_one_shot_iterator().get_next()
    all_prob = []

    #sess.run(tf.global_variables_initializer())
    examples = sess.run(iterator)

    prob = \
        sess.run(probabilities,
                 feed_dict={input_ids: examples["input_ids"],
                            input_mask: examples["input_mask"],
                            segment_ids: examples["segment_ids"],
                            label_ids: examples["label_ids"]})
    all_prob.extend(prob[:, 1].tolist())
    #print(time.time() - t1)
    return 6-all_prob[0]'''




import bert_model.modeling
import bert_model.tokenization
from bert_model.run_classifier import create_model, file_based_input_fn_builder, ColaProcessor, \
    file_based_convert_examples_to_features
import tensorflow as tf
import bert_model.optimization
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef, roc_auc_score, accuracy_score, \
    confusion_matrix, roc_curve
import os
import time
import sys
from tensorflow.python.platform import gfile
import math


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
batch_size = 128
seq_length = 128
vocab_file = "./bert_model/vocab_1kmer.txt"
input_file = "predict.tf_record"
name_to_features = {
    "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
    "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
    "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
    "label_ids": tf.FixedLenFeature([], tf.int64),
    "is_real_example": tf.FixedLenFeature([], tf.int64),
}
drop_remainder = False
f = gfile.FastGFile('./bert_model/bert.pb', 'rb')

def fasta2record(data, vocab_file, step=1):
    lines=[]
    for i in data:
        lines.append('>s')
        lines.append(i)

    for index, line in enumerate(lines):
        if index % 2 == 0:
            if line[0] != ">":
                print("Row " + str(index + 1) + " is wrong!")
                exit()
        else:
            if line[0] == ">":
                print("Row " + str(index + 1) + " is wrong!")
                exit()
    seq_num = int(len(lines) / 2)


    processor = ColaProcessor()
    label_list = processor.get_labels()
    examples = processor.ljy_get_dev_examples(data)
    train_file = "predict.tf_record"
    tokenizer = bert_model.tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=True)
    file_based_convert_examples_to_features(
        examples, label_list, 128, tokenizer, train_file)
    return seq_num


def cal_activity(data):
    #t1=time.time()
    ss = []
    for i in data:
        s=''
        for j in i:
            s = s + j
        ss.append(s)
    samples_num = fasta2record(ss, vocab_file, step=1)
    batch_num = math.ceil(samples_num / batch_size)

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        batch_size = params["batch_size"]
        #
        d = tf.data.TFRecordDataset(input_file)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size, ))

        return d

    predict_data = input_fn({"batch_size": batch_size})
    iterator = predict_data.make_one_shot_iterator().get_next()

    all_prob=[]
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')  # 导入计算图
        input_0 = sess.graph.get_tensor_by_name('Placeholder:0')  # 此处的x一定要和之前保存时输入的名称一致！
        input_1 = sess.graph.get_tensor_by_name('Placeholder_1:0')
        input_2 = sess.graph.get_tensor_by_name('Placeholder_2:0')  # 此处的x一定要和之前保存时输入的名称一致！
        input_3 = sess.graph.get_tensor_by_name('Placeholder_3:0')
        logits = sess.graph.get_tensor_by_name('loss/BiasAdd:0')
        for _ in range(batch_num):
            examples = sess.run(iterator)

            prob = \
                sess.run(logits,
                         feed_dict={input_0: examples["input_ids"],
                                    input_1: examples["input_mask"],
                                    input_2: examples["segment_ids"],
                                    input_3: examples["label_ids"]})
            for i in range(len(prob)):
                prob[i][1] = 6 - prob[i][1]
                '''if prob[i][1] < 1:
                    prob[i][1] = math.log(prob[i][1], 1.5)
                else:
                    prob[i][1] = prob[i][1] - 1
                prob[i][1] = (prob[i][1] + 13) / 25'''
                prob[i][1]=prob[i][1]/12
                all_prob.append(prob[i][1])
    return all_prob
