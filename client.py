import grpc
import collections
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import tokenization
from tf_record_create import convert_examples_to_features,FeatureWriter
from run_squad_sp import write_predictions,read_squad_examples
import numpy as np
import pickle
#!/usr/bin/python
# -*- coding: utf-8 -*-

RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])
def process_inputs(input_data):
    eval_examples = read_squad_examples(data=input_data, input_data=True,
                                  is_training=False)
    eval_features = []
    tokenizer = \
        tokenization.FullTokenizer(vocab_file='/home/vietnguyen/albert_vi/assets/albertvi_30k-clean.vocab'
                                   , do_lower_case=True,
                                   spm_model_file='/home/vietnguyen/albert_vi/assets/albertvi_30k-clean.model'
                                   )
    eval_writer = \
        FeatureWriter(filename='/home/vietnguyen/albert_vi/Data/predict_file1.rtf',
                  is_training=False)

    def append_feature(feature):
        eval_features.append(feature)
        eval_writer.process_feature(feature)

    convert_examples_to_features(
        eval_examples,
        tokenizer=tokenizer,
        max_seq_length=384,
        doc_stride=328,
        max_query_length=64,
        is_training=False,
        output_fn=append_feature,
        do_lower_case=True,
        )
    eval_writer.close()

    with tf.gfile.Open("/home/vietnguyen/albert_vi/Data/predict_left_evalFeature.pkl", 'wb') as fout:
        pickle.dump(eval_features, fout)

    return (eval_examples, eval_features)


def process_result(result):
    unique_id = int(result['unique_ids'].int64_val[0])
    start_logits = [float(x) for x in result['start_logits'].float_val]
    end_logits = [float(x) for x in result['end_logits'].float_val]
    formatted_result=[]
    formatted_result.append(RawResult(unique_id=unique_id,
                                 start_logits=start_logits,
                                 end_logits=end_logits))

    return formatted_result


def process_output(
    all_results,
    eval_examples,
    eval_features,
    input_data,
    n_best,
    n_best_size,
    max_answer_length,
    ):

    (all_predictions, all_nbest_json) = write_predictions(
        eval_examples,
        eval_features,
        all_results,
        n_best_size=14,
        max_answer_length=40,
        do_lower_case=True,
        output_prediction_file='/home/vietnguyen/albert_vi/output/predictions.json'
            ,
        output_nbest_file='/home/vietnguyen/albert_vi/output/nbest_predictions.json'
            ,
        output_null_log_odds_file='/home/vietnguyen/albert_vi/output/null_odds.json'
            ,
        )
    re = []

        # for i in range(len(all_predictions)):

    for data in input_data['data']:
        for paragraph in data['paragraphs']:
            for qa in paragraph['qas']:
                    id=qa['id']
                    for i in range(0,len(all_nbest_json[str(id)])):
                        if n_best:
                            re.append(collections.OrderedDict({'question': qa['question'
                                      ], 'best_prediction':all_predictions[str(id)],
                                      'n_best_predictions':all_nbest_json[str(id)][i]['text'] }))
                        else:
                            re.append(collections.OrderedDict({'question': qa['question'
                                      ], 'best_prediction':all_predictions[str(id)] }))
    return re


input_data = {
    'options': {
        'n_best': True,
        'n_best_size': 3,
        'max_answer_length': 30},
    'data': [{
        'paragraphs': [{
            'qas': [{
                'question': 'Covid-19 là gì?',
                'id': 27527,
                'answers': [{
                    'answer_id': 25239,
                    'document_id': 37254,
                    'question_id': 27527,
                    'text': ""
                        ,
                    'answer_start': 14,
                }],
                'is_impossible': False,
            }],
        "context": "Virus Corona mới (Covid-19, SARS CoV-2) là một dạng mới của Coronavirus gây nhiễm trùng cấp tính với các triệu chứng hô hấp.\
         Virus này là một loại Coronavirus khác với loại gây ra SARS hoặc MERS. Nó cũng khác với loại Coronavirus gây nhiễm trùng theo mùa ở Hoa Kỳ.\
          Các ca đầu tiên của Coronavirus 2019-nCoV đã được phát hiện ở Vũ Hán,\
          Hồ Bắc, Trung Quốc. Kể từ đầu tháng 2 năm 2020, vi rút đã lan rộng bên trong Trung Quốc và lan đến một số quốc gia khác, bao gồm cả Hoa Kỳ.",
        "document_id": 37255
        }]
    }]
}

(eval_examples, eval_features) = process_inputs(input_data=input_data)
hostport = '127.0.0.1:8500'
channel = grpc.insecure_channel(hostport)
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
model_request = predict_pb2.PredictRequest()
model_request.model_spec.name = 'MyQA'
record_iterator = \
    tf.python_io.tf_record_iterator(path='/home/vietnguyen/albert_vi/Data/predict_file1.rtf'
                                    )
for string_record in record_iterator:
    model_request.inputs['examples'
                         ].CopyFrom(tf.contrib.util.make_tensor_proto(string_record,
                                    dtype=tf.string,
                                    shape=[1]))
result_future = stub.Predict.future(model_request, 30.0)
raw_result = result_future.result().outputs
formatted_result = process_result(raw_result)
re = process_output(
        formatted_result,
        eval_examples,
        eval_features,
        input_data,
        n_best=True,
        n_best_size=20,
        max_answer_length=60,
        )
print(re)
