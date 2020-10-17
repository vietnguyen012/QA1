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
        tokenization.FullTokenizer(vocab_file='/home/saplab/albert_vi/assets/albertvi_30k-clean.vocab'
                                   , do_lower_case=True,
                                   spm_model_file='/home/saplab/albert_vi/assets/albertvi_30k-clean.model'
                                   )
    eval_writer = \
        FeatureWriter(filename='/home/saplab/albert_vi/Data/predict_file1.rtf',
                  is_training=False)

    def append_feature(feature):
        eval_features.append(feature)
        eval_writer.process_feature(feature)

    convert_examples_to_features(
        eval_examples,
        tokenizer=tokenizer,
        max_seq_length=512,
        doc_stride=128,
        max_query_length=64,
        is_training=False,
        output_fn=append_feature,
        do_lower_case=True,
        )
    eval_writer.close()

    with tf.gfile.Open("/home/saplab/albert_vi/Data/predict_left_evalFeature.pkl", 'wb') as fout:
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
        max_answer_length=30,
        do_lower_case=True,
        output_prediction_file='/home/saplab/albert_vi/output/predictions.json'
            ,
        output_nbest_file='/home/saplab/albert_vi/output/nbest_predictions.json'
            ,
        output_null_log_odds_file='/home/saplab/albert_vi/output/null_odds.json'
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


input_data = {'options': {'n_best': True, 'n_best_size': 3,
              'max_answer_length': 30},
              'data': [{'paragraphs': [{'qas': [{
    'question': 'Tác hại của covid-19?'
        ,
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
        "context": "Vi-rút corona mới là loại vi-rút corona mới chưa từng được phát hiện trước đây. Vi-rút gây ra bệnh vi-rút corona 2019 (COVID-19), không cùng loại vi-rút corona thường lan truyền ở người và gây ra bệnh nhẹ, giống như cảm lạnh thông thường. Vào ngày 11 tháng 2, 2020 Tổ Chức Y Tế Thế Giới đã tuyên bố tên chính thức cho căn bệnh đang gây bùng phát là vi-rút corona 2019 mới, lần đầu được xác định tại Vũ Hán, Trung Quốc. Tên mới cho căn bệnh này là bệnh vi-rút corona 2019, gọi tắt là COVID-19. Trong chữ COVID-19, 'CO' viết tắt của từ 'corona, VI' viết tắt của từ 'vi-rút,' và 'D' là bệnh. Trước đó, căn bệnh này được gọi là \\\"vi-rút corona mới 2019\\\" hoặc \\\"nCoV-2019\\\". Có nhiều loại vi-rút corona ở người bao gồm một số loại thường gây ra các chứng bệnh nhẹ ở đường hô hấp trên. COVID-19 là một bệnh mới, do một loại vi-rút corona mới chưa từng thấy ở người gây ra.",
          "document_id": 37255}]}]}

(eval_examples, eval_features) = process_inputs(input_data=input_data)
hostport = '127.0.0.1:8500'
channel = grpc.insecure_channel(hostport)
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
model_request = predict_pb2.PredictRequest()
model_request.model_spec.name = 'MyQA'
record_iterator = \
    tf.python_io.tf_record_iterator(path='/home/saplab/albert_vi/Data/predict_file1.rtf'
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
        max_answer_length=30,
        )
print(re)
