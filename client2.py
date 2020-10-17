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
import re
#!/usr/bin/python
# -*- coding: utf-8 -*-
# from vncorenlp import VnCoreNLP
# rdrsegmenter = VnCoreNLP(address="http://127.0.0.1", port=9050)


def post_process(context,answerSentence,combined=False):
    if combined == False:
      return answerSentence
    try:
      print(answerSentence)
      answerSentence = answerSentence.split(".")
      answerSentence = max(answerSentence, key=len)
      answerSentence = [answerSentence]
      print("max answer:",answerSentence)
      answer = []
      contextList = context.split(".")
      print("contextList:",contextList)
      for text in contextList:
        for subanswer in answerSentence:
          print("subanswer:",subanswer)
          if subanswer in text:
            if answer == []:
              answer.append(text)
              break
            else:
              continue
      return '.'.join(answer)
    except:
      print("*")
      return answerSentence


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
        doc_stride=128,
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
    context,
    combined =False
    ):
    try:
      (all_predictions, all_nbest_json) = write_predictions(
          eval_examples,
          eval_features,
          all_results,
          n_best_size=5,
          max_answer_length=100,
          do_lower_case=True,
          output_prediction_file='/home/vietnguyen/albert_vi/output/predictions.json'
              ,
          output_nbest_file='/home/vietnguyen/albert_vi/output/nbest_predictions.json'
              ,
          output_null_log_odds_file='/home/vietnguyen/albert_vi/output/null_odds.json'
              ,
          )
    except:
      return "empty","empty"
    re = []

        # for i in range(len(all_predictions)):
           
    # print('-'*30)
    # print('context: ', context)
    # print('-'*30)
    combination = []

    for data in input_data['data']:
        for paragraph in data['paragraphs']:
            for qa in paragraph['qas']:
                    id=qa['id']
                    all_pred = all_predictions[str(id)]
                            # print("--------------------------------------------")
                            # print("n_best_pred:",n_best_pred)
                    official_best_pred = post_process(context,all_pred,combined)
                            
                    if official_best_pred!="empty":
                            
                        if(official_best_pred not in combination):
                                  combination.append(official_best_pred)
                              # print("official_best_pred",official_N_best)
                              # print("probability answer:",probability)
                              # print("--------------------------------------------")
                        re.append(collections.OrderedDict({'question': qa['question'
                                      ], 'best_prediction':official_best_pred,
        }))
                    else:
                            all_pred = all_predictions[str(id)]
                            official_best_pred = post_process(context,all_pred,combined)

                            re.append(collections.OrderedDict({'question': qa['question'
                                      ], 'best_prediction':official_best_pred }))
    combinedText = '.'.join(combination)
    return re,combinedText

if __name__ == '__main__':


    input_data = {'options': {'n_best': True, 'n_best_size': 3,
                  'max_answer_length': 30},
                  'data': [{'paragraphs': [{'qas': [{
        'question': 'Vào ngày 30 tháng 3, Bộ trưởng Ngoại giao Vương quốc Anh đã đưa ra tuyên bố gì?'
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
            "context": "Một khả năng khác, bỏ qua lo ngại về sự lây nhiễm, là nước này không muốn công dân của họ bị mắc kẹt ở nước khác nếu các chuyến bay bị đình chỉ vào một ngày sau đó. Ví dụ, vào thứ Hai, ngày 30 tháng 3, Bộ trưởng Ngoại giao Vương quốc Anh Dominic Raab đã công bố kế hoạch 75 triệu bảng thuê các chuyến bay để hồi hương những người Anh mắc kẹt, đồng thời khuyến cáo bất kỳ công dân nào còn ở các quốc gia vẫn có các chuyến bay thương mại để bay về nước càng sớm càng tốt...Nếu công dân của một quốc gia bị mắc kẹt ở nước ngoài, chính phủ sẽ có trách nhiệm đưa họ về nước.",
              "document_id": 37255}]}]}

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
    context = "Một khả năng khác, bỏ qua lo ngại về sự lây nhiễm, là nước này không muốn công dân của họ bị mắc kẹt ở nước khác nếu các chuyến bay bị đình chỉ vào một ngày sau đó. Ví dụ, vào thứ Hai, ngày 30 tháng 3, Bộ trưởng Ngoại giao Vương quốc Anh Dominic Raab đã công bố kế hoạch 75 triệu bảng thuê các chuyến bay để hồi hương những người Anh mắc kẹt, đồng thời khuyến cáo bất kỳ công dân nào còn ở các quốc gia vẫn có các chuyến bay thương mại để bay về nước càng sớm càng tốt...Nếu công dân của một quốc gia bị mắc kẹt ở nước ngoài, chính phủ sẽ có trách nhiệm đưa họ về nước."
    re,combinedText = process_output(
            formatted_result,
            eval_examples,
            eval_features,
            input_data,
            n_best=True,
            n_best_size=100,
            max_answer_length=30,
            context = context
            )
    print(re)
