
from flask import Flask, request
import flask
import json
import hyper as hp
from client2 import *
import sys
sys.path.insert(1, '/home/vietnguyen/albert_vi/botchat-api/search-engine')
from  test import top_k_contexts
import tensorflow as tf
app = Flask(__name__)

def modify_input_data(question, context):
  input_data = {
    'options': {
        'n_best': True, 
        'n_best_size': 3,
        'max_answer_length': 30},
    'data': [{
        'paragraphs': [{
            'qas': [{
                'question': question,
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
        "context": context,
        "document_id": 37255
        }]
    }]
  }
  return input_data

def gen_output_qa(status, question, answers):
    re={
        'status': status,
        'question': question,
        'answers': answers
    }
    return re



@app.route("/")
def _hello_world():
    return "Hello world"


@app.route("/botchat-api", methods=["POST","GET"])
def predict():
    status = False
    if request.args.get('question'):
        status = True
    question = request.args.get('question')
    question=question+"?"
    k = 4
    try:
        contexts = top_k_contexts(question, k)
    except:
        res = []
        return res.append({
            'status': True,
            'question': question,
            'answers': "empty"
            })

    # contexts = ['1','2','3','4']
    # for context in contexts:
    #     print(context)

    # context = "Sự nổi trội của mối liên hệ giữa tác nhân gây bệnh ở người và dơi đã dẫn đến cuộc tranh luận về việc liệu loài dơi có đóng góp một cách tương xứng vào việc lây nhiễm virus mới vượt qua hàng rào loài vào người hay không"
    # pool = Pool(4)
    # eval_examples, eval_features = pool.map(process_inputs, (idata) for idata in input_data_lst)

    # inputs = []
    res = []
    resultContext = []

    k = len(contexts)
    
    for i in range(k):
        try:
            print("contexts",contexts[i])
            if "copyright 2019 vinmec. all rights reserved" in contexts[i] or "코로나바이러스감염증-19 국내 발생 현황일일집계통계, 9시 기준 corona virus infection-19 domestic occurrence daily statistics, 900 bằng tiếng hàn. ngày 24 tháng 2 năm 2020" in contexts[i]:
                continue
            input_data = modify_input_data(question, contexts[i])
        except:
            res = []
            return  res.append({
            'status': True,
            'question': question,
            'answers': "empty"
            })
        print("-"*30)
        print("context:",contexts[i])
        print("-"*30)
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
        

        _,combinedText= process_output(
                formatted_result,
                eval_examples,
                eval_features,
                input_data,
                n_best=True,
                n_best_size=20,
                max_answer_length=30,
                context = contexts[i],
                combined=True
                )

        combinedText=combinedText.strip()
        resultContext.append(combinedText)

    resultCombined = '.'.join(resultContext)
    resultCombined=re.sub(' +', ' ', resultCombined)
    resultCombined=re.sub('[+]','',resultCombined)
    input_data_last = modify_input_data(question,resultCombined)

    (last_eval_examples,last_eval_features) = process_inputs(input_data = input_data_last)
    # hostport = '127.0.0.1:8500'
    # channel = grpc.insecure_channel(hostport)
    # stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    # model_request = predict_pb2.PredictRequest()
    # model_request.model_spec.name = 'MyQA'
    record_iterator = \
    tf.python_io.tf_record_iterator(path='/home/vietnguyen/albert_vi/Data/predict_file1.rtf'
                                            )
    for string_record in record_iterator:
        model_request.inputs['examples'
                        ].CopyFrom(tf.contrib.util.make_tensor_proto(string_record,
                                                dtype=tf.string,
                                                shape=[1]))
    last_result_future = stub.Predict.future(model_request, 30.0)
    last_raw_result = last_result_future.result().outputs
    last_formatted_result = process_result(last_raw_result)
    print("-"*40)
    print("resultCombined:",resultCombined)
    print("-"*40)
    _,last_combinedText= process_output(
                    last_formatted_result,
                    last_eval_examples,
                    last_eval_features,
                    input_data_last,
                    n_best=True,
                    n_best_size=20,
                    max_answer_length=30,
                    context = resultCombined,
                    combined=True
                    )

    res.append(gen_output_qa(status, question, last_combinedText))
        
    return json.dumps(res, ensure_ascii=False)


if __name__ == "__main__":
    print("App run!")
    app.run(debug=False, host=hp.IP, port=5001, threaded=False)

if __name__ == "__main__":
    main()
