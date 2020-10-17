
from flask import Flask, request
import flask
import json
import hyper as hp

from client import *

import tensorflow as tf
app = Flask(__name__)


@app.route("/")
def _hello_world():
    return "Hello world"


@app.route("/botchat-api", methods=["POST","GET"])
def predict():
    data = {"status": False}
    if request.args.get('question'):
        data["status"] = True
    question = request.args.get('question')
    print("************************************************************")
    print("question: " + question)
    print("************************************************************")
    data["question"] = question
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
            "context": "Vi-rút corona mới là loại vi-rút corona mới chưa từng được phát hiện trước đây. Vi-rút gây ra bệnh vi-rút corona 2019 (COVID-19), không cùng loại vi-rút corona thường lan truyền ở người và gây ra bệnh nhẹ, giống như cảm lạnh thông thường. Vào ngày 11 tháng 2, 2020 Tổ Chức Y Tế Thế Giới đã tuyên bố tên chính thức cho căn bệnh đang gây bùng phát là vi-rút corona 2019 mới, lần đầu được xác định tại Vũ Hán, Trung Quốc. Tên mới cho căn bệnh này là bệnh vi-rút corona 2019, gọi tắt là COVID-19. Trong chữ COVID-19, 'CO' viết tắt của từ 'corona, VI' viết tắt của từ 'vi-rút,' và 'D' là bệnh. Trước đó, căn bệnh này được gọi là \\\"vi-rút corona mới 2019\\\" hoặc \\\"nCoV-2019\\\". Có nhiều loại vi-rút corona ở người bao gồm một số loại thường gây ra các chứng bệnh nhẹ ở đường hô hấp trên. COVID-19 là một bệnh mới, do một loại vi-rút corona mới chưa từng thấy ở người gây ra.",
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
    data["answer"] = re
    return json.dumps(data, ensure_ascii=False)


if __name__ == "__main__":
    print("App run!")
    app.run(debug=False, host=hp.IP, threaded=False)

if __name__ == "__main__":
    main()
