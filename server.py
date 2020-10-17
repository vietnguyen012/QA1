
from flask import Flask, request
import flask
import json
import hyper as hp

from client2 import *

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
            "context": "Các triệu chứng thường gặp nhất:sốt ho khan mệt mỏi Các \
             triệu chứng ít gặp hơn:đau nhức đau họng tiêu chảy viêm kết mạc đau đầu mất vị giác hoặc khứu giác da nổi mẩn hay ngón tay hoặc \
             ngón chân bị tấy đỏ hoặc tím tái Các triệu chứng nghiêm trọng:khó thở đau hoặc tức ngực mất khả năng nói hoặc cử động.",
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
    app.run(debug=False, host=hp.IP, port=5001, threaded=False)

if __name__ == "__main__":
    main()
