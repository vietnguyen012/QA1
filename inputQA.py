from run_squad_sp import read_squad_examples,FeatureWriter,RawResult,write_predictions
import tokenization
from tf_record_create import convert_examples_to_features
def process_inputs(input_data):
    eval_examples = read_squad_examples(data=input_data,input_data=True)
    eval_features = []
    tokenizer=tokenization.FullTokenizer(vocab_file="/home/saplab/albert_vi/assets/albertvi_30k-clean.vocab",do_lower_case=True,spm_model_file="/home/saplab/albert_vi/assets/albertvi_30k-clean.model")
    eval_writer = FeatureWriter(
			filename="/home/saplab/albert_vi/Data/predict_file",
			is_training=False)
    def append_feature(feature):
        eval_features.append(feature)
        eval_writer.process_feature(feature)

    convert_examples_to_features(
			eval_examples,tokenizer=tokenizer,
            max_seq_length=384,
            doc_stride=128,
            max_query_length=64,
            is_training=False,
            output_fn=eval_writer.process_feature,
            do_lower_case=True)
    eval_writer.close()

    return eval_examples, eval_features
input_data={
 "options": {
  "n_best": True,
  "n_best_size": 3,
  "max_answer_length": 30
 },
  "data": [
  {
    "paragraphs": [
      {
        "qas": [
          {
            "question": "Virus corona có phải virus mới hay không?",
            "id": 27527,
            "answers": [
              {
                "answer_id": 25239,
                "document_id": 37254,
                "question_id": 27527,
                "text": "",
                "answer_start": 14,
              }
            ],
            "is_impossible": False
          }
        ],
        "context": "Vào ngày xxx .Vi-rút corona mới là loại vi-rút corona mới chưa từng được phát hiện trước đây. Vi-rút gây ra bệnh vi-rút corona 2019 (COVID-19), không cùng loại vi-rút corona thường lan truyền ở người và gây ra bệnh nhẹ, giống như cảm lạnh thông thường. xxx Vào ngày 11 tháng 2, 2020 Tổ Chức Y Tế Thế Giới đã tuyên bố tên chính thức cho căn bệnh đang gây bùng phát là vi-rút corona 2019 mới, lần đầu được xác định tại Vũ Hán, Trung Quốc. Tên mới cho căn bệnh này là bệnh vi-rút corona 2019, gọi tắt là COVID-19. Trong chữ COVID-19, 'CO' viết tắt của từ 'corona, VI' viết tắt của từ 'vi-rút,' và 'D' là bệnh. Trước đó, căn bệnh này được gọi là \\\"vi-rút corona mới 2019\\\" hoặc \\\"nCoV-2019\\\". Có nhiều loại vi-rút corona ở người bao gồm một số loại thường gây ra các chứng bệnh nhẹ ở đường hô hấp trên. COVID-19 là một bệnh mới, do một loại vi-rút corona mới chưa từng thấy ở người gây ra.",
        "document_id": 37254
      }
    ]
  }]
}
eval_examples, eval_features=process_inputs(input_data)

print(eval_features)
