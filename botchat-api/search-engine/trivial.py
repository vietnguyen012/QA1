import postprocess

text = ['Bệnh Cúm (Cúm) và COVID-19 đều là các bệnh hô hấp truyền nhiễm nhưng do các loại vi-rút khác nhau gây ra. COVID-19 là do nhiễm vi-rút Corona chủng mới (gọi là SARS-CoV-2) còn cúm là do nhiễm các vi-rút cúm. Bởi vì cúm và COVID-19 có một số các triệu chứng tương tự nhau nên có thể khó để phân biệt hai loại bệnh này chỉ dựa trên triệu chứng, do đó có thể cần thực hiện xét nghiệm để giúp xác nhận chẩn đoán bệnh. Cúm và COVID-19 có thể có nhiều đặc điểm chung nhưng giữa hai loại bệnh này vẫn có một số sự khác biệt quan trọng.']

# print(postprocess.remove_non_word_char(text))
print(postprocess.post_process(text))
