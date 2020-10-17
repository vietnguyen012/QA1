python3.6 create_pretraining_data.py \
  --input_file=/home/saplab/albert_vi/Dataset/train.csv \
  --dupe_factor=10 \
  --vocab_file=/home/saplab/albert_vi/assets/albertvi_30k-clean.vocab \
  --spm_model_file=/home/saplab/albert_vi/assets/albertvi_30k-clean.model \
  --output_file=/home/saplab/albert_vi/Dataset/train_feature_file.tf \
  --vocab_file assets/albertvi_30k-clean.vocab \
  --spm_model_file assets/albertvi_30k-clean.model
