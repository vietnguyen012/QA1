python3.6 run_squad_sp.py \
--vocab_file=/home/saplab/albert_vi/assets/albertvi_30k-clean.vocab \
--albert_config_file=/home/saplab/albert_vi/config/base/albert_config.json \
--init_checkpoint=/home/saplab/albert_vi/output/model.ckpt-120 \
--spm_model_file=/home/saplab/albert_vi/assets/albertvi_30k-clean.model \
--do_predict=True \
--predict_feature_left_file=/home/saplab/albert_vi/Data/predict_left_file.pkl \
--interact=True \
--context=/home/saplab/albert_vi/output/data.txt \
--output_dir=/home/saplab/albert_vi/Data
