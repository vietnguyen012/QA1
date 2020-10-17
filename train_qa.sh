export CUDA_VISIBLE_DEVICES=${0,1}
N_GPUS=$[$(echo ${CUDA_VISIBLE_DEVICES} | grep -o ',' | wc -l)+1]

python3.6 run_qa.py \
--task_name='zqa' \
--do_train=True \
--use_tpu=False \
--do_eval=False \
--data_dir=/home/saplab/QA/zaloqa2019/Dataset \
--vocab_file=/home/saplab/QA/zaloqa2019/albert_vi/assets/albertvi_30k-clean.vocab \
--albert_config_file=/home/saplab/QA/zaloqa2019/albert_vi/config/base/albert_config.json \
--init_checkpoint=/home/saplab/QA/zaloqa2019/albert_vi/checkpoint/model.ckpt-1015000 \
--max_seq_length=512 \
--train_batch_size=16 \
--predict_batch_size=8 \
--learning_rate=2e-5 \
--num_train_epochs=3.0 \
--output_dir=/home/saplab/QA/zaloqa2019/output \
--do_predict=True \
--swap_input=False \
--do_lower_case=False \
--use_class_weights=True