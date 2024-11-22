#for pred_len in 96; do
#    python -u E:/MyCode/PyCharm_Code/TimeDART/run.py \
#        --task_name finetune \
#        --is_training 1 \
#        --root_path E:/MyCode/PyCharm_Code/TimeDART/datasets/ETT-small/ \
#        --data_path ETTh1.csv \
#        --model_id ETTh1 \
#        --model TimeDART \
#        --data ETTh1 \
#        --features M \
#        --input_len 336 \
#        --label_len 48 \
#        --pred_len $pred_len \
#        --e_layers 2 \
#        --enc_in 7 \
#        --dec_in 7 \
#        --c_out 7 \
#        --n_heads 16 \
#        --d_model 32 \
#        --d_ff 64 \
#        --patch_len 2 \
#        --stride 2 \
#        --dropout 0.2 \
#        --head_dropout 0.1 \
#        --batch_size 16 \
#        --lr_decay 0.5 \
#        --lradj step \
#        --time_steps 1000 \
#        --scheduler cosine \
#        --patience 3 \
#        --learning_rate 0.0001 \
#        --pct_start 0.3 \
#        --pretrain_checkpoints E:/MyCode/PyCharm_Code/TimeDART/outputs/pretrain_checkpoints/ \
#      > E:/MyCode/PyCharm_Code/TimeDART/.012.log 2>&1 \
#    done
#


    python -u E:/MyCode/PyCharm_Code/TimeDART/run.py \
        --task_name finetune \
        --is_training 1 \
        --root_path E:/MyCode/PyCharm_Code/TimeDART/datasets/ETT-small/ \
        --data_path ETTh1.csv \
        --model_id ETTh1 \
        --model TimeDART \
        --data ETTh1 \
        --features M \
        --input_len 336 \
        --label_len 48 \
        --pred_len 96 \
        --e_layers 2 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --n_heads 16 \
        --d_model 32 \
        --d_ff 64 \
        --patch_len 2 \
        --stride 2 \
        --dropout 0.2 \
        --head_dropout 0.1 \
        --batch_size 16 \
        --lr_decay 0.5 \
        --lradj step \
        --time_steps 1000 \
        --scheduler cosine \
        --patience 3 \
        --learning_rate 0.0001 \
        --pct_start 0.3 \
        --pretrain_checkpoints E:/MyCode/PyCharm_Code/TimeDART/outputs/pretrain_checkpoints/ \
      > E:/MyCode/PyCharm_Code/TimeDART/.012.log 2>&1