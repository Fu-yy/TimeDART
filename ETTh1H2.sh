#python -u run.py \
#    --task_name pretrain \
#    --root_path ./datasets/ETT-small/ \
#    --data_path ETTh1.csv \
#    --model_id ETTh1 \
#    --model TimeDART_my \
#    --data ETTh1 \
#    --features M \
#    --input_len 336 \
#    --e_layers 2 \
#    --d_layers 1 \
#    --enc_in 7 \
#    --dec_in 7 \
#    --c_out 7 \
#    --n_heads 16 \
#    --d_model 32 \
#    --d_ff 64 \
#    --patch_len 2 \
#    --stride 2 \
#    --head_dropout 0.1 \
#    --dropout 0.2 \
#    --time_steps 1000 \
#    --scheduler cosine \
#    --lr_decay 0.9 \
#    --learning_rate 0.0001 \
#    --batch_size 16 \
#    --train_epochs 50 \
#> 'TimeDART_my_ETTh1_pretrain'0.01.log 2>&1




for pred_len in 96 192 336 720; do
    python -u run.py \
        --task_name finetune \
        --is_training 1 \
        --root_path ./datasets/ETT-small/ \
        --data_path ETTh1.csv \
        --model_id ETTh1 \
        --model TimeDART_my \
        --data ETTh1 \
        --features M \
        --input_len 336 \
        --label_len 48 \
        --pred_len $pred_len \
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
> 'TimeDART_my_ETTh1_finetune'$pred_len_0.01.log 2>&1

done






#python -u run.py \
#    --task_name pretrain \
#    --root_path ./datasets/ETT-small/ \
#    --data_path ETTh2.csv \
#    --model_id ETTh2 \
#    --model TimeDART_my \
#    --data ETTh2 \
#    --features M \
#    --input_len 336 \
#    --e_layers 2 \
#    --d_layers 1 \
#    --enc_in 7 \
#    --dec_in 7 \
#    --c_out 7 \
#    --n_heads 8 \
#    --d_model 8 \
#    --d_ff 32 \
#    --patch_len 2 \
#    --stride 2 \
#    --head_dropout 0.1 \
#    --dropout 0.2 \
#    --time_steps 1000 \
#    --scheduler cosine \
#    --lr_decay 0.95 \
#    --learning_rate 0.0005 \
#    --batch_size 16 \
#    --train_epochs 50 \
#> 'TimeDART_my_ETTh2_pretrain'0.01.log 2>&1




for pred_len in 96 192 336 720; do
    python -u run.py \
        --task_name finetune \
        --is_training 1 \
        --root_path ./datasets/ETT-small/ \
        --data_path ETTh2.csv \
        --model_id ETTh2 \
        --model TimeDART_my \
        --data ETTh2 \
        --features M \
        --input_len 336 \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 2 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --n_heads 8 \
        --d_model 8 \
        --d_ff 32 \
        --patch_len 2 \
        --stride 2 \
        --dropout 0.4 \
        --head_dropout 0.1 \
        --batch_size 16 \
        --lr_decay 0.5 \
        --lradj step \
        --time_steps 1000 \
        --scheduler cosine \
        --patience 3 \
        --learning_rate 0.0001 \
        --pct_start 0.3 \
> 'TimeDART_my_ETTh2_finetune'$pred_len_0.01.log 2>&1

done




python -u run.py \
    --task_name pretrain \
    --root_path ./datasets/ETT-small/ \
    --data_path ETTm1.csv \
    --model_id ETTm1 \
    --model TimeDART \
    --data ETTm1 \
    --features M \
    --input_len 336 \
    --e_layers 2 \
    --d_layers 1 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --n_heads 8 \
    --d_model 32 \
    --d_ff 64 \
    --patch_len 2 \
    --stride 2 \
    --head_dropout 0.1 \
    --dropout 0.2 \
    --time_steps 1000 \
    --scheduler cosine \
    --lr_decay 0.95 \
    --learning_rate 0.0001 \
    --batch_size 64 \
    --train_epochs 50 \
> 'TimeDART_ETTm1_pretrain'0.01.log 2>&1



for pred_len in 96 192 336 720; do
    python -u run.py \
        --task_name finetune \
        --is_training 1 \
        --root_path ./datasets/ETT-small/ \
        --data_path ETTm1.csv \
        --model_id ETTm1 \
        --model TimeDART \
        --data ETTm1 \
        --features M \
        --input_len 336 \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 2 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --n_heads 8 \
        --d_model 32 \
        --d_ff 64 \
        --patch_len 2 \
        --stride 2 \
        --dropout 0.2 \
        --head_dropout 0.0 \
        --batch_size 64 \
        --lr_decay 0.5 \
        --lradj decay \
        --time_steps 1000 \
        --scheduler cosine \
        --patience 3 \
        --learning_rate 0.0001 \
        --pct_start 0.3 \
> 'TimeDART_ETTm1_finetune'$pred_len_0.01.log 2>&1

done



python -u run.py \
    --task_name pretrain \
    --root_path ./datasets/ETT-small/ \
    --data_path ETTm2.csv \
    --model_id ETTm2 \
    --model TimeDART \
    --data ETTm2 \
    --features M \
    --input_len 336 \
    --e_layers 2 \
    --d_layers 1 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --n_heads 8 \
    --d_model 8 \
    --d_ff 16 \
    --patch_len 2 \
    --stride 2 \
    --head_dropout 0.1 \
    --dropout 0.2 \
    --time_steps 1000 \
    --scheduler cosine \
    --lr_decay 0.8 \
    --learning_rate 0.001 \
    --batch_size 64 \
    --train_epochs 50 \
> 'TimeDART_ETTm2_pretrain'0.01.log 2>&1



for pred_len in 96 192 336 720; do
    python -u run.py \
        --task_name finetune \
        --is_training 1 \
        --root_path ./datasets/ETT-small/ \
        --data_path ETTm2.csv \
        --model_id ETTm2 \
        --model TimeDART \
        --data ETTm2 \
        --features M \
        --input_len 336 \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 2 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --n_heads 8 \
        --d_model 8 \
        --d_ff 16 \
        --patch_len 2 \
        --stride 2 \
        --dropout 0.4 \
        --head_dropout 0.1 \
        --batch_size 64 \
        --lr_decay 0.5 \
        --lradj step \
        --time_steps 1000 \
        --scheduler cosine \
        --patience 3 \
        --learning_rate 0.0001 \
        --pct_start 0.2 \
> 'TimeDART_ETTm2_finetune'$pred_len_0.01.log 2>&1

done
