if [ ! -d "./run_log" ]; then
    mkdir ./run_log
fi
if [ ! -d "./run_log/log_202411241520_win" ]; then
    mkdir ./run_log/log_202411241520_win
fi
if [ ! -d "./run_log/log_202411241520_win/ETTm1" ]; then
    mkdir ./run_log/log_202411241520_win/ETTm1
fi
if [ ! -d "./run_log/log_202411241520_win/ETTh1" ]; then
    mkdir ./run_log/log_202411241520_win/ETTh1
fi
if [ ! -d "./run_log/log_202411241520_win/ETTm2" ]; then
    mkdir ./run_log/log_202411241520_win/ETTm2
fi

if [ ! -d "./run_log/log_202411241520_win/ETTh2" ]; then
    mkdir ./run_log/log_202411241520_win/ETTh2
fi
if [ ! -d "./run_log/log_202411241520_win/electricity" ]; then
    mkdir ./run_log/log_202411241520_win/electricity
fi

if [ ! -d "./run_log/log_202411241520_win/Exchange" ]; then
    mkdir ./run_log/log_202411241520_win/Exchange
fi

if [ ! -d "./run_log/log_202411241520_win/Solar" ]; then
    mkdir ./run_log/log_202411241520_win/Solar
fi

if [ ! -d "./run_log/log_202411241520_win/weather" ]; then
    mkdir ./run_log/log_202411241520_win/weather
fi

if [ ! -d "./run_log/log_202411241520_win/Traffic" ]; then
    mkdir ./run_log/log_202411241520_win/Traffic
fi

if [ ! -d "./run_log/log_202411241520_win/PEMS03" ]; then
    mkdir ./run_log/log_202411241520_win/PEMS03
fi

if [ ! -d "./run_log/log_202411241520_win/PEMS04" ]; then
    mkdir ./run_log/log_202411241520_win/PEMS04
fi

if [ ! -d "./run_log/log_202411241520_win/PEMS07" ]; then
    mkdir ./run_log/log_202411241520_win/PEMS07
fi
if [ ! -d "./run_log/log_202411241520_win/PEMS08" ]; then
    mkdir ./run_log/log_202411241520_win/PEMS08
fi



python -u run.py \
    --task_name pretrain \
    --root_path ./datasets/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1 \
    --model TimeDART \
    --data ETTh1 \
    --features M \
    --input_len 336 \
    --e_layers 2 \
    --d_layers 1 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --n_heads 16 \
    --d_model 32 \
    --d_ff 64 \
    --patch_len 2 \
    --stride 2 \
    --head_dropout 0.1 \
    --dropout 0.2 \
    --time_steps 1000 \
    --scheduler cosine \
    --lr_decay 0.9 \
    --learning_rate 0.0001 \
    --batch_size 16 \
    --train_epochs 50 \
    --down_sampling_layer 2 \
    --down_sampling_window 2 \
> ./run_log/log_202411241520_win/ETTh1/'TimeDART_pretrain'0.01.log 2>&1

for pred_len in 96; do
    python -u run.py \
        --task_name finetune \
        --is_training 1 \
        --root_path ./datasets/ETT-small/ \
        --data_path ETTh1.csv \
        --model_id ETTh1 \
        --model TimeDART \
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
        --down_sampling_layer 2 \
        --down_sampling_window 2 \
> ./run_log/log_202411241520_win/ETTh1/'TimeDART_finetune'$pred_len_0.01.log 2>&1
done
#
# ETTh2
python -u run.py \
     --task_name pretrain \
    --root_path ./datasets/ETT-small/ \
    --data_path ETTh2.csv \
    --model_id ETTh2 \
    --model TimeDART \
    --data ETTh2 \
    --features M \
    --input_len 336 \
    --e_layers 2 \
    --d_layers 1 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --n_heads 8 \
    --d_model 8 \
    --d_ff 32 \
    --patch_len 2 \
    --stride 2 \
    --head_dropout 0.1 \
    --dropout 0.2 \
    --time_steps 1000 \
    --scheduler cosine \
    --lr_decay 0.95 \
    --learning_rate 0.0005 \
    --batch_size 16 \
    --train_epochs 50 \
    --down_sampling_layer 2 \
    --down_sampling_window 2 \
> ./run_log/log_202411241520_win/ETTh2/'TimeDART_pretrain'0.01.log 2>&1



for pred_len in 96; do
    python -u run.py \
        --task_name finetune \
        --is_training 1 \
        --root_path ./datasets/ETT-small/ \
        --data_path ETTh2.csv \
        --model_id ETTh2 \
        --model TimeDART \
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
        --down_sampling_layer 2 \
        --down_sampling_window 2 \
> ./run_log/log_202411241520_win/ETTh2/'TimeDART_finetune'$pred_len_0.01.log 2>&1

done




# ETTm1
#python -u run.py \
#    --task_name pretrain \
#    --root_path ./datasets/ETT-small/ \
#    --data_path ETTm1.csv \
#    --model_id ETTm1 \
#    --model TimeDART \
#    --data ETTm1 \
#    --features M \
#    --input_len 336 \
#    --e_layers 2 \
#    --d_layers 1 \
#    --enc_in 7 \
#    --dec_in 7 \
#    --c_out 7 \
#    --n_heads 8 \
#    --d_model 32 \
#    --d_ff 64 \
#    --patch_len 2 \
#    --stride 2 \
#    --head_dropout 0.1 \
#    --dropout 0.2 \
#    --time_steps 1000 \
#    --scheduler cosine \
#    --lr_decay 0.95 \
#    --learning_rate 0.0001 \
#    --batch_size 64 \
#    --train_epochs 50 \
#    --down_sampling_layer 2 \
#    --down_sampling_window 2 \
#> ./run_log/log_202411241520_win/ETTm1/'TimeDART_pretrain'0.01.log 2>&1
#
#
#
#for pred_len in 96; do
#    python -u run.py \
#        --task_name finetune \
#        --is_training 1 \
#        --root_path ./datasets/ETT-small/ \
#        --data_path ETTm1.csv \
#        --model_id ETTm1 \
#        --model TimeDART \
#        --data ETTm1 \
#        --features M \
#        --input_len 336 \
#        --label_len 48 \
#        --pred_len $pred_len \
#        --e_layers 2 \
#        --enc_in 7 \
#        --dec_in 7 \
#        --c_out 7 \
#        --n_heads 8 \
#        --d_model 32 \
#        --d_ff 64 \
#        --patch_len 2 \
#        --stride 2 \
#        --dropout 0.2 \
#        --head_dropout 0.0 \
#        --batch_size 64 \
#        --lr_decay 0.5 \
#        --lradj decay \
#        --time_steps 1000 \
#        --scheduler cosine \
#        --patience 3 \
#        --learning_rate 0.0001 \
#        --pct_start 0.3 \
#       --down_sampling_layer 2 \
#       --down_sampling_window 2 \
#> ./run_log/log_202411241520_win/ETTm1/'TimeDART_finetune'$pred_len_0.01.log 2>&1
#
#done
#
#
#
#
#
#
##ETTm2
#python -u run.py \
#    --task_name pretrain \
#    --root_path ./datasets/ETT-small/ \
#    --data_path ETTm2.csv \
#    --model_id ETTm2 \
#    --model TimeDART \
#    --data ETTm2 \
#    --features M \
#    --input_len 336 \
#    --e_layers 2 \
#    --d_layers 1 \
#    --enc_in 7 \
#    --dec_in 7 \
#    --c_out 7 \
#    --n_heads 8 \
#    --d_model 8 \
#    --d_ff 16 \
#    --patch_len 2 \
#    --stride 2 \
#    --head_dropout 0.1 \
#    --dropout 0.2 \
#    --time_steps 1000 \
#    --scheduler cosine \
#    --lr_decay 0.8 \
#    --learning_rate 0.001 \
#    --batch_size 64 \
#    --train_epochs 50 \
#    --down_sampling_layer 2 \
#    --down_sampling_window 2 \
#> ./run_log/log_202411241520_win/ETTm2/'TimeDART_pretrain'0.01.log 2>&1
#
#
#
#for pred_len in 96; do
#
#    python -u run.py \
#        --task_name finetune \
#        --is_training 1 \
#        --root_path ./datasets/ETT-small/ \
#        --data_path ETTm2.csv \
#        --model_id ETTm2 \
#        --model TimeDART \
#        --data ETTm2 \
#        --features M \
#        --input_len 336 \
#        --label_len 48 \
#        --pred_len $pred_len \
#        --e_layers 2 \
#        --enc_in 7 \
#        --dec_in 7 \
#        --c_out 7 \
#        --n_heads 8 \
#        --d_model 8 \
#        --d_ff 16 \
#        --patch_len 2 \
#        --stride 2 \
#        --dropout 0.4 \
#        --head_dropout 0.1 \
#        --batch_size 64 \
#        --lr_decay 0.5 \
#        --lradj step \
#        --time_steps 1000 \
#        --scheduler cosine \
#        --patience 3 \
#        --learning_rate 0.0001 \
#        --pct_start 0.2 \
#       --down_sampling_layer 2 \
#       --down_sampling_window 2 \
#> ./run_log/log_202411241520_win/ETTm2/'TimeDART_finetune'$pred_len_0.01.log 2>&1
#
#done

#
#
##Exchange
#python -u run.py \
#    --task_name pretrain \
#    --root_path ./datasets/exchange-rate/ \
#    --data_path exchange.csv \
#    --model_id Exchange \
#    --model TimeDART \
#    --data Exchange \
#    --features M \
#    --input_len 336 \
#    --e_layers 2 \
#    --d_layers 1 \
#    --enc_in 8 \
#    --dec_in 8 \
#    --c_out 8 \
#    --n_heads 8 \
#    --d_model 32 \
#    --d_ff 64 \
#    --patch_len 2 \
#    --stride 2 \
#    --head_dropout 0.1 \
#    --dropout 0.2 \
#    --time_steps 1000 \
#    --scheduler cosine \
#    --lr_decay 0.95 \
#    --learning_rate 0.001 \
#    --batch_size 16 \
#    --train_epochs 50 \
#    --down_sampling_layer 2 \
#    --down_sampling_window 2 \
#> ./run_log/log_202411241520_win/Exchange/'TimeDART_pretrain'0.01.log 2>&1
#
#
#
#for pred_len in 96; do
#    python -u run.py \
#        --task_name finetune \
#        --is_training 1 \
#        --root_path ./datasets/exchange-rate/ \
#        --data_path exchange.csv \
#        --model_id Exchange \
#        --model TimeDART \
#        --data Exchange \
#        --features M \
#        --input_len 336 \
#        --label_len 48 \
#        --pred_len $pred_len \
#        --e_layers 2 \
#        --enc_in 8 \
#        --dec_in 8 \
#        --c_out 8 \
#        --n_heads 8 \
#        --d_model 32 \
#        --d_ff 64 \
#        --patch_len 2 \
#        --stride 2 \
#        --dropout 0.2 \
#        --head_dropout 0.1 \
#        --batch_size 16 \
#        --lr_decay 0.8 \
#        --lradj decay \
#        --time_steps 1000 \
#        --scheduler cosine \
#        --patience 3 \
#        --learning_rate 0.0001 \
#        --pct_start 0.3 \
#       --down_sampling_layer 2 \
#       --down_sampling_window 2 \
#> ./run_log/log_202411241520_win/Exchange/'TimeDART_finetune'$pred_len_0.01.log 2>&1
#
#done
#
##WTH
#
#python -u run.py \
#    --task_name pretrain \
#    --root_path ./datasets/weather/ \
#    --data_path weather.csv \
#    --model_id Weather \
#    --model TimeDART \
#    --data Weather \
#    --features M \
#    --input_len 336 \
#    --e_layers 2 \
#    --d_layers 1 \
#    --enc_in 21 \
#    --dec_in 21 \
#    --c_out 21 \
#    --n_heads 8 \
#    --d_model 64 \
#    --d_ff 64 \
#    --patch_len 2 \
#    --stride 2 \
#    --head_dropout 0.1 \
#    --dropout 0.2 \
#    --time_steps 1000 \
#    --scheduler cosine \
#    --lr_decay 0.95 \
#    --learning_rate 0.001 \
#    --batch_size 16 \
#    --train_epochs 50 \
#    --down_sampling_layer 2 \
#    --down_sampling_window 2 \
#> ./run_log/log_202411241520_win/weather/'TimeDART_pretrain'0.01.log 2>&1
#
#
#
#for pred_len in 96; do
#    python -u run.py \
#        --task_name finetune \
#        --is_training 1 \
#        --root_path ./datasets/weather/ \
#        --data_path weather.csv \
#        --model_id Weather \
#        --model TimeDART \
#        --data Weather \
#        --features M \
#        --input_len 336 \
#        --label_len 48 \
#        --pred_len $pred_len \
#        --e_layers 2 \
#        --enc_in 21 \
#        --dec_in 21 \
#        --c_out 21 \
#        --n_heads 8 \
#        --d_model 64 \
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
#        --learning_rate 0.0004 \
#        --pct_start 0.3 \
#    --down_sampling_layer 2 \
#    --down_sampling_window 2 \
#> ./run_log/log_202411241520_win/weather/'TimeDART_finetune'$pred_len_0.01.log 2>&1
#
#done
#
##Traffic
#python -u run.py \
#    --task_name pretrain \
#    --root_path ./datasets/traffic/ \
#    --data_path traffic.csv \
#    --model_id Traffic \
#    --model TimeDART \
#    --data Traffic \
#    --features M \
#    --input_len 336 \
#    --e_layers 3 \
#    --d_layers 1 \
#    --enc_in 862 \
#    --dec_in 862 \
#    --c_out 862 \
#    --n_heads 16 \
#    --d_model 128 \
#    --d_ff 256 \
#    --patch_len 8 \
#    --stride 8 \
#    --head_dropout 0.1 \
#    --dropout 0.2 \
#    --time_steps 1000 \
#    --scheduler cosine \
#    --lr_decay 0.95 \
#    --learning_rate 0.0001 \
#    --batch_size 8 \
#    --train_epochs 50 \
#    --down_sampling_layer 2 \
#    --down_sampling_window 2 \
#> ./run_log/log_202411241520_win/Traffic/'TimeDART_pretrain'0.01.log 2>&1
#
#
#
#for pred_len in 96; do
#    python -u run.py \
#        --task_name finetune \
#        --is_training 1 \
#        --root_path ./datasets/traffic/ \
#        --data_path traffic.csv \
#        --model_id Traffic \
#        --model TimeDART \
#        --data Traffic \
#        --features M \
#        --input_len 336 \
#        --label_len 48 \
#        --pred_len $pred_len \
#        --e_layers 3 \
#        --enc_in 862 \
#        --dec_in 862 \
#        --c_out 862 \
#        --n_heads 16 \
#        --d_model 128 \
#        --d_ff 256 \
#        --patch_len 8 \
#        --stride 8 \
#        --dropout 0.2 \
#        --head_dropout 0.1 \
#        --batch_size 8 \
#        --lr_decay 0.5 \
#        --lradj step \
#        --time_steps 1000 \
#        --scheduler cosine \
#        --patience 3 \
#        --learning_rate 0.003 \
#        --pct_start 0.2 \
#    --down_sampling_layer 2 \
#    --down_sampling_window 2 \
#> ./run_log/log_202411241520_win/Traffic/'TimeDART_finetune'$pred_len_0.01.log 2>&1
#
#done
#
##ecl
#
#python -u run.py \
#    --task_name pretrain \
#    --root_path ./datasets/electricity/ \
#    --data_path electricity.csv \
#    --model_id Electricity \
#    --model TimeDART \
#    --data Electricity \
#    --features M \
#    --input_len 336 \
#    --e_layers 2 \
#    --d_layers 1 \
#    --enc_in 321 \
#    --dec_in 321 \
#    --c_out 321 \
#    --n_heads 16 \
#    --d_model 128 \
#    --d_ff 256 \
#    --patch_len 8 \
#    --stride 8 \
#    --head_dropout 0.1 \
#    --dropout 0.2 \
#    --time_steps 1000 \
#    --scheduler cosine \
#    --lr_decay 0.95 \
#    --learning_rate 0.0001 \
#    --batch_size 16 \
#    --down_sampling_layer 2 \
#    --down_sampling_window 2 \
#    --train_epochs 50 \
#> ./run_log/log_202411241520_win/electricity/'TimeDART_pretrain'0.01.log 2>&1
#
#for pred_len in 96; do
#    python -u run.py \
#                --task_name finetune \
#        --is_training 1 \
#        --root_path ./datasets/electricity/ \
#        --data_path electricity.csv \
#        --model_id Electricity \
#        --model TimeDART \
#        --data Electricity \
#        --features M \
#        --input_len 336 \
#        --label_len 48 \
#        --pred_len $pred_len \
#        --e_layers 2 \
#        --enc_in 321 \
#        --dec_in 321 \
#        --c_out 321 \
#        --n_heads 16 \
#        --d_model 128 \
#        --d_ff 256 \
#        --patch_len 8 \
#        --stride 8 \
#        --dropout 0.2 \
#        --head_dropout 0.0 \
#        --batch_size 16 \
#        --lr_decay 0.5 \
#        --lradj step \
#        --time_steps 1000 \
#        --scheduler cosine \
#        --patience 3 \
#        --learning_rate 0.0004 \
#    --down_sampling_layer 2 \
#    --down_sampling_window 2 \
#        --pct_start 0.3 \
#> ./run_log/log_202411241520_win/electricity/'TimeDART_finetune'$pred_len_0.01.log 2>&1
#
#done