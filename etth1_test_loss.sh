if [ ! -d "./run_log" ]; then
    mkdir ./run_log
fi
if [ ! -d "./run_log/log_202504070853_win" ]; then
    mkdir ./run_log/log_202504070853_win
fi
if [ ! -d "./run_log/log_202504070853_win/ETTm1" ]; then
    mkdir ./run_log/log_202504070853_win/ETTm1
fi
if [ ! -d "./run_log/log_202504070853_win/ETTh1" ]; then
    mkdir ./run_log/log_202504070853_win/ETTh1
fi
if [ ! -d "./run_log/log_202504070853_win/ETTm2" ]; then
    mkdir ./run_log/log_202504070853_win/ETTm2
fi

if [ ! -d "./run_log/log_202504070853_win/ETTh2" ]; then
    mkdir ./run_log/log_202504070853_win/ETTh2
fi
if [ ! -d "./run_log/log_202504070853_win/electricity" ]; then
    mkdir ./run_log/log_202504070853_win/electricity
fi

if [ ! -d "./run_log/log_202504070853_win/Exchange" ]; then
    mkdir ./run_log/log_202504070853_win/Exchange
fi

#if [ ! -d "./run_log/log_202504070853_win/Solar" ]; then
#    mkdir ./run_log/log_202504070853_win/Solar
#fi

if [ ! -d "./run_log/log_202504070853_win/weather" ]; then
    mkdir ./run_log/log_202504070853_win/weather
fi

if [ ! -d "./run_log/log_202504070853_win/Traffic" ]; then
    mkdir ./run_log/log_202504070853_win/Traffic
fi
#
#if [ ! -d "./run_log/log_202504070853_win/PEMS03" ]; then
#    mkdir ./run_log/log_202504070853_win/PEMS03
#fi
#
#if [ ! -d "./run_log/log_202504070853_win/PEMS04" ]; then
#    mkdir ./run_log/log_202504070853_win/PEMS04
#fi
#
#if [ ! -d "./run_log/log_202504070853_win/PEMS07" ]; then
#    mkdir ./run_log/log_202504070853_win/PEMS07
#fi
#if [ ! -d "./run_log/log_202504070853_win/PEMS08" ]; then
#    mkdir ./run_log/log_202504070853_win/PEMS08
#fi




#上个个版本  2213
#这个版本 3322

del_orth_loss=0
del_season_freq_loss=1
del_smoothness_loss=0
del_freq_loss=1










for denoise_layers_num in 3 4 5 6 7;do

#
#
echo "ETTh1 $denoise_layers_num"
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
    --denoise_layers_num $denoise_layers_num \
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
    --del_orth_loss $del_orth_loss \
    --del_season_freq_loss $del_season_freq_loss \
    --del_smoothness_loss $del_smoothness_loss \
    --del_freq_loss $del_freq_loss \
    --down_sampling_layer 2 \
    --down_sampling_window 2 \
> ./run_log/log_202504070853_win/ETTh1/'TimeDART_pretrain'0.01.log 2>&1

for pred_len in   96 192 336 720; do
echo "ETTh1 $denoise_layers_num _ $pred_len"

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
        --denoise_layers_num $denoise_layers_num \
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
        --del_orth_loss $del_orth_loss \
        --del_season_freq_loss $del_season_freq_loss \
        --del_smoothness_loss $del_smoothness_loss \
        --del_freq_loss $del_freq_loss \
        --down_sampling_layer 2 \
        --down_sampling_window 2 \
> ./run_log/log_202504070853_win/ETTh1/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_'0.01.log 2>&1
done

# ETTh2

echo "ETTh2 $denoise_layers_num "

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
    --d_model 32 \
    --denoise_layers_num $denoise_layers_num \
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
    --del_orth_loss $del_orth_loss \
    --del_season_freq_loss $del_season_freq_loss \
    --del_smoothness_loss $del_smoothness_loss \
    --del_freq_loss $del_freq_loss \
    --down_sampling_layer 2 \
    --down_sampling_window 2 \
> ./run_log/log_202504070853_win/ETTh2/'TimeDART_pretrain'0.01.log 2>&1



for pred_len in 96 192 336 720; do
echo "ETTh2 $denoise_layers_num _ $pred_len"

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
        --denoise_layers_num $denoise_layers_num \
        --n_heads 8 \
        --d_model 32 \
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
        --del_orth_loss $del_orth_loss \
        --del_season_freq_loss $del_season_freq_loss \
        --del_smoothness_loss $del_smoothness_loss \
        --del_freq_loss $del_freq_loss \
        --learning_rate 0.0001 \
        --pct_start 0.3 \
        --down_sampling_layer 2 \
        --down_sampling_window 2 \
> ./run_log/log_202504070853_win/ETTh2/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_'0.01.log 2>&1

done




# ETTm1
echo "ETTm1 $denoise_layers_num "

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
    --denoise_layers_num $denoise_layers_num \
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
    --del_orth_loss $del_orth_loss \
    --del_season_freq_loss $del_season_freq_loss \
    --del_smoothness_loss $del_smoothness_loss \
    --del_freq_loss $del_freq_loss \
    --down_sampling_layer 2 \
    --down_sampling_window 2 \
> ./run_log/log_202504070853_win/ETTm1/'TimeDART_pretrain'0.01.log 2>&1



for pred_len in 96 192 336 720; do
echo "ETTm1 $denoise_layers_num _ $pred_len"

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
        --denoise_layers_num $denoise_layers_num \
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
        --del_orth_loss $del_orth_loss \
        --del_season_freq_loss $del_season_freq_loss \
        --del_smoothness_loss $del_smoothness_loss \
        --del_freq_loss $del_freq_loss \
        --patience 3 \
        --learning_rate 0.0001 \
        --pct_start 0.3 \
       --down_sampling_layer 2 \
       --down_sampling_window 2 \
> ./run_log/log_202504070853_win/ETTm1/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_'0.01.log 2>&1

done






##ETTm2
echo "ETTm2 $denoise_layers_num "

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
    --d_model 32 \
    --d_ff 16 \
    --patch_len 2 \
    --stride 2 \
    --head_dropout 0.1 \
    --denoise_layers_num $denoise_layers_num \
    --dropout 0.2 \
    --time_steps 1000 \
    --scheduler cosine \
    --lr_decay 0.8 \
    --learning_rate 0.001 \
    --del_orth_loss $del_orth_loss \
    --del_season_freq_loss $del_season_freq_loss \
    --del_smoothness_loss $del_smoothness_loss \
    --del_freq_loss $del_freq_loss \
    --batch_size 64 \
    --train_epochs 50 \
    --down_sampling_layer 2 \
    --down_sampling_window 2 \
> ./run_log/log_202504070853_win/ETTm2/'TimeDART_pretrain'0.01.log 2>&1



for pred_len in 96 192 336 720; do
echo "ETTm2 $denoise_layers_num _ $pred_len"

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
        --d_model 32 \
        --d_ff 16 \
        --patch_len 2 \
        --stride 2 \
        --dropout 0.4 \
        --head_dropout 0.1 \
        --denoise_layers_num $denoise_layers_num \
        --batch_size 64 \
        --lr_decay 0.5 \
        --lradj step \
        --del_orth_loss $del_orth_loss \
        --del_season_freq_loss $del_season_freq_loss \
        --del_smoothness_loss $del_smoothness_loss \
        --del_freq_loss $del_freq_loss \
        --time_steps 1000 \
        --scheduler cosine \
        --patience 3 \
        --learning_rate 0.0001 \
        --pct_start 0.2 \
       --down_sampling_layer 2 \
       --down_sampling_window 2 \
> ./run_log/log_202504070853_win/ETTm2/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_'0.01.log 2>&1

done

done