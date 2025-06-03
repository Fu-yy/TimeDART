if [ ! -d "./run_log" ]; then
    mkdir ./run_log
fi
if [ ! -d "./run_log/log_202506011034_win" ]; then
    mkdir ./run_log/log_202506011034_win
fi
if [ ! -d "./run_log/log_202506011034_win/ETTm1" ]; then
    mkdir ./run_log/log_202506011034_win/ETTm1
fi
if [ ! -d "./run_log/log_202506011034_win/ETTh1" ]; then
    mkdir ./run_log/log_202506011034_win/ETTh1
fi
if [ ! -d "./run_log/log_202506011034_win/ETTm2" ]; then
    mkdir ./run_log/log_202506011034_win/ETTm2
fi

if [ ! -d "./run_log/log_202506011034_win/ETTh2" ]; then
    mkdir ./run_log/log_202506011034_win/ETTh2
fi
if [ ! -d "./run_log/log_202506011034_win/electricity" ]; then
    mkdir ./run_log/log_202506011034_win/electricity
fi

if [ ! -d "./run_log/log_202506011034_win/Exchange" ]; then
    mkdir ./run_log/log_202506011034_win/Exchange
fi

#if [ ! -d "./run_log/log_202506011034_win/Solar" ]; then
#    mkdir ./run_log/log_202506011034_win/Solar
#fi

if [ ! -d "./run_log/log_202506011034_win/weather" ]; then
    mkdir ./run_log/log_202506011034_win/weather
fi

if [ ! -d "./run_log/log_202506011034_win/Traffic" ]; then
    mkdir ./run_log/log_202506011034_win/Traffic
fi
#
#if [ ! -d "./run_log/log_202506011034_win/PEMS03" ]; then
#    mkdir ./run_log/log_202506011034_win/PEMS03
#fi
#
#if [ ! -d "./run_log/log_202506011034_win/PEMS04" ]; then
#    mkdir ./run_log/log_202506011034_win/PEMS04
#fi
#
#if [ ! -d "./run_log/log_202506011034_win/PEMS07" ]; then
#    mkdir ./run_log/log_202506011034_win/PEMS07
#fi
#if [ ! -d "./run_log/log_202506011034_win/PEMS08" ]; then
#    mkdir ./run_log/log_202506011034_win/PEMS08
#fi




#上个个版本  2213
#这个版本 3322

del_orth_loss=0
del_smoothness_loss=0
del_season_freq_loss=0
del_freq_loss=0

log_var_orth=0.1

log_var_smooth=0.1
log_var_season_freq=0.1
log_var_freq=0.1

for log_var_freq in 0.1 0.3 0.5 0.7 0.9;do
for denoise_layers_num in 2;do

echo "Weather $denoise_layers_num "

python -u run.py \
    --task_name pretrain \
    --root_path ./datasets/weather/ \
    --data_path weather.csv \
    --model_id Weather \
    --model TimeDART \
    --data Weather \
    --features M \
    --input_len 336 \
    --e_layers 2 \
    --d_layers 1 \
    --enc_in 21 \
    --dec_in 21 \
    --denoise_layers_num $denoise_layers_num \
    --c_out 21 \
    --n_heads 8 \
    --d_model 64 \
    --d_ff 64 \
    --patch_len 2 \
    --stride 2 \
    --head_dropout 0.1 \
    --dropout 0.2 \
    --time_steps 1000 \
    --scheduler cosine \
    --del_orth_loss $del_orth_loss \
    --del_season_freq_loss $del_season_freq_loss \
    --del_smoothness_loss $del_smoothness_loss \
    --del_freq_loss $del_freq_loss \
    --log_var_freq $log_var_freq \
    --log_var_orth $log_var_orth \
    --log_var_smooth $log_var_smooth \
    --log_var_season_freq $log_var_season_freq \
    --lr_decay 0.95 \
    --learning_rate 0.001 \
    --batch_size 16 \
    --train_epochs 50 \
    --down_sampling_layer 2 \
    --down_sampling_window 2 \
> ./run_log/log_202506011034_win/weather/'TimeDART_pretrain'0.01.log 2>&1



for pred_len in 96 192 336 720; do
echo "Weather $denoise_layers_num _ $pred_len"

    python -u run.py \
        --task_name finetune \
        --is_training 1 \
        --root_path ./datasets/weather/ \
        --data_path weather.csv \
        --model_id Weather \
        --model TimeDART \
        --data Weather \
        --denoise_layers_num $denoise_layers_num \
        --features M \
        --input_len 336 \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 2 \
        --enc_in 21 \
        --dec_in 21 \
        --c_out 21 \
        --n_heads 8 \
        --d_model 32 \
        --d_ff 64 \
        --patch_len 2 \
        --stride 2 \
        --dropout 0.2 \
        --head_dropout 0.1 \
        --batch_size 16 \
        --del_orth_loss $del_orth_loss \
        --del_season_freq_loss $del_season_freq_loss \
        --del_smoothness_loss $del_smoothness_loss \
        --del_freq_loss $del_freq_loss \
        --lr_decay 0.5 \
        --lradj step \
        --time_steps 1000 \
        --scheduler cosine \
        --patience 3 \
        --learning_rate 0.0004 \
        --pct_start 0.3 \
    --down_sampling_layer 2 \
    --down_sampling_window 2 \
> ./run_log/log_202506011034_win/weather/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_'0.01.log 2>&1

done

done
done

#------------------------

log_var_orth=0.1

log_var_smooth=0.1
log_var_season_freq=0.1
log_var_freq=0.1

for log_var_smooth in 0.3 0.5 0.7 0.9;do
for denoise_layers_num in 2;do

echo "Weather $denoise_layers_num "

python -u run.py \
    --task_name pretrain \
    --root_path ./datasets/weather/ \
    --data_path weather.csv \
    --model_id Weather \
    --model TimeDART \
    --data Weather \
    --features M \
    --input_len 336 \
    --e_layers 2 \
    --d_layers 1 \
    --enc_in 21 \
    --dec_in 21 \
    --denoise_layers_num $denoise_layers_num \
    --c_out 21 \
    --n_heads 8 \
    --d_model 64 \
    --d_ff 64 \
    --patch_len 2 \
    --stride 2 \
    --head_dropout 0.1 \
    --dropout 0.2 \
    --time_steps 1000 \
    --scheduler cosine \
    --del_orth_loss $del_orth_loss \
    --del_season_freq_loss $del_season_freq_loss \
    --del_smoothness_loss $del_smoothness_loss \
    --del_freq_loss $del_freq_loss \
    --log_var_freq $log_var_freq \
    --log_var_orth $log_var_orth \
    --log_var_smooth $log_var_smooth \
    --log_var_season_freq $log_var_season_freq \
    --lr_decay 0.95 \
    --learning_rate 0.001 \
    --batch_size 16 \
    --train_epochs 50 \
    --down_sampling_layer 2 \
    --down_sampling_window 2 \
> ./run_log/log_202506011034_win/weather/'TimeDART_pretrain'0.01.log 2>&1



for pred_len in 96 192 336 720; do
echo "Weather $denoise_layers_num _ $pred_len"

    python -u run.py \
        --task_name finetune \
        --is_training 1 \
        --root_path ./datasets/weather/ \
        --data_path weather.csv \
        --model_id Weather \
        --model TimeDART \
        --data Weather \
        --denoise_layers_num $denoise_layers_num \
        --features M \
        --input_len 336 \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 2 \
        --enc_in 21 \
        --dec_in 21 \
        --c_out 21 \
        --n_heads 8 \
        --d_model 32 \
        --d_ff 64 \
        --patch_len 2 \
        --stride 2 \
        --dropout 0.2 \
        --head_dropout 0.1 \
        --batch_size 16 \
        --del_orth_loss $del_orth_loss \
        --del_season_freq_loss $del_season_freq_loss \
        --del_smoothness_loss $del_smoothness_loss \
        --del_freq_loss $del_freq_loss \
        --lr_decay 0.5 \
        --lradj step \
        --time_steps 1000 \
        --scheduler cosine \
        --patience 3 \
        --learning_rate 0.0004 \
        --pct_start 0.3 \
    --down_sampling_layer 2 \
    --down_sampling_window 2 \
> ./run_log/log_202506011034_win/weather/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_'0.01.log 2>&1

done

done
done


#-----------------------
log_var_orth=0.1

log_var_smooth=0.1
log_var_season_freq=0.1
log_var_freq=0.1

for log_var_season_freq in 0.3 0.5 0.7 0.9;do
for denoise_layers_num in 2;do

echo "Weather $denoise_layers_num "

python -u run.py \
    --task_name pretrain \
    --root_path ./datasets/weather/ \
    --data_path weather.csv \
    --model_id Weather \
    --model TimeDART \
    --data Weather \
    --features M \
    --input_len 336 \
    --e_layers 2 \
    --d_layers 1 \
    --enc_in 21 \
    --dec_in 21 \
    --denoise_layers_num $denoise_layers_num \
    --c_out 21 \
    --n_heads 8 \
    --d_model 64 \
    --d_ff 64 \
    --patch_len 2 \
    --stride 2 \
    --head_dropout 0.1 \
    --dropout 0.2 \
    --time_steps 1000 \
    --scheduler cosine \
    --del_orth_loss $del_orth_loss \
    --del_season_freq_loss $del_season_freq_loss \
    --del_smoothness_loss $del_smoothness_loss \
    --del_freq_loss $del_freq_loss \
    --log_var_freq $log_var_freq \
    --log_var_orth $log_var_orth \
    --log_var_smooth $log_var_smooth \
    --log_var_season_freq $log_var_season_freq \
    --lr_decay 0.95 \
    --learning_rate 0.001 \
    --batch_size 16 \
    --train_epochs 50 \
    --down_sampling_layer 2 \
    --down_sampling_window 2 \
> ./run_log/log_202506011034_win/weather/'TimeDART_pretrain'0.01.log 2>&1



for pred_len in 96 192 336 720; do
echo "Weather $denoise_layers_num _ $pred_len"

    python -u run.py \
        --task_name finetune \
        --is_training 1 \
        --root_path ./datasets/weather/ \
        --data_path weather.csv \
        --model_id Weather \
        --model TimeDART \
        --data Weather \
        --denoise_layers_num $denoise_layers_num \
        --features M \
        --input_len 336 \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 2 \
        --enc_in 21 \
        --dec_in 21 \
        --c_out 21 \
        --n_heads 8 \
        --d_model 32 \
        --d_ff 64 \
        --patch_len 2 \
        --stride 2 \
        --dropout 0.2 \
        --head_dropout 0.1 \
        --batch_size 16 \
        --del_orth_loss $del_orth_loss \
        --del_season_freq_loss $del_season_freq_loss \
        --del_smoothness_loss $del_smoothness_loss \
        --del_freq_loss $del_freq_loss \
        --lr_decay 0.5 \
        --lradj step \
        --time_steps 1000 \
        --scheduler cosine \
        --patience 3 \
        --learning_rate 0.0004 \
        --pct_start 0.3 \
    --down_sampling_layer 2 \
    --down_sampling_window 2 \
> ./run_log/log_202506011034_win/weather/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_'0.01.log 2>&1

done

done
done


#-----------------
log_var_orth=0.1

log_var_smooth=0.1
log_var_season_freq=0.1
log_var_freq=0.1

for log_var_orth in 0.1 0.3 0.7 0.9;do
for denoise_layers_num in 2;do

echo "Weather $denoise_layers_num "

python -u run.py \
    --task_name pretrain \
    --root_path ./datasets/weather/ \
    --data_path weather.csv \
    --model_id Weather \
    --model TimeDART \
    --data Weather \
    --features M \
    --input_len 336 \
    --e_layers 2 \
    --d_layers 1 \
    --enc_in 21 \
    --dec_in 21 \
    --denoise_layers_num $denoise_layers_num \
    --c_out 21 \
    --n_heads 8 \
    --d_model 64 \
    --d_ff 64 \
    --patch_len 2 \
    --stride 2 \
    --head_dropout 0.1 \
    --dropout 0.2 \
    --time_steps 1000 \
    --scheduler cosine \
    --del_orth_loss $del_orth_loss \
    --del_season_freq_loss $del_season_freq_loss \
    --del_smoothness_loss $del_smoothness_loss \
    --del_freq_loss $del_freq_loss \
    --log_var_freq $log_var_freq \
    --log_var_orth $log_var_orth \
    --log_var_smooth $log_var_smooth \
    --log_var_season_freq $log_var_season_freq \
    --lr_decay 0.95 \
    --learning_rate 0.001 \
    --batch_size 16 \
    --train_epochs 50 \
    --down_sampling_layer 2 \
    --down_sampling_window 2 \
> ./run_log/log_202506011034_win/weather/'TimeDART_pretrain'0.01.log 2>&1



for pred_len in 96 192 336 720; do
echo "Weather $denoise_layers_num _ $pred_len"

    python -u run.py \
        --task_name finetune \
        --is_training 1 \
        --root_path ./datasets/weather/ \
        --data_path weather.csv \
        --model_id Weather \
        --model TimeDART \
        --data Weather \
        --denoise_layers_num $denoise_layers_num \
        --features M \
        --input_len 336 \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 2 \
        --enc_in 21 \
        --dec_in 21 \
        --c_out 21 \
        --n_heads 8 \
        --d_model 32 \
        --d_ff 64 \
        --patch_len 2 \
        --stride 2 \
        --dropout 0.2 \
        --head_dropout 0.1 \
        --batch_size 16 \
        --del_orth_loss $del_orth_loss \
        --del_season_freq_loss $del_season_freq_loss \
        --del_smoothness_loss $del_smoothness_loss \
        --del_freq_loss $del_freq_loss \
        --lr_decay 0.5 \
        --lradj step \
        --time_steps 1000 \
        --scheduler cosine \
        --patience 3 \
        --learning_rate 0.0004 \
        --pct_start 0.3 \
    --down_sampling_layer 2 \
    --down_sampling_window 2 \
> ./run_log/log_202506011034_win/weather/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_'0.01.log 2>&1

done

done
done


