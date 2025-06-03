if [ ! -d "./run_log" ]; then
    mkdir ./run_log
fi
if [ ! -d "./run_log/log_202505251118_win" ]; then
    mkdir ./run_log/log_202505251118_win
fi
if [ ! -d "./run_log/log_202505251118_win/ETTm1" ]; then
    mkdir ./run_log/log_202505251118_win/ETTm1
fi
if [ ! -d "./run_log/log_202505251118_win/ETTh1" ]; then
    mkdir ./run_log/log_202505251118_win/ETTh1
fi
if [ ! -d "./run_log/log_202505251118_win/ETTm2" ]; then
    mkdir ./run_log/log_202505251118_win/ETTm2
fi

if [ ! -d "./run_log/log_202505251118_win/ETTh2" ]; then
    mkdir ./run_log/log_202505251118_win/ETTh2
fi
if [ ! -d "./run_log/log_202505251118_win/electricity" ]; then
    mkdir ./run_log/log_202505251118_win/electricity
fi

if [ ! -d "./run_log/log_202505251118_win/Exchange" ]; then
    mkdir ./run_log/log_202505251118_win/Exchange
fi

#if [ ! -d "./run_log/log_202505251118_win/Solar" ]; then
#    mkdir ./run_log/log_202505251118_win/Solar
#fi

if [ ! -d "./run_log/log_202505251118_win/weather" ]; then
    mkdir ./run_log/log_202505251118_win/weather
fi

if [ ! -d "./run_log/log_202505251118_win/Traffic" ]; then
    mkdir ./run_log/log_202505251118_win/Traffic
fi
#
#if [ ! -d "./run_log/log_202505251118_win/PEMS03" ]; then
#    mkdir ./run_log/log_202505251118_win/PEMS03
#fi
#
#if [ ! -d "./run_log/log_202505251118_win/PEMS04" ]; then
#    mkdir ./run_log/log_202505251118_win/PEMS04
#fi
#
#if [ ! -d "./run_log/log_202505251118_win/PEMS07" ]; then
#    mkdir ./run_log/log_202505251118_win/PEMS07
#fi
#if [ ! -d "./run_log/log_202505251118_win/PEMS08" ]; then
#    mkdir ./run_log/log_202505251118_win/PEMS08
#fi




#上个个版本  2213
#这个版本 3322

del_orth_loss=0
del_season_freq_loss=1
del_smoothness_loss=0
del_freq_loss=1







for denoise_layers_num in 1;do



# 测试 dmodel 原： 32  改 64  patchlen-stride   原： 2  改 4   0.432  差
#336->96, 0.427, 0.431,2025-05-21 19:39:59
#336->192, 0.449, 0.444,2025-05-21 19:49:13
# 测试 patchlen-stride   原： 2  改 4
#336->96, 0.430, 0.431,2025-05-21 21:27:22
#336->192, 0.452, 0.444,2025-05-21 21:36:23
#336->336, 0.478, 0.458,2025-05-21 21:45:29
#336->720, 0.521, 0.481,2025-05-21 21:53:48
# 测试 dmodel   原： 32  改 64
#336->96, 0.425, 0.430,2025-05-22 11:13:47
#336->192, 0.448, 0.443,2025-05-22 11:38:07
#336->336, 0.476, 0.458,2025-05-22 11:53:49
#336->720, 0.518, 0.480,2025-05-22 12:10:46
# ETTm1
echo "ETTm1 $denoise_layers_num "

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
#    --denoise_layers_num $denoise_layers_num \
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
#    --del_orth_loss $del_orth_loss \
#    --del_season_freq_loss $del_season_freq_loss \
#    --del_smoothness_loss $del_smoothness_loss \
#    --del_freq_loss $del_freq_loss \
#    --down_sampling_layer 2 \
#    --down_sampling_window 2 \
#> ./run_log/log_202505251118_win/ETTm1/'TimeDART_pretrain'0.01.log 2>&1

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
        --n_heads  1 \
        --d_model 128 \
        --d_ff 64 \
        --patch_len 2 \
        --stride 2 \
        --dropout 0.1 \
        --head_dropout 0.0 \
        --batch_size 8 \
        --lr_decay 0.5 \
        --lradj step \
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
> ./run_log/log_202505251118_win/ETTm1/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_'0.01.log 2>&1

done






 ## 不好
##Exchange
# 测试 patch_len 原： 2  改 4
#336->96, 0.115, 0.241,2025-05-22 07:22:36
#336->192, 0.214, 0.333,2025-05-22 07:23:55
#336->336, 0.378, 0.450,2025-05-22 07:25:07
#336->720, 1.123, 0.806,2025-05-22 07:25:37
# 测试 d_model 原： 32  改 64

#336->96, 0.132, 0.259,2025-05-22 12:38:03
#336->192, 0.223, 0.340,2025-05-22 12:39:59
#336->336, 0.385, 0.455,2025-05-22 12:41:51
#336->720, 1.118, 0.804,2025-05-22 12:42:46

echo "Exchange $denoise_layers_num"

#python -u run.py \
#    --task_name pretrain \
#    --root_path ./datasets/exchange_rate/ \
#    --data_path exchange_rate.csv \
#    --model_id Exchange \
#    --model TimeDART \
#    --data Exchange \
#    --features M \
#    --input_len 336 \
#    --e_layers 2 \
#    --d_layers 1 \
#    --enc_in 8 \
#    --dec_in 8 \
#    --denoise_layers_num $denoise_layers_num \
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
#    --del_orth_loss $del_orth_loss \
#    --del_season_freq_loss $del_season_freq_loss \
#    --del_smoothness_loss $del_smoothness_loss \
#    --del_freq_loss $del_freq_loss \
#    --down_sampling_layer 2 \
#    --down_sampling_window 2 \
#> ./run_log/log_202505251118_win/Exchange/'TimeDART_pretrain'0.01.log 2>&1

for pred_len in 96 192 336 720; do
echo "Exchange $denoise_layers_num _ $pred_len"
    python -u run.py \
        --task_name finetune \
        --is_training 1 \
        --root_path ./datasets/exchange_rate/ \
        --data_path exchange_rate.csv \
        --model_id Exchange \
        --model TimeDART \
        --data Exchange \
        --features M \
        --input_len 336 \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 2 \
        --denoise_layers_num $denoise_layers_num \
        --enc_in 8 \
        --dec_in 8 \
        --c_out 8 \
        --n_heads 1 \
        --d_model 256 \
        --d_ff 64 \
        --patch_len 2 \
        --stride 2 \
        --dropout 0.4 \
        --head_dropout 0.1 \
        --batch_size 8 \
        --lr_decay 0.8 \
        --lradj step \
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
> ./run_log/log_202505251118_win/Exchange/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_'0.01.log 2>&1

done
done


#---------------------------------------------------------------weather

#for denoise_layers_num in 1 2 3 4 5 6 7 8;do
#
#
## 测试 dmodel 原： 64  改 32
## 336->96, 0.148, 0.198,2025-05-22 05:29:57
## 336->192, 0.191, 0.240,2025-05-22 06:03:08
## 336->336, 0.241, 0.277,2025-05-22 06:36:30
## 336->720, 0.317, 0.331,2025-05-22 07:09:52
#
#echo "Weather $denoise_layers_num "
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
#    --denoise_layers_num $denoise_layers_num \
#    --c_out 21 \
#    --n_heads 8 \
#    --d_model 32 \
#    --d_ff 64 \
#    --patch_len 2 \
#    --stride 2 \
#    --head_dropout 0.1 \
#    --dropout 0.2 \
#    --time_steps 1000 \
#    --scheduler cosine \
#    --del _orth_loss $del_orth_loss \
#    --del_season_freq_loss $del_season_freq_loss \
#    --del_smoothness_loss $del_smoothness_loss \
#    --del_freq_loss $del_freq_loss \
#    --lr_decay 0.95 \
#    --learning_rate 0.001 \
#    --batch_size 16 \
#    --train_epochs 50 \
#    --down_sampling_layer 2 \
#    --down_sampling_window 2 \
#> ./run_log/log_202505251118_win/weather/'TimeDART_pretrain'0.01.log 2>&1
#
#
#
#for pred_len in 96 192 336 720; do
#echo "Weather $denoise_layers_num _ $pred_len"
#
#    python -u run.py \
#        --task_name finetune \
#        --is_training 1 \
#        --root_path ./datasets/weather/ \
#        --data_path weather.csv \
#        --model_id Weather \
#        --model TimeDART \
#        --data Weather \
#        --denoise_layers_num $denoise_layers_num \
#        --features M \
#        --input_len 336 \
#        --label_len 48 \
#        --pred_len $pred_len \
#        --e_layers 2 \
#        --enc_in 21 \
#        --dec_in 21 \
#        --c_out 21 \
#        --n_heads 8 \
#        --d_model 32 \
#        --d_ff 64 \
#        --patch_len 2 \
#        --stride 2 \
#        --dropout 0.2 \
#        --head_dropout 0.1 \
#        --batch_size 16 \
#        --del_orth_loss $del_orth_loss \
#        --del_season_freq_loss $del_season_freq_loss \
#        --del_smoothness_loss $del_smoothness_loss \
#        --del_freq_loss $del_freq_loss \
#        --lr_decay 0.5 \
#        --lradj step \
#        --time_steps 1000 \
#        --scheduler cosine \
#        --patience 3 \
#        --learning_rate 0.0004 \
#        --pct_start 0.3 \
#    --down_sampling_layer 2 \
#    --down_sampling_window 2 \
#> ./run_log/log_202505251118_win/weather/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_'0.01.log 2>&1
#
#done
#
#done