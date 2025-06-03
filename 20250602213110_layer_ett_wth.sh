if [ ! -d "./run_log" ]; then
    mkdir ./run_log
fi
if [ ! -d "./run_log/log_202506022128_win" ]; then
    mkdir ./run_log/log_202506022128_win
fi
if [ ! -d "./run_log/log_202506022128_win/ETTm1" ]; then
    mkdir ./run_log/log_202506022128_win/ETTm1
fi
if [ ! -d "./run_log/log_202506022128_win/ETTh1" ]; then
    mkdir ./run_log/log_202506022128_win/ETTh1
fi
if [ ! -d "./run_log/log_202506022128_win/ETTm2" ]; then
    mkdir ./run_log/log_202506022128_win/ETTm2
fi

if [ ! -d "./run_log/log_202506022128_win/ETTh2" ]; then
    mkdir ./run_log/log_202506022128_win/ETTh2
fi
if [ ! -d "./run_log/log_202506022128_win/electricity" ]; then
    mkdir ./run_log/log_202506022128_win/electricity
fi

if [ ! -d "./run_log/log_202506022128_win/Exchange" ]; then
    mkdir ./run_log/log_202506022128_win/Exchange
fi

#if [ ! -d "./run_log/log_202506022128_win/Solar" ]; then
#    mkdir ./run_log/log_202506022128_win/Solar
#fi

if [ ! -d "./run_log/log_202506022128_win/weather" ]; then
    mkdir ./run_log/log_202506022128_win/weather
fi

if [ ! -d "./run_log/log_202506022128_win/Traffic" ]; then
    mkdir ./run_log/log_202506022128_win/Traffic
fi
#
#if [ ! -d "./run_log/log_202506022128_win/PEMS03" ]; then
#    mkdir ./run_log/log_202506022128_win/PEMS03
#fi
#
#if [ ! -d "./run_log/log_202506022128_win/PEMS04" ]; then
#    mkdir ./run_log/log_202506022128_win/PEMS04
#fi
#
#if [ ! -d "./run_log/log_202506022128_win/PEMS07" ]; then
#    mkdir ./run_log/log_202506022128_win/PEMS07
#fi
#if [ ! -d "./run_log/log_202506022128_win/PEMS08" ]; then
#    mkdir ./run_log/log_202506022128_win/PEMS08
#fi




#上个个版本  2213
#这个版本 3322








#------------------------
del_orth_loss=0
del_season_freq_loss=0
del_smoothness_loss=0
del_freq_loss=0
del_recon_loss=0

log_var_freq=0.2
log_var_orth=6.0
log_var_smooth=1.0
log_var_season_freq=4.0
log_var_recon=0.1

for denoise_layers_num in 1 2 3 4 5 6 7 8;do

#
#

#echo "ETTh1 $denoise_layers_num"
#python -u run.py \
#    --task_name pretrain \
#    --root_path ./datasets/ETT-small/ \
#    --data_path ETTh1.csv \
#    --model_id ETTh1 \
#    --model TimeDART \
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
#    --denoise_layers_num $denoise_layers_num \
#    --patch_len 2 \
#    --stride 2 \
#    --head_dropout 0.1 \
#    --dropout 0.2 \
#    --time_steps 1000 \
#    --scheduler cosine \
#    --lr_decay 0.9 \
#    --learning_rate 0.0001 \
#    --batch_size 16 \
#    --train_epochs 10 \
#    --del_orth_loss $del_orth_loss \
#    --del_season_freq_loss $del_season_freq_loss \
#    --del_smoothness_loss $del_smoothness_loss \
#    --del_freq_loss $del_freq_loss \
#    --del_recon_loss $del_recon_loss \
#    --log_var_recon $log_var_recon \
#    --log_var_freq $log_var_freq \
#    --log_var_orth $log_var_orth \
#    --log_var_smooth $log_var_smooth \
#    --log_var_season_freq $log_var_season_freq \
#    --down_sampling_layer 2 \
#    --down_sampling_window 2 \
#> ./run_log/log_202506022128_win/ETTh1/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_'0.01.log 2>&1
#
#for pred_len in   96 192 336 720; do
#echo "ETTh1 $denoise_layers_num _ $pred_len"
#
#    python -u run.py \
#        --task_name finetune \
#        --is_training 1 \
#        --root_path ./datasets/ETT-small/ \
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
#        --denoise_layers_num $denoise_layers_num \
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
#        --del_orth_loss $del_orth_loss \
#        --del_season_freq_loss $del_season_freq_loss \
#        --del_smoothness_loss $del_smoothness_loss \
#        --del_recon_loss $del_recon_loss \
#        --log_var_recon $log_var_recon \
#        --del_freq_loss $del_freq_loss \
#        --log_var_freq $log_var_freq \
#        --log_var_orth $log_var_orth \
#        --log_var_smooth $log_var_smooth \
#        --log_var_season_freq $log_var_season_freq \
#        --down_sampling_layer 2 \
#        --down_sampling_window 2 \
#> ./run_log/log_202506022128_win/ETTh1/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_'0.01.log 2>&1
#done





log_var_freq=0.2
log_var_orth=6.0
log_var_smooth=1.0
log_var_season_freq=4.0
log_var_recon=0.1

echo "ETTh2 $denoise_layers_num "

#python -u run.py \
#     --task_name pretrain \
#    --root_path ./datasets/ETT-small/ \
#    --data_path ETTh2.csv \
#    --model_id ETTh2 \
#    --model TimeDART \
#    --data ETTh2 \
#    --features M \
#    --input_len 336 \
#    --e_layers 2 \
#    --d_layers 1 \
#    --enc_in 7 \
#    --dec_in 7 \
#    --c_out 7 \
#    --n_heads 8 \
#    --d_model 32 \
#    --denoise_layers_num $denoise_layers_num \
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
#    --train_epochs 10 \
#    --del_orth_loss $del_orth_loss \
#    --del_season_freq_loss $del_season_freq_loss \
#    --del_smoothness_loss $del_smoothness_loss \
#    --del_freq_loss $del_freq_loss \
#    --del_recon_loss $del_recon_loss \
#    --log_var_recon $log_var_recon \
#    --log_var_freq $log_var_freq \
#    --log_var_orth $log_var_orth \
#    --log_var_smooth $log_var_smooth \
#    --log_var_season_freq $log_var_season_freq \
#    --down_sampling_layer 2 \
#    --down_sampling_window 2 \
#> ./run_log/log_202506022128_win/ETTh2/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_'0.01.log 2>&1



#for pred_len in 96 192 336 720; do
#echo "ETTh2 $denoise_layers_num _ $pred_len"
#
#    python -u run.py \
#        --task_name finetune \
#        --is_training 1 \
#        --root_path ./datasets/ETT-small/ \
#        --data_path ETTh2.csv \
#        --model_id ETTh2 \
#        --model TimeDART \
#        --data ETTh2 \
#        --features M \
#        --input_len 336 \
#        --label_len 48 \
#        --pred_len $pred_len \
#        --e_layers 2 \
#        --enc_in 7 \
#        --dec_in 7 \
#        --c_out 7 \
#        --denoise_layers_num $denoise_layers_num \
#        --n_heads 8 \
#        --d_model 32 \
#        --d_ff 32 \
#        --patch_len 2 \
#        --stride 2 \
#        --dropout 0.4 \
#        --head_dropout 0.1 \
#        --batch_size 16 \
#        --lr_decay 0.5 \
#        --lradj step \
#        --time_steps 1000 \
#        --scheduler cosine \
#        --patience 3 \
#        --del_orth_loss $del_orth_loss \
#        --del_season_freq_loss $del_season_freq_loss \
#        --del_smoothness_loss $del_smoothness_loss \
#        --del_freq_loss $del_freq_loss \
#        --del_recon_loss $del_recon_loss \
#        --log_var_recon $log_var_recon \
#        --log_var_freq $log_var_freq \
#        --log_var_orth $log_var_orth \
#        --log_var_smooth $log_var_smooth \
#        --log_var_season_freq $log_var_season_freq \
#        --learning_rate 0.0001 \
#        --pct_start 0.3 \
#        --down_sampling_layer 2 \
#        --down_sampling_window 2 \
#> ./run_log/log_202506022128_win/ETTh2/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_'0.01.log 2>&1
#
#done



log_var_freq=0.2
log_var_orth=8.0
log_var_smooth=1.0
log_var_season_freq=4.0
log_var_recon=0.1
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
#    --train_epochs 10 \
#    --del_orth_loss $del_orth_loss \
#    --del_season_freq_loss $del_season_freq_loss \
#    --del_smoothness_loss $del_smoothness_loss \
#    --del_freq_loss $del_freq_loss \
#    --del_recon_loss $del_recon_loss \
#    --log_var_recon $log_var_recon \
#    --log_var_freq $log_var_freq \
#    --log_var_orth $log_var_orth \
#    --log_var_smooth $log_var_smooth \
#    --log_var_season_freq $log_var_season_freq \
#    --down_sampling_layer 2 \
#    --down_sampling_window 2 \
#> ./run_log/log_202506022128_win/ETTm1/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_'0.01.log 2>&1

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
        --n_heads  8 \
        --d_model 128 \
        --d_ff 64 \
        --patch_len 2 \
        --stride 2 \
        --dropout 0.1 \
        --head_dropout 0.0 \
        --batch_size 16 \
        --lr_decay 0.5 \
        --lradj step \
        --time_steps 1000 \
        --scheduler cosine \
        --del_orth_loss $del_orth_loss \
        --del_season_freq_loss $del_season_freq_loss \
        --del_smoothness_loss $del_smoothness_loss \
        --del_freq_loss $del_freq_loss \
        --del_recon_loss $del_recon_loss \
        --log_var_recon $log_var_recon \
        --log_var_freq $log_var_freq \
        --log_var_orth $log_var_orth \
        --log_var_smooth $log_var_smooth \
        --log_var_season_freq $log_var_season_freq \
        --patience 3 \
        --learning_rate 0.0001 \
        --pct_start 0.3 \
       --down_sampling_layer 2 \
       --down_sampling_window 2 \
> ./run_log/log_202506022128_win/ETTm1/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_'0.01.log 2>&1

done



log_var_freq=0.2
log_var_orth=8.0
log_var_smooth=1.0
log_var_season_freq=4.0
log_var_recon=0.1
##ETTm2
#denoise_layers_num=3
echo "ETTm2 $denoise_layers_num "

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
#    --denoise_layers_num $denoise_layers_num \
#    --dropout 0.2 \
#    --time_steps 1000 \
#    --scheduler cosine \
#    --lr_decay 0.8 \
#    --learning_rate 0.001 \
#    --batch_size 64 \
#    --del_orth_loss $del_orth_loss \
#    --del_season_freq_loss $del_season_freq_loss \
#    --del_smoothness_loss $del_smoothness_loss \
#    --del_recon_loss $del_recon_loss \
#    --log_var_recon $log_var_recon \
#    --del_freq_loss $del_freq_loss \
#    --log_var_freq $log_var_freq \
#    --log_var_orth $log_var_orth \
#    --log_var_smooth $log_var_smooth \
#    --log_var_season_freq $log_var_season_freq \
#    --train_epochs 10 \
#    --down_sampling_layer 2 \
#    --down_sampling_window 2 \
#> ./run_log/log_202506022128_win/ETTm2/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_'0.01.log 2>&1



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
        --d_model 8 \
        --d_ff 16 \
        --patch_len 2 \
        --stride 2 \
        --dropout 0.4 \
        --head_dropout 0.1 \
        --denoise_layers_num $denoise_layers_num \
        --batch_size 64 \
        --lr_decay 0.5 \
        --lradj step \
        --time_steps 1000 \
        --scheduler cosine \
        --patience 3 \
        --learning_rate 0.0001 \
        --del_orth_loss $del_orth_loss \
        --del_season_freq_loss $del_season_freq_loss \
        --del_smoothness_loss $del_smoothness_loss \
        --del_recon_loss $del_recon_loss \
        --log_var_recon $log_var_recon \
        --del_freq_loss $del_freq_loss \
        --log_var_freq $log_var_freq \
        --log_var_orth $log_var_orth \
        --log_var_smooth $log_var_smooth \
        --log_var_season_freq $log_var_season_freq \
        --pct_start 0.2 \
       --down_sampling_layer 2 \
       --down_sampling_window 2 \
> ./run_log/log_202506022128_win/ETTm2/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_'0.01.log 2>&1

done
log_var_freq=0.2
log_var_orth=8.0
log_var_smooth=1.0
log_var_season_freq=4.0
log_var_recon=0.1

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
    --del_recon_loss $del_recon_loss \
    --log_var_recon $log_var_recon \
    --log_var_freq $log_var_freq \
    --log_var_orth $log_var_orth \
    --log_var_smooth $log_var_smooth \
    --log_var_season_freq $log_var_season_freq \
    --lr_decay 0.95 \
    --learning_rate 0.001 \
    --batch_size 16 \
    --train_epochs 10 \
    --down_sampling_layer 2 \
    --down_sampling_window 2 \
> ./run_log/log_202506022128_win/weather/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_'0.01.log 2>&1



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
        --del_recon_loss $del_recon_loss \
        --log_var_recon $log_var_recon \
        --log_var_freq $log_var_freq \
        --log_var_orth $log_var_orth \
        --log_var_smooth $log_var_smooth \
        --log_var_season_freq $log_var_season_freq \
        --lr_decay 0.5 \
        --lradj step \
        --time_steps 1000 \
        --scheduler cosine \
        --patience 3 \
        --learning_rate 0.0004 \
        --pct_start 0.3 \
    --down_sampling_layer 2 \
    --down_sampling_window 2 \
> ./run_log/log_202506022128_win/weather/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_'0.01.log 2>&1

done



done




