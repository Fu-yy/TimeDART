if [ ! -d "./run_log" ]; then
    mkdir ./run_log
fi
if [ ! -d "./run_log/log_20250928150909_win" ]; then
    mkdir ./run_log/log_20250928150909_win
fi
if [ ! -d "./run_log/log_20250928150909_win/ETTm1" ]; then
    mkdir ./run_log/log_20250928150909_win/ETTm1
fi
if [ ! -d "./run_log/log_20250928150909_win/ETTh1" ]; then
    mkdir ./run_log/log_20250928150909_win/ETTh1
fi
if [ ! -d "./run_log/log_20250928150909_win/ETTm2" ]; then
    mkdir ./run_log/log_20250928150909_win/ETTm2
fi

if [ ! -d "./run_log/log_20250928150909_win/ETTh2" ]; then
    mkdir ./run_log/log_20250928150909_win/ETTh2
fi
if [ ! -d "./run_log/log_20250928150909_win/electricity" ]; then
    mkdir ./run_log/log_20250928150909_win/electricity
fi

if [ ! -d "./run_log/log_20250928150909_win/Exchange" ]; then
    mkdir ./run_log/log_20250928150909_win/Exchange
fi

#if [ ! -d "./run_log/log_20250928150909_win/Solar" ]; then
#    mkdir ./run_log/log_20250928150909_win/Solar
#fi

if [ ! -d "./run_log/log_20250928150909_win/weather" ]; then
    mkdir ./run_log/log_20250928150909_win/weather
fi

if [ ! -d "./run_log/log_20250928150909_win/Traffic" ]; then
    mkdir ./run_log/log_20250928150909_win/Traffic
fi
#
#if [ ! -d "./run_log/log_20250928150909_win/PEMS03" ]; then
#    mkdir ./run_log/log_20250928150909_win/PEMS03
#fi
#
#if [ ! -d "./run_log/log_20250928150909_win/PEMS04" ]; then
#    mkdir ./run_log/log_20250928150909_win/PEMS04
#fi
#
#if [ ! -d "./run_log/log_20250928150909_win/PEMS07" ]; then
#    mkdir ./run_log/log_20250928150909_win/PEMS07
#fi
#if [ ! -d "./run_log/log_20250928150909_win/PEMS08" ]; then
#    mkdir ./run_log/log_20250928150909_win/PEMS08
#fi

#-------------------------------------------------

#----------------------------------------------- version

use_defire_noise=0
use_inner_encoder=0
use_positional_encoding=1
use_sostoken=1




use_loss_compute=1
use_new_decomp=1
use_denoise=1
use_inner_new_decomp=1


mask_ratio=0.3
mask_ratio=0.8

use_geo_mask=1
use_geo_mask=0
pretrain_mode=mask
predict_eps=0


#del_orth_loss=0
#del_season_freq_loss=0
#del_freq_loss=0
#del_smoothness_loss=0
#del_recon_loss=0
#
#for mask_ratio in 0.7;do
#
#use_init_loss=1
#
#
#
## h1m1 01000
#
#
#
#log_var_freq=0.3
#log_var_orth=0.01
#log_var_smooth=0.0
#log_var_season_freq=0.05
#log_var_recon=0.0
#
#layer_h1=1
#
#
#
#
#
#
#
#
#
#
##pretrain_mode=noise
#
#
#
#
#
#
#
#
#
#
#for denoise_layers_num in 1;do
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
#    --input_len 96 \
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
#    --train_epochs 50 \
#    --del_orth_loss $del_orth_loss \
#    --mask_ratio $mask_ratio \
#    --use_geo_mask $use_geo_mask \
#    --pretrain_mode $pretrain_mode \
#    --predict_eps $predict_eps \
#    --del_season_freq_loss $del_season_freq_loss \
#    --del_smoothness_loss $del_smoothness_loss \
#    --del_freq_loss $del_freq_loss \
#    --del_recon_loss $del_recon_loss \
#    --log_var_recon $log_var_recon \
#    --log_var_freq $log_var_freq \
#    --log_var_orth $log_var_orth \
#    --log_var_smooth $log_var_smooth \
#    --log_var_season_freq $log_var_season_freq \
#    --use_init_loss $use_init_loss \
#    --use_new_decomp $use_new_decomp \
#    --use_loss_compute $use_loss_compute \
#    --use_denoise $use_denoise \
#    --use_inner_new_decomp $use_inner_new_decomp \
#    --use_defire_noise $use_defire_noise \
#    --use_positional_encoding $use_positional_encoding \
#    --use_sostoken $use_sostoken \
#    --use_inner_encoder $use_inner_encoder \
#    --down_sampling_window 2 \
#> ./run_log/log_20250928150909_win/ETTh1/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1
#d_model=32
#n_heads=16
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
#        --input_len 96 \
#        --label_len 48 \
#        --pred_len $pred_len \
#        --e_layers 2 \
#        --enc_in 7 \
#        --dec_in 7 \
#        --c_out 7 \
#        --n_heads $n_heads \
#        --d_model $d_model \
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
#    --mask_ratio $mask_ratio \
#    --use_geo_mask $use_geo_mask \
#    --pretrain_mode $pretrain_mode \
#    --predict_eps $predict_eps \
#        --del_season_freq_loss $del_season_freq_loss \
#        --del_smoothness_loss $del_smoothness_loss \
#        --del_recon_loss $del_recon_loss \
#        --log_var_recon $log_var_recon \
#        --del_freq_loss $del_freq_loss \
#        --log_var_freq $log_var_freq \
#        --log_var_orth $log_var_orth \
#        --log_var_smooth $log_var_smooth \
#        --log_var_season_freq $log_var_season_freq \
#        --use_init_loss $use_init_loss \
#    --use_new_decomp $use_new_decomp \
#    --use_loss_compute $use_loss_compute \
#    --use_denoise $use_denoise \
#    --use_inner_new_decomp $use_inner_new_decomp \
#    --use_defire_noise $use_defire_noise \
#    --use_positional_encoding $use_positional_encoding \
#    --use_sostoken $use_sostoken \
#    --use_inner_encoder $use_inner_encoder \
#        --down_sampling_window 2 \
#> ./run_log/log_20250928150909_win/ETTh1/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1
#
#done
#
#done
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#layer_m1=1
#
#for denoise_layers_num in 1;do
#
#
#
#
#
#
#
#
#log_var_freq=0.3
#log_var_orth=0.01
#log_var_smooth=0.0
#log_var_season_freq=0.05
#log_var_recon=0.0
#echo "ETTm1 $denoise_layers_num "
#
#python -u run.py \
#    --task_name pretrain \
#    --root_path ./datasets/ETT-small/ \
#    --data_path ETTm1.csv \
#    --model_id ETTm1 \
#    --model TimeDART \
#    --data ETTm1 \
#    --features M \
#    --input_len 96 \
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
#    --mask_ratio $mask_ratio \
#    --use_geo_mask $use_geo_mask \
#    --pretrain_mode $pretrain_mode \
#    --predict_eps $predict_eps \
#    --del_season_freq_loss $del_season_freq_loss \
#    --del_smoothness_loss $del_smoothness_loss \
#    --del_freq_loss $del_freq_loss \
#    --del_recon_loss $del_recon_loss \
#    --log_var_recon $log_var_recon \
#    --log_var_freq $log_var_freq \
#    --log_var_orth $log_var_orth \
#    --log_var_smooth $log_var_smooth \
#    --log_var_season_freq $log_var_season_freq \
#    --use_init_loss $use_init_loss \
#    --use_new_decomp $use_new_decomp \
#    --use_loss_compute $use_loss_compute \
#    --use_denoise $use_denoise \
#    --use_inner_new_decomp $use_inner_new_decomp \
#    --use_defire_noise $use_defire_noise \
#    --use_positional_encoding $use_positional_encoding \
#    --use_sostoken $use_sostoken \
#    --use_inner_encoder $use_inner_encoder \
#    --down_sampling_window 2 \
#> ./run_log/log_20250928150909_win/ETTm1/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1
#
#d_model=128 #  老
#d_model=16
#d_model=32
#n_heads=8
#
#for pred_len in 96 192 336 720; do
#echo "ETTm1 $denoise_layers_num _ $pred_len"
#
#    python -u run.py \
#        --task_name finetune \
#        --is_training 1 \
#        --root_path ./datasets/ETT-small/ \
#        --data_path ETTm1.csv \
#        --model_id ETTm1 \
#        --model TimeDART \
#        --data ETTm1 \
#        --features M \
#        --input_len 96 \
#        --label_len 48 \
#        --pred_len $pred_len \
#        --e_layers 2 \
#        --enc_in 7 \
#        --dec_in 7 \
#        --c_out 7 \
#        --denoise_layers_num $denoise_layers_num \
#        --n_heads  $n_heads \
#        --d_model $d_model \
#        --d_ff 64 \
#        --patch_len 2 \
#        --stride 2 \
#        --dropout 0.1 \
#        --head_dropout 0.0 \
#        --batch_size 64 \
#        --lr_decay 0.5 \
#        --lradj step \
#        --time_steps 1000 \
#        --scheduler cosine \
#        --del_orth_loss $del_orth_loss \
#    --mask_ratio $mask_ratio \
#    --use_geo_mask $use_geo_mask \
#    --pretrain_mode $pretrain_mode \
#    --predict_eps $predict_eps \
#        --del_season_freq_loss $del_season_freq_loss \
#        --del_smoothness_loss $del_smoothness_loss \
#        --del_freq_loss $del_freq_loss \
#        --del_recon_loss $del_recon_loss \
#        --log_var_recon $log_var_recon \
#        --log_var_freq $log_var_freq \
#        --log_var_orth $log_var_orth \
#        --log_var_smooth $log_var_smooth \
#        --log_var_season_freq $log_var_season_freq \
#        --patience 3 \
#        --learning_rate 0.0001 \
#        --pct_start 0.3 \
#       --use_init_loss $use_init_loss \
#    --use_new_decomp $use_new_decomp \
#    --use_loss_compute $use_loss_compute \
#    --use_denoise $use_denoise \
#    --use_inner_new_decomp $use_inner_new_decomp \
#    --use_defire_noise $use_defire_noise \
#    --use_positional_encoding $use_positional_encoding \
#    --use_sostoken $use_sostoken \
#    --use_inner_encoder $use_inner_encoder \
#       --down_sampling_window 2 \
#> ./run_log/log_20250928150909_win/ETTm1/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1
#
#done
#done
#
#
##------------------------
##del_orth_loss=0
##del_season_freq_loss=1
##del_smoothness_loss=1
##del_freq_loss=1
##del_recon_loss=1
#
#layer_h2=2
#for denoise_layers_num in 1;do
#
#
#
#
#
#log_var_freq=0.3
#log_var_orth=0.01
#log_var_smooth=0.0
#log_var_season_freq=0.05
#log_var_recon=0.0
#echo "ETTh2 $denoise_layers_num "
#
#python -u run.py \
#     --task_name pretrain \
#    --root_path ./datasets/ETT-small/ \
#    --data_path ETTh2.csv \
#    --model_id ETTh2 \
#    --model TimeDART \
#    --data ETTh2 \
#    --features M \
#    --input_len 96 \
#    --e_layers 2 \
#    --d_layers 1 \
#    --enc_in 7 \
#    --dec_in 7 \
#    --c_out 7 \
#    --n_heads 8 \
#    --d_model 8 \
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
#    --train_epochs 50 \
#    --del_orth_loss $del_orth_loss \
#    --mask_ratio $mask_ratio \
#    --use_geo_mask $use_geo_mask \
#    --pretrain_mode $pretrain_mode \
#    --predict_eps $predict_eps \
#    --del_season_freq_loss $del_season_freq_loss \
#    --del_smoothness_loss $del_smoothness_loss \
#    --del_freq_loss $del_freq_loss \
#    --del_recon_loss $del_recon_loss \
#    --log_var_recon $log_var_recon \
#    --log_var_freq $log_var_freq \
#    --log_var_orth $log_var_orth \
#    --log_var_smooth $log_var_smooth \
#    --log_var_season_freq $log_var_season_freq \
#    --use_init_loss $use_init_loss \
#    --use_new_decomp $use_new_decomp \
#    --use_loss_compute $use_loss_compute \
#    --use_denoise $use_denoise \
#    --use_inner_new_decomp $use_inner_new_decomp \
#    --use_defire_noise $use_defire_noise \
#    --use_positional_encoding $use_positional_encoding \
#    --use_sostoken $use_sostoken \
#    --use_inner_encoder $use_inner_encoder \
#    --down_sampling_window 2 \
#> ./run_log/log_20250928150909_win/ETTh2/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1
#
#
#d_model=32
#d_model=8
#n_heads=8
#
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
#        --input_len 96 \
#        --label_len 48 \
#        --pred_len $pred_len \
#        --e_layers 2 \
#        --enc_in 7 \
#        --dec_in 7 \
#        --c_out 7 \
#        --denoise_layers_num $denoise_layers_num \
#        --n_heads $n_heads \
#        --d_model $d_model \
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
#    --mask_ratio $mask_ratio \
#    --use_geo_mask $use_geo_mask \
#    --pretrain_mode $pretrain_mode \
#    --predict_eps $predict_eps \
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
#        --use_init_loss $use_init_loss \
#    --use_new_decomp $use_new_decomp \
#    --use_loss_compute $use_loss_compute \
#    --use_denoise $use_denoise \
#    --use_inner_new_decomp $use_inner_new_decomp \
#    --use_defire_noise $use_defire_noise \
#    --use_positional_encoding $use_positional_encoding \
#    --use_sostoken $use_sostoken \
#    --use_inner_encoder $use_inner_encoder \
#        --down_sampling_window 2 \
#> ./run_log/log_20250928150909_win/ETTh2/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1
#
#
#done
#
#
#done
#
#
#
#done



use_geo_mask=1
use_geo_mask=1
pretrain_mode=mask
predict_eps=0


del_orth_loss=1
del_season_freq_loss=1
del_freq_loss=1
del_smoothness_loss=1
del_recon_loss=1

for mask_ratio in 0.7;do

use_init_loss=1



# h1m1 01000



log_var_freq=0.3
log_var_orth=0.01
log_var_smooth=0.0
log_var_season_freq=0.05
log_var_recon=0.0

layer_h1=1










#pretrain_mode=noise










for denoise_layers_num in 1;do

echo "ETTh1 $denoise_layers_num"
python -u run.py \
    --task_name pretrain \
    --root_path ./datasets/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1 \
    --model TimeDART \
    --data ETTh1 \
    --features M \
    --input_len 96 \
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
    --mask_ratio $mask_ratio \
    --use_geo_mask $use_geo_mask \
    --pretrain_mode $pretrain_mode \
    --predict_eps $predict_eps \
    --del_season_freq_loss $del_season_freq_loss \
    --del_smoothness_loss $del_smoothness_loss \
    --del_freq_loss $del_freq_loss \
    --del_recon_loss $del_recon_loss \
    --log_var_recon $log_var_recon \
    --log_var_freq $log_var_freq \
    --log_var_orth $log_var_orth \
    --log_var_smooth $log_var_smooth \
    --log_var_season_freq $log_var_season_freq \
    --use_init_loss $use_init_loss \
    --use_new_decomp $use_new_decomp \
    --use_loss_compute $use_loss_compute \
    --use_denoise $use_denoise \
    --use_inner_new_decomp $use_inner_new_decomp \
    --use_defire_noise $use_defire_noise \
    --use_positional_encoding $use_positional_encoding \
    --use_sostoken $use_sostoken \
    --use_inner_encoder $use_inner_encoder \
    --down_sampling_window 2 \
> ./run_log/log_20250928150909_win/ETTh1/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1
d_model=32
n_heads=16

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
        --input_len 96 \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 2 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --n_heads $n_heads \
        --d_model $d_model \
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
    --mask_ratio $mask_ratio \
    --use_geo_mask $use_geo_mask \
    --pretrain_mode $pretrain_mode \
    --predict_eps $predict_eps \
        --del_season_freq_loss $del_season_freq_loss \
        --del_smoothness_loss $del_smoothness_loss \
        --del_recon_loss $del_recon_loss \
        --log_var_recon $log_var_recon \
        --del_freq_loss $del_freq_loss \
        --log_var_freq $log_var_freq \
        --log_var_orth $log_var_orth \
        --log_var_smooth $log_var_smooth \
        --log_var_season_freq $log_var_season_freq \
        --use_init_loss $use_init_loss \
    --use_new_decomp $use_new_decomp \
    --use_loss_compute $use_loss_compute \
    --use_denoise $use_denoise \
    --use_inner_new_decomp $use_inner_new_decomp \
    --use_defire_noise $use_defire_noise \
    --use_positional_encoding $use_positional_encoding \
    --use_sostoken $use_sostoken \
    --use_inner_encoder $use_inner_encoder \
        --down_sampling_window 2 \
> ./run_log/log_20250928150909_win/ETTh1/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1

done

done


















layer_m1=1

for denoise_layers_num in 1;do








log_var_freq=0.3
log_var_orth=0.01
log_var_smooth=0.0
log_var_season_freq=0.05
log_var_recon=0.0
echo "ETTm1 $denoise_layers_num "

python -u run.py \
    --task_name pretrain \
    --root_path ./datasets/ETT-small/ \
    --data_path ETTm1.csv \
    --model_id ETTm1 \
    --model TimeDART \
    --data ETTm1 \
    --features M \
    --input_len 96 \
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
    --mask_ratio $mask_ratio \
    --use_geo_mask $use_geo_mask \
    --pretrain_mode $pretrain_mode \
    --predict_eps $predict_eps \
    --del_season_freq_loss $del_season_freq_loss \
    --del_smoothness_loss $del_smoothness_loss \
    --del_freq_loss $del_freq_loss \
    --del_recon_loss $del_recon_loss \
    --log_var_recon $log_var_recon \
    --log_var_freq $log_var_freq \
    --log_var_orth $log_var_orth \
    --log_var_smooth $log_var_smooth \
    --log_var_season_freq $log_var_season_freq \
    --use_init_loss $use_init_loss \
    --use_new_decomp $use_new_decomp \
    --use_loss_compute $use_loss_compute \
    --use_denoise $use_denoise \
    --use_inner_new_decomp $use_inner_new_decomp \
    --use_defire_noise $use_defire_noise \
    --use_positional_encoding $use_positional_encoding \
    --use_sostoken $use_sostoken \
    --use_inner_encoder $use_inner_encoder \
    --down_sampling_window 2 \
> ./run_log/log_20250928150909_win/ETTm1/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1

d_model=128 #  老
d_model=16
d_model=32
n_heads=8

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
        --input_len 96 \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 2 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --denoise_layers_num $denoise_layers_num \
        --n_heads  $n_heads \
        --d_model $d_model \
        --d_ff 64 \
        --patch_len 2 \
        --stride 2 \
        --dropout 0.1 \
        --head_dropout 0.0 \
        --batch_size 64 \
        --lr_decay 0.5 \
        --lradj step \
        --time_steps 1000 \
        --scheduler cosine \
        --del_orth_loss $del_orth_loss \
    --mask_ratio $mask_ratio \
    --use_geo_mask $use_geo_mask \
    --pretrain_mode $pretrain_mode \
    --predict_eps $predict_eps \
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
       --use_init_loss $use_init_loss \
    --use_new_decomp $use_new_decomp \
    --use_loss_compute $use_loss_compute \
    --use_denoise $use_denoise \
    --use_inner_new_decomp $use_inner_new_decomp \
    --use_defire_noise $use_defire_noise \
    --use_positional_encoding $use_positional_encoding \
    --use_sostoken $use_sostoken \
    --use_inner_encoder $use_inner_encoder \
       --down_sampling_window 2 \
> ./run_log/log_20250928150909_win/ETTm1/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1

done
done


#------------------------
#del_orth_loss=0
#del_season_freq_loss=1
#del_smoothness_loss=1
#del_freq_loss=1
#del_recon_loss=1

layer_h2=2
for denoise_layers_num in 1;do





log_var_freq=0.3
log_var_orth=0.01
log_var_smooth=0.0
log_var_season_freq=0.05
log_var_recon=0.0
echo "ETTh2 $denoise_layers_num "

python -u run.py \
     --task_name pretrain \
    --root_path ./datasets/ETT-small/ \
    --data_path ETTh2.csv \
    --model_id ETTh2 \
    --model TimeDART \
    --data ETTh2 \
    --features M \
    --input_len 96 \
    --e_layers 2 \
    --d_layers 1 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --n_heads 8 \
    --d_model 8 \
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
    --mask_ratio $mask_ratio \
    --use_geo_mask $use_geo_mask \
    --pretrain_mode $pretrain_mode \
    --predict_eps $predict_eps \
    --del_season_freq_loss $del_season_freq_loss \
    --del_smoothness_loss $del_smoothness_loss \
    --del_freq_loss $del_freq_loss \
    --del_recon_loss $del_recon_loss \
    --log_var_recon $log_var_recon \
    --log_var_freq $log_var_freq \
    --log_var_orth $log_var_orth \
    --log_var_smooth $log_var_smooth \
    --log_var_season_freq $log_var_season_freq \
    --use_init_loss $use_init_loss \
    --use_new_decomp $use_new_decomp \
    --use_loss_compute $use_loss_compute \
    --use_denoise $use_denoise \
    --use_inner_new_decomp $use_inner_new_decomp \
    --use_defire_noise $use_defire_noise \
    --use_positional_encoding $use_positional_encoding \
    --use_sostoken $use_sostoken \
    --use_inner_encoder $use_inner_encoder \
    --down_sampling_window 2 \
> ./run_log/log_20250928150909_win/ETTh2/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1


d_model=32
d_model=8
n_heads=8

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
        --input_len 96 \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 2 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --denoise_layers_num $denoise_layers_num \
        --n_heads $n_heads \
        --d_model $d_model \
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
    --mask_ratio $mask_ratio \
    --use_geo_mask $use_geo_mask \
    --pretrain_mode $pretrain_mode \
    --predict_eps $predict_eps \
        --del_season_freq_loss $del_season_freq_loss \
        --del_smoothness_loss $del_smoothness_loss \
        --del_freq_loss $del_freq_loss \
        --del_recon_loss $del_recon_loss \
        --log_var_recon $log_var_recon \
        --log_var_freq $log_var_freq \
        --log_var_orth $log_var_orth \
        --log_var_smooth $log_var_smooth \
        --log_var_season_freq $log_var_season_freq \
        --learning_rate 0.0001 \
        --pct_start 0.3 \
        --use_init_loss $use_init_loss \
    --use_new_decomp $use_new_decomp \
    --use_loss_compute $use_loss_compute \
    --use_denoise $use_denoise \
    --use_inner_new_decomp $use_inner_new_decomp \
    --use_defire_noise $use_defire_noise \
    --use_positional_encoding $use_positional_encoding \
    --use_sostoken $use_sostoken \
    --use_inner_encoder $use_inner_encoder \
        --down_sampling_window 2 \
> ./run_log/log_20250928150909_win/ETTh2/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1


done


done



done
