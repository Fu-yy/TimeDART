if [ ! -d "./run_log" ]; then
    mkdir ./run_log
fi
if [ ! -d "./run_log/log_202603251055256_win" ]; then
    mkdir ./run_log/log_202603251055256_win
fi
if [ ! -d "./run_log/log_202603251055256_win/ETTm1" ]; then
    mkdir ./run_log/log_202603251055256_win/ETTm1
fi
if [ ! -d "./run_log/log_202603251055256_win/ETTh1" ]; then
    mkdir ./run_log/log_202603251055256_win/ETTh1
fi
if [ ! -d "./run_log/log_202603251055256_win/ETTm2" ]; then
    mkdir ./run_log/log_202603251055256_win/ETTm2
fi

if [ ! -d "./run_log/log_202603251055256_win/ETTh2" ]; then
    mkdir ./run_log/log_202603251055256_win/ETTh2
fi
if [ ! -d "./run_log/log_202603251055256_win/electricity" ]; then
    mkdir ./run_log/log_202603251055256_win/electricity
fi

if [ ! -d "./run_log/log_202603251055256_win/Exchange" ]; then
    mkdir ./run_log/log_202603251055256_win/Exchange
fi

#if [ ! -d "./run_log/log_202603251055256_win/Solar" ]; then
#    mkdir ./run_log/log_202603251055256_win/Solar
#fi

if [ ! -d "./run_log/log_202603251055256_win/weather" ]; then
    mkdir ./run_log/log_202603251055256_win/weather
fi

if [ ! -d "./run_log/log_202603251055256_win/Traffic" ]; then
    mkdir ./run_log/log_202603251055256_win/Traffic
fi
#
#if [ ! -d "./run_log/log_202603251055256_win/PEMS03" ]; then
#    mkdir ./run_log/log_202603251055256_win/PEMS03
#fi
#
#if [ ! -d "./run_log/log_202603251055256_win/PEMS04" ]; then
#    mkdir ./run_log/log_202603251055256_win/PEMS04
#fi
#
#if [ ! -d "./run_log/log_202603251055256_win/PEMS07" ]; then
#    mkdir ./run_log/log_202603251055256_win/PEMS07
#fi
#if [ ! -d "./run_log/log_202603251055256_win/PEMS08" ]; then
#    mkdir ./run_log/log_202603251055256_win/PEMS08
#fi

#-------------------------------------------------

#----------------------------------------------- version


#
use_defire_noise=0
use_inner_encoder=0
use_positional_encoding=1
use_sostoken=1




use_loss_compute=1

use_denoise=0
use_inner_new_decomp=0







# 测试seed以及traffic
use_pretrain_encoder=1
use_finetune_encoder=1
destroy_mode=season
echo "use_new_decomp=1）"
use_new_decomp=1
use_pretrain_in_ft=0
film_mode=full


del_orth_loss=1
del_season_freq_loss=1
del_freq_loss=1
del_smoothness_loss=1
del_recon_loss=1


use_init_loss=1





#-----------------traffic--noise

use_geo_mask=1
use_geo_mask=0
predict_eps=1
pretrain_mode=noise
mask_ratio=0.9
layer_traffic=1
for denoise_layers_num in 1;do



#
n_heads=16
d_model=128
batch_size=8

echo "Traffic $denoise_layers_num "


python -u run.py \
    --task_name pretrain \
    --root_path ./datasets/traffic/ \
    --data_path traffic.csv \
    --model_id Traffic \
    --model TimeDART \
    --data Traffic \
    --features M \
    --input_len 96 \
    --e_layers 2 \
    --d_layers 1 \
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862 \
    --n_heads $n_heads \
    --d_model $d_model \
    --d_ff 256 \
    --patch_len 8 \
    --stride 8 \
    --head_dropout 0.1 \
    --dropout 0.2 \
    --time_steps 1000 \
    --denoise_layers_num $denoise_layers_num \
    --use_pretrain_encoder $use_pretrain_encoder \
    --use_finetune_encoder $use_finetune_encoder \
    --use_pretrain_in_ft $use_pretrain_in_ft \
    --scheduler cosine \
    --lr_decay 0.95 \
    --learning_rate 0.0001 \
    --del_orth_loss $del_orth_loss \
    --mask_ratio $mask_ratio \
    --use_geo_mask $use_geo_mask \
    --pretrain_mode $pretrain_mode \
    --predict_eps $predict_eps \
    --del_season_freq_loss $del_season_freq_loss \
    --del_smoothness_loss $del_smoothness_loss \
    --del_freq_loss $del_freq_loss \
    --del_recon_loss $del_recon_loss \
    --batch_size $batch_size \
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
    --train_epochs 50 \
> ./run_log/log_202603251055256_win/Traffic/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1

n_heads=16
d_model=128
batch_size=8
#batch_size=8
#d_model=64

for pred_len in 96 192 336 720; do
echo "Traffic $denoise_layers_num _ $pred_len"
    python -u run.py \
        --task_name finetune \
        --is_training 1 \
        --root_path ./datasets/traffic/ \
        --data_path traffic.csv \
        --model_id Traffic \
        --model TimeDART \
        --data Traffic \
        --features M \
        --input_len 96 \
        --label_len 48 \
        --denoise_layers_num $denoise_layers_num \
    --use_pretrain_encoder $use_pretrain_encoder \
    --use_finetune_encoder $use_finetune_encoder \
    --use_pretrain_in_ft $use_pretrain_in_ft \
        --pred_len $pred_len \
        --e_layers 2 \
        --enc_in 862 \
        --dec_in 862 \
        --c_out 862 \
        --n_heads $n_heads \
        --d_model $d_model \
        --d_ff 256 \
        --patch_len 8 \
        --stride 8 \
        --dropout 0.2 \
        --head_dropout 0.0 \
        --batch_size $batch_size \
        --lr_decay 0.5 \
        --lradj step \
        --time_steps 1000 \
        --del_orth_loss $del_orth_loss \
    --mask_ratio $mask_ratio \
    --use_geo_mask $use_geo_mask \
    --pretrain_mode $pretrain_mode \
    --predict_eps $predict_eps \
        --del_season_freq_loss $del_season_freq_loss \
        --del_smoothness_loss $del_smoothness_loss \
        --del_freq_loss $del_freq_loss \
        --del_recon_loss $del_recon_loss \
        --scheduler cosine \
        --patience 3 \
        --learning_rate 0.0004 \
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
        --pct_start 0.3 \
> ./run_log/log_202603251055256_win/Traffic/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1


done
done



#-----------------traffic--mask

use_geo_mask=1
use_geo_mask=0
predict_eps=0
pretrain_mode=mask
mask_ratio=0.5
layer_traffic=1
for denoise_layers_num in 1;do



#
n_heads=16
d_model=128
batch_size=8

echo "Traffic $denoise_layers_num "


python -u run.py \
    --task_name pretrain \
    --root_path ./datasets/traffic/ \
    --data_path traffic.csv \
    --model_id Traffic \
    --model TimeDART \
    --data Traffic \
    --features M \
    --input_len 96 \
    --e_layers 2 \
    --d_layers 1 \
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862 \
    --n_heads $n_heads \
    --d_model $d_model \
    --d_ff 256 \
    --patch_len 8 \
    --stride 8 \
    --head_dropout 0.1 \
    --dropout 0.2 \
    --time_steps 1000 \
    --denoise_layers_num $denoise_layers_num \
    --use_pretrain_encoder $use_pretrain_encoder \
    --use_finetune_encoder $use_finetune_encoder \
    --use_pretrain_in_ft $use_pretrain_in_ft \
    --scheduler cosine \
    --lr_decay 0.95 \
    --learning_rate 0.0001 \
    --del_orth_loss $del_orth_loss \
    --mask_ratio $mask_ratio \
    --use_geo_mask $use_geo_mask \
    --pretrain_mode $pretrain_mode \
    --predict_eps $predict_eps \
    --del_season_freq_loss $del_season_freq_loss \
    --del_smoothness_loss $del_smoothness_loss \
    --del_freq_loss $del_freq_loss \
    --del_recon_loss $del_recon_loss \
    --batch_size $batch_size \
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
    --train_epochs 50 \
> ./run_log/log_202603251055256_win/Traffic/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1

n_heads=16
d_model=128
batch_size=8
#batch_size=8
#d_model=64

for pred_len in 96 192 336 720; do
echo "Traffic $denoise_layers_num _ $pred_len"
    python -u run.py \
        --task_name finetune \
        --is_training 1 \
        --root_path ./datasets/traffic/ \
        --data_path traffic.csv \
        --model_id Traffic \
        --model TimeDART \
        --data Traffic \
        --features M \
        --input_len 96 \
        --label_len 48 \
        --denoise_layers_num $denoise_layers_num \
    --use_pretrain_encoder $use_pretrain_encoder \
    --use_finetune_encoder $use_finetune_encoder \
    --use_pretrain_in_ft $use_pretrain_in_ft \
        --pred_len $pred_len \
        --e_layers 2 \
        --enc_in 862 \
        --dec_in 862 \
        --c_out 862 \
        --n_heads $n_heads \
        --d_model $d_model \
        --d_ff 256 \
        --patch_len 8 \
        --stride 8 \
        --dropout 0.2 \
        --head_dropout 0.0 \
        --batch_size $batch_size \
        --lr_decay 0.5 \
        --lradj step \
        --time_steps 1000 \
        --del_orth_loss $del_orth_loss \
    --mask_ratio $mask_ratio \
    --use_geo_mask $use_geo_mask \
    --pretrain_mode $pretrain_mode \
    --predict_eps $predict_eps \
        --del_season_freq_loss $del_season_freq_loss \
        --del_smoothness_loss $del_smoothness_loss \
        --del_freq_loss $del_freq_loss \
        --del_recon_loss $del_recon_loss \
        --scheduler cosine \
        --patience 3 \
        --learning_rate 0.0004 \
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
        --pct_start 0.3 \
> ./run_log/log_202603251055256_win/Traffic/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1


done
done


#-----------------traffic--mask

use_geo_mask=1
use_geo_mask=0
predict_eps=0
pretrain_mode=mask
mask_ratio=0.7
layer_traffic=1
for denoise_layers_num in 1;do



#
n_heads=16
d_model=128
batch_size=8

echo "Traffic $denoise_layers_num "


python -u run.py \
    --task_name pretrain \
    --root_path ./datasets/traffic/ \
    --data_path traffic.csv \
    --model_id Traffic \
    --model TimeDART \
    --data Traffic \
    --features M \
    --input_len 96 \
    --e_layers 2 \
    --d_layers 1 \
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862 \
    --n_heads $n_heads \
    --d_model $d_model \
    --d_ff 256 \
    --patch_len 8 \
    --stride 8 \
    --head_dropout 0.1 \
    --dropout 0.2 \
    --time_steps 1000 \
    --denoise_layers_num $denoise_layers_num \
    --use_pretrain_encoder $use_pretrain_encoder \
    --use_finetune_encoder $use_finetune_encoder \
    --use_pretrain_in_ft $use_pretrain_in_ft \
    --scheduler cosine \
    --lr_decay 0.95 \
    --learning_rate 0.0001 \
    --del_orth_loss $del_orth_loss \
    --mask_ratio $mask_ratio \
    --use_geo_mask $use_geo_mask \
    --pretrain_mode $pretrain_mode \
    --predict_eps $predict_eps \
    --del_season_freq_loss $del_season_freq_loss \
    --del_smoothness_loss $del_smoothness_loss \
    --del_freq_loss $del_freq_loss \
    --del_recon_loss $del_recon_loss \
    --batch_size $batch_size \
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
    --train_epochs 50 \
> ./run_log/log_202603251055256_win/Traffic/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1

n_heads=16
d_model=128
batch_size=8
#batch_size=8
#d_model=64

for pred_len in 96 192 336 720; do
echo "Traffic $denoise_layers_num _ $pred_len"
    python -u run.py \
        --task_name finetune \
        --is_training 1 \
        --root_path ./datasets/traffic/ \
        --data_path traffic.csv \
        --model_id Traffic \
        --model TimeDART \
        --data Traffic \
        --features M \
        --input_len 96 \
        --label_len 48 \
        --denoise_layers_num $denoise_layers_num \
    --use_pretrain_encoder $use_pretrain_encoder \
    --use_finetune_encoder $use_finetune_encoder \
    --use_pretrain_in_ft $use_pretrain_in_ft \
        --pred_len $pred_len \
        --e_layers 2 \
        --enc_in 862 \
        --dec_in 862 \
        --c_out 862 \
        --n_heads $n_heads \
        --d_model $d_model \
        --d_ff 256 \
        --patch_len 8 \
        --stride 8 \
        --dropout 0.2 \
        --head_dropout 0.0 \
        --batch_size $batch_size \
        --lr_decay 0.5 \
        --lradj step \
        --time_steps 1000 \
        --del_orth_loss $del_orth_loss \
    --mask_ratio $mask_ratio \
    --use_geo_mask $use_geo_mask \
    --pretrain_mode $pretrain_mode \
    --predict_eps $predict_eps \
        --del_season_freq_loss $del_season_freq_loss \
        --del_smoothness_loss $del_smoothness_loss \
        --del_freq_loss $del_freq_loss \
        --del_recon_loss $del_recon_loss \
        --scheduler cosine \
        --patience 3 \
        --learning_rate 0.0004 \
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
        --pct_start 0.3 \
> ./run_log/log_202603251055256_win/Traffic/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1


done
done


#-----------------traffic--mask

use_geo_mask=1
use_geo_mask=0
predict_eps=0
pretrain_mode=mask
mask_ratio=0.9
layer_traffic=1
for denoise_layers_num in 1;do



#
n_heads=16
d_model=128
batch_size=8

echo "Traffic $denoise_layers_num "


python -u run.py \
    --task_name pretrain \
    --root_path ./datasets/traffic/ \
    --data_path traffic.csv \
    --model_id Traffic \
    --model TimeDART \
    --data Traffic \
    --features M \
    --input_len 96 \
    --e_layers 2 \
    --d_layers 1 \
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862 \
    --n_heads $n_heads \
    --d_model $d_model \
    --d_ff 256 \
    --patch_len 8 \
    --stride 8 \
    --head_dropout 0.1 \
    --dropout 0.2 \
    --time_steps 1000 \
    --denoise_layers_num $denoise_layers_num \
    --use_pretrain_encoder $use_pretrain_encoder \
    --use_finetune_encoder $use_finetune_encoder \
    --use_pretrain_in_ft $use_pretrain_in_ft \
    --scheduler cosine \
    --lr_decay 0.95 \
    --learning_rate 0.0001 \
    --del_orth_loss $del_orth_loss \
    --mask_ratio $mask_ratio \
    --use_geo_mask $use_geo_mask \
    --pretrain_mode $pretrain_mode \
    --predict_eps $predict_eps \
    --del_season_freq_loss $del_season_freq_loss \
    --del_smoothness_loss $del_smoothness_loss \
    --del_freq_loss $del_freq_loss \
    --del_recon_loss $del_recon_loss \
    --batch_size $batch_size \
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
    --train_epochs 50 \
> ./run_log/log_202603251055256_win/Traffic/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1

n_heads=16
d_model=128
batch_size=8
#batch_size=8
#d_model=64

for pred_len in 96 192 336 720; do
echo "Traffic $denoise_layers_num _ $pred_len"
    python -u run.py \
        --task_name finetune \
        --is_training 1 \
        --root_path ./datasets/traffic/ \
        --data_path traffic.csv \
        --model_id Traffic \
        --model TimeDART \
        --data Traffic \
        --features M \
        --input_len 96 \
        --label_len 48 \
        --denoise_layers_num $denoise_layers_num \
    --use_pretrain_encoder $use_pretrain_encoder \
    --use_finetune_encoder $use_finetune_encoder \
    --use_pretrain_in_ft $use_pretrain_in_ft \
        --pred_len $pred_len \
        --e_layers 2 \
        --enc_in 862 \
        --dec_in 862 \
        --c_out 862 \
        --n_heads $n_heads \
        --d_model $d_model \
        --d_ff 256 \
        --patch_len 8 \
        --stride 8 \
        --dropout 0.2 \
        --head_dropout 0.0 \
        --batch_size $batch_size \
        --lr_decay 0.5 \
        --lradj step \
        --time_steps 1000 \
        --del_orth_loss $del_orth_loss \
    --mask_ratio $mask_ratio \
    --use_geo_mask $use_geo_mask \
    --pretrain_mode $pretrain_mode \
    --predict_eps $predict_eps \
        --del_season_freq_loss $del_season_freq_loss \
        --del_smoothness_loss $del_smoothness_loss \
        --del_freq_loss $del_freq_loss \
        --del_recon_loss $del_recon_loss \
        --scheduler cosine \
        --patience 3 \
        --learning_rate 0.0004 \
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
        --pct_start 0.3 \
> ./run_log/log_202603251055256_win/Traffic/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1


done
done



python /root/autodl-tmp/TimeDART/send_email.py


shutdown -h now


