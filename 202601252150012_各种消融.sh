if [ ! -d "./run_log" ]; then
    mkdir ./run_log
fi
if [ ! -d "./run_log/log_202601252150012_win" ]; then
    mkdir ./run_log/log_202601252150012_win
fi
if [ ! -d "./run_log/log_202601252150012_win/ETTm1" ]; then
    mkdir ./run_log/log_202601252150012_win/ETTm1
fi
if [ ! -d "./run_log/log_202601252150012_win/ETTh1" ]; then
    mkdir ./run_log/log_202601252150012_win/ETTh1
fi
if [ ! -d "./run_log/log_202601252150012_win/ETTm2" ]; then
    mkdir ./run_log/log_202601252150012_win/ETTm2
fi

if [ ! -d "./run_log/log_202601252150012_win/ETTh2" ]; then
    mkdir ./run_log/log_202601252150012_win/ETTh2
fi
if [ ! -d "./run_log/log_202601252150012_win/electricity" ]; then
    mkdir ./run_log/log_202601252150012_win/electricity
fi

if [ ! -d "./run_log/log_202601252150012_win/Exchange" ]; then
    mkdir ./run_log/log_202601252150012_win/Exchange
fi

#if [ ! -d "./run_log/log_202601252150012_win/Solar" ]; then
#    mkdir ./run_log/log_202601252150012_win/Solar
#fi

if [ ! -d "./run_log/log_202601252150012_win/weather" ]; then
    mkdir ./run_log/log_202601252150012_win/weather
fi

if [ ! -d "./run_log/log_202601252150012_win/Traffic" ]; then
    mkdir ./run_log/log_202601252150012_win/Traffic
fi
#
#if [ ! -d "./run_log/log_202601252150012_win/PEMS03" ]; then
#    mkdir ./run_log/log_202601252150012_win/PEMS03
#fi
#
#if [ ! -d "./run_log/log_202601252150012_win/PEMS04" ]; then
#    mkdir ./run_log/log_202601252150012_win/PEMS04
#fi
#
#if [ ! -d "./run_log/log_202601252150012_win/PEMS07" ]; then
#    mkdir ./run_log/log_202601252150012_win/PEMS07
#fi
#if [ ! -d "./run_log/log_202601252150012_win/PEMS08" ]; then
#    mkdir ./run_log/log_202601252150012_win/PEMS08
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


if false;then
use_pretrain_encoder=0
use_finetune_encoder=0
destroy_mode=season
echo "预训练和微调的encoder消融"
use_new_decomp=1
use_pretrain_in_ft=0
film_mode=full
for _ in 1;do

del_orth_loss=1
del_season_freq_loss=1
del_freq_loss=1
del_smoothness_loss=1
del_recon_loss=1


use_init_loss=1





layer_h1=1
# ETTh1 noise
use_geo_mask=1
use_geo_mask=0
predict_eps=1
pretrain_mode=noise
mask_ratio=0.41
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
    --use_pretrain_encoder $use_pretrain_encoder \
    --use_finetune_encoder $use_finetune_encoder \
    --use_pretrain_in_ft $use_pretrain_in_ft \
    --film_mode $film_mode \
    --patch_len 2 \
    --stride 2 \
    --head_dropout 0.1 \
    --dropout 0.2 \
    --time_steps 1000 \
    --scheduler cosine \
    --lr_decay 0.9 \
    --learning_rate 0.0001 \
    --batch_size 16 \
    --train_epochs 20 \
    --del_orth_loss $del_orth_loss \
    --mask_ratio $mask_ratio \
    --use_geo_mask $use_geo_mask \
    --pretrain_mode $pretrain_mode \
    --predict_eps $predict_eps \
    --del_season_freq_loss $del_season_freq_loss \
    --del_smoothness_loss $del_smoothness_loss \
    --del_freq_loss $del_freq_loss \
    --del_recon_loss $del_recon_loss \
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
> ./run_log/log_202601252150012_win/ETTh1/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1
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
    --use_pretrain_encoder $use_pretrain_encoder \
    --use_finetune_encoder $use_finetune_encoder \
    --use_pretrain_in_ft $use_pretrain_in_ft \
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
        --del_freq_loss $del_freq_loss \
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
> ./run_log/log_202601252150012_win/ETTh1/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1

done

done


# ETTh2 mask
layer_h2=2
use_geo_mask=1
use_geo_mask=0
predict_eps=0
pretrain_mode=noise
mask_ratio=0.7
for denoise_layers_num in 1;do




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
    --use_pretrain_encoder $use_pretrain_encoder \
    --use_finetune_encoder $use_finetune_encoder \
    --use_pretrain_in_ft $use_pretrain_in_ft \
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
    --train_epochs 20 \
    --del_orth_loss $del_orth_loss \
    --mask_ratio $mask_ratio \
    --use_geo_mask $use_geo_mask \
    --pretrain_mode $pretrain_mode \
    --predict_eps $predict_eps \
    --del_season_freq_loss $del_season_freq_loss \
    --del_smoothness_loss $del_smoothness_loss \
    --del_freq_loss $del_freq_loss \
    --del_recon_loss $del_recon_loss \
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
> ./run_log/log_202601252150012_win/ETTh2/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1


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
    --use_pretrain_encoder $use_pretrain_encoder \
    --use_finetune_encoder $use_finetune_encoder \
    --use_pretrain_in_ft $use_pretrain_in_ft \
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
> ./run_log/log_202601252150012_win/ETTh2/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1


done


done




layer_m1=1
# ETTM1 noise
use_geo_mask=1
use_geo_mask=0
predict_eps=1
pretrain_mode=noise

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
    --use_pretrain_encoder $use_pretrain_encoder \
    --use_finetune_encoder $use_finetune_encoder \
    --use_pretrain_in_ft $use_pretrain_in_ft \
    --patch_len 2 \
    --stride 2 \
    --head_dropout 0.1 \
    --dropout 0.2 \
    --time_steps 1000 \
    --scheduler cosine \
    --lr_decay 0.95 \
    --learning_rate 0.0001 \
    --batch_size 64 \
    --train_epochs 20 \
    --del_orth_loss $del_orth_loss \
    --mask_ratio $mask_ratio \
    --use_geo_mask $use_geo_mask \
    --pretrain_mode $pretrain_mode \
    --predict_eps $predict_eps \
    --del_season_freq_loss $del_season_freq_loss \
    --del_smoothness_loss $del_smoothness_loss \
    --del_freq_loss $del_freq_loss \
    --del_recon_loss $del_recon_loss \
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
> ./run_log/log_202601252150012_win/ETTm1/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1

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
    --use_pretrain_encoder $use_pretrain_encoder \
    --use_finetune_encoder $use_finetune_encoder \
    --use_pretrain_in_ft $use_pretrain_in_ft \
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
> ./run_log/log_202601252150012_win/ETTm1/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1

done
done


# ETTm2 mask
use_geo_mask=1
use_geo_mask=0
predict_eps=0
pretrain_mode=noise
mask_ratio=0.3
layer_m2=1
for denoise_layers_num in 1;do


# -------------- ETTm2 script
echo "ETTm2 $denoise_layers_num "

python -u run.py \
    --task_name pretrain \
    --root_path ./datasets/ETT-small/ \
    --data_path ETTm2.csv \
    --model_id ETTm2 \
    --model TimeDART \
    --data ETTm2 \
    --features M \
    --input_len 96 \
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
    --denoise_layers_num $denoise_layers_num \
    --use_pretrain_encoder $use_pretrain_encoder \
    --use_finetune_encoder $use_finetune_encoder \
    --use_pretrain_in_ft $use_pretrain_in_ft \
    --dropout 0.2 \
    --time_steps 1000 \
    --scheduler cosine \
    --lr_decay 0.8 \
    --learning_rate 0.001 \
    --batch_size 64 \
    --del_orth_loss $del_orth_loss \
    --mask_ratio $mask_ratio \
    --use_geo_mask $use_geo_mask \
    --pretrain_mode $pretrain_mode \
    --predict_eps $predict_eps \
    --del_season_freq_loss $del_season_freq_loss \
    --del_smoothness_loss $del_smoothness_loss \
    --del_recon_loss $del_recon_loss \
    --del_freq_loss $del_freq_loss \
    --train_epochs 20 \
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
> ./run_log/log_202601252150012_win/ETTm2/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1

d_model=8
n_heads=8 # 老
#n_heads=4

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
        --input_len 96 \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 2 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --n_heads $n_heads \
        --d_model $d_model \
        --d_ff 16 \
        --patch_len 2 \
        --stride 2 \
        --dropout 0.4 \
        --head_dropout 0.1 \
        --denoise_layers_num $denoise_layers_num \
    --use_pretrain_encoder $use_pretrain_encoder \
    --use_finetune_encoder $use_finetune_encoder \
    --use_pretrain_in_ft $use_pretrain_in_ft \
        --batch_size 64 \
        --lr_decay 0.5 \
        --lradj step \
        --time_steps 1000 \
        --scheduler cosine \
        --patience 3 \
        --learning_rate 0.0001 \
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
        --pct_start 0.2 \
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
> ./run_log/log_202601252150012_win/ETTm2/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1

done



done



#---------------- Exchange script
# Exchange mask
use_geo_mask=1
use_geo_mask=0
predict_eps=0
pretrain_mode=noise
mask_ratio=0.5
layer_exchange=1
for denoise_layers_num in 1 ;do


d_model=64
d_model=32
n_heads=8
batch_size=16

echo "Exchange $denoise_layers_num"

python -u run.py \
    --task_name pretrain \
    --root_path ./datasets/exchange_rate/ \
    --data_path exchange_rate.csv \
    --model_id Exchange \
    --model TimeDART \
    --data Exchange \
    --features M \
    --input_len 96 \
    --e_layers 2 \
    --d_layers 1 \
    --enc_in 8 \
    --dec_in 8 \
    --denoise_layers_num $denoise_layers_num \
    --use_pretrain_encoder $use_pretrain_encoder \
    --use_finetune_encoder $use_finetune_encoder \
    --use_pretrain_in_ft $use_pretrain_in_ft \
    --c_out 8 \
    --n_heads $n_heads \
    --d_model $d_model \
    --d_ff 64 \
    --patch_len 2 \
    --stride 2 \
    --head_dropout 0.1 \
    --dropout 0.2 \
    --time_steps 1000 \
    --scheduler cosine \
    --lr_decay 0.95 \
    --learning_rate 0.001 \
    --batch_size $batch_size \
    --train_epochs 20 \
    --del_orth_loss $del_orth_loss \
    --mask_ratio $mask_ratio \
    --use_geo_mask $use_geo_mask \
    --pretrain_mode $pretrain_mode \
    --predict_eps $predict_eps \
    --del_season_freq_loss $del_season_freq_loss \
    --del_smoothness_loss $del_smoothness_loss \
    --del_freq_loss $del_freq_loss \
    --del_recon_loss $del_recon_loss \
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
> ./run_log/log_202601252150012_win/Exchange/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1

#d_model=64
#n_heads=8
#batch_size=16

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
        --input_len 96 \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 2 \
        --denoise_layers_num $denoise_layers_num \
    --use_pretrain_encoder $use_pretrain_encoder \
    --use_finetune_encoder $use_finetune_encoder \
    --use_pretrain_in_ft $use_pretrain_in_ft \
        --enc_in 8 \
        --dec_in 8 \
        --c_out 8 \
        --n_heads $n_heads \
        --d_model $d_model \
        --d_ff 64 \
        --patch_len 2 \
        --stride 2 \
        --dropout 0.2 \
        --head_dropout 0.1 \
        --batch_size $batch_size \
        --lr_decay 0.8 \
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
> ./run_log/log_202601252150012_win/Exchange/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1

done
done



#----------------- WTH script
# WTH mask
use_geo_mask=1
use_geo_mask=0
predict_eps=0

#原始32dmodel 16batch
d_model=32
d_model=64
batch_size=16
layer_wth=1
pretrain_mode=noise
mask_ratio=0.7
for denoise_layers_num in 1;do

echo "Weather $denoise_layers_num "


python -u run.py \
    --task_name pretrain \
    --root_path ./datasets/weather/ \
    --data_path weather.csv \
    --model_id Weather \
    --model TimeDART \
    --data Weather \
    --features M \
    --input_len 96 \
    --e_layers 2 \
    --d_layers 1 \
    --enc_in 21 \
    --dec_in 21 \
    --denoise_layers_num $denoise_layers_num \
    --use_pretrain_encoder $use_pretrain_encoder \
    --use_finetune_encoder $use_finetune_encoder \
    --use_pretrain_in_ft $use_pretrain_in_ft \
    --c_out 21 \
    --n_heads 8 \
    --d_model $d_model \
    --d_ff 64 \
    --patch_len 2 \
    --stride 2 \
    --head_dropout 0.1 \
    --dropout 0.2 \
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
    --lr_decay 0.95 \
    --learning_rate 0.001 \
    --batch_size $batch_size \
    --train_epochs 20 \
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
> ./run_log/log_202601252150012_win/weather/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1



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
    --use_pretrain_encoder $use_pretrain_encoder \
    --use_finetune_encoder $use_finetune_encoder \
    --use_pretrain_in_ft $use_pretrain_in_ft \
        --features M \
        --input_len 96 \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 2 \
        --enc_in 21 \
        --dec_in 21 \
        --c_out 21 \
        --n_heads 8 \
        --d_model $d_model \
        --d_ff 64 \
        --patch_len 2 \
        --stride 2 \
        --dropout 0.2 \
        --head_dropout 0.1 \
        --batch_size $batch_size \
        --del_orth_loss $del_orth_loss \
    --mask_ratio $mask_ratio \
    --use_geo_mask $use_geo_mask \
    --pretrain_mode $pretrain_mode \
    --predict_eps $predict_eps \
        --del_season_freq_loss $del_season_freq_loss \
        --del_smoothness_loss $del_smoothness_loss \
        --del_freq_loss $del_freq_loss \
        --del_recon_loss $del_recon_loss \
        --lr_decay 0.5 \
        --lradj step \
        --time_steps 1000 \
        --scheduler cosine \
        --patience 3 \
        --learning_rate 0.0004 \
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
> ./run_log/log_202601252150012_win/weather/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1

done





done



#---------------- ELE script

# ELE mask
use_geo_mask=1
use_geo_mask=0
predict_eps=0
pretrain_mode=noise
mask_ratio=0.9
layer_ecl=1
for denoise_layers_num in 1;do



#
n_heads=16
d_model=128
batch_size=16

echo "ECL $denoise_layers_num "


python -u run.py \
    --task_name pretrain \
    --root_path ./datasets/electricity/ \
    --data_path electricity.csv \
    --model_id Electricity \
    --model TimeDART \
    --data Electricity \
    --features M \
    --input_len 96 \
    --e_layers 2 \
    --d_layers 1 \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
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
    --train_epochs 20 \
> ./run_log/log_202601252150012_win/electricity/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1

n_heads = 16
d_model=128
batch_size=16
#batch_size=8
#d_model=64

for pred_len in 96 192 336 720; do
echo "ECL $denoise_layers_num _ $pred_len"
    python -u run.py \
        --task_name finetune \
        --is_training 1 \
        --root_path ./datasets/electricity/ \
        --data_path electricity.csv \
        --model_id Electricity \
        --model TimeDART \
        --data Electricity \
        --features M \
        --input_len 96 \
        --label_len 48 \
        --denoise_layers_num $denoise_layers_num \
    --use_pretrain_encoder $use_pretrain_encoder \
    --use_finetune_encoder $use_finetune_encoder \
    --use_pretrain_in_ft $use_pretrain_in_ft \
        --pred_len $pred_len \
        --e_layers 2 \
        --enc_in 321 \
        --dec_in 321 \
        --c_out 321 \
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
> ./run_log/log_202601252150012_win/electricity/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1


done
done

done




use_pretrain_encoder=0
use_finetune_encoder=1
destroy_mode=season
echo "只用微调的encoder消融缺少预训练）"
use_new_decomp=1
use_pretrain_in_ft=0
film_mode=full
for _ in 1;do

del_orth_loss=1
del_season_freq_loss=1
del_freq_loss=1
del_smoothness_loss=1
del_recon_loss=1


use_init_loss=1





layer_h1=1
# ETTh1 noise
use_geo_mask=1
use_geo_mask=0
predict_eps=1
pretrain_mode=noise
mask_ratio=0.41
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
    --use_pretrain_encoder $use_pretrain_encoder \
    --use_finetune_encoder $use_finetune_encoder \
    --use_pretrain_in_ft $use_pretrain_in_ft \
    --film_mode $film_mode \
    --patch_len 2 \
    --stride 2 \
    --head_dropout 0.1 \
    --dropout 0.2 \
    --time_steps 1000 \
    --scheduler cosine \
    --lr_decay 0.9 \
    --learning_rate 0.0001 \
    --batch_size 16 \
    --train_epochs 20 \
    --del_orth_loss $del_orth_loss \
    --mask_ratio $mask_ratio \
    --use_geo_mask $use_geo_mask \
    --pretrain_mode $pretrain_mode \
    --predict_eps $predict_eps \
    --del_season_freq_loss $del_season_freq_loss \
    --del_smoothness_loss $del_smoothness_loss \
    --del_freq_loss $del_freq_loss \
    --del_recon_loss $del_recon_loss \
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
> ./run_log/log_202601252150012_win/ETTh1/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1
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
    --use_pretrain_encoder $use_pretrain_encoder \
    --use_finetune_encoder $use_finetune_encoder \
    --use_pretrain_in_ft $use_pretrain_in_ft \
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
        --del_freq_loss $del_freq_loss \
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
> ./run_log/log_202601252150012_win/ETTh1/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1

done

done


# ETTh2 mask
layer_h2=2
use_geo_mask=1
use_geo_mask=0
predict_eps=0
pretrain_mode=noise
mask_ratio=0.7
for denoise_layers_num in 1;do




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
    --use_pretrain_encoder $use_pretrain_encoder \
    --use_finetune_encoder $use_finetune_encoder \
    --use_pretrain_in_ft $use_pretrain_in_ft \
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
    --train_epochs 20 \
    --del_orth_loss $del_orth_loss \
    --mask_ratio $mask_ratio \
    --use_geo_mask $use_geo_mask \
    --pretrain_mode $pretrain_mode \
    --predict_eps $predict_eps \
    --del_season_freq_loss $del_season_freq_loss \
    --del_smoothness_loss $del_smoothness_loss \
    --del_freq_loss $del_freq_loss \
    --del_recon_loss $del_recon_loss \
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
> ./run_log/log_202601252150012_win/ETTh2/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1


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
    --use_pretrain_encoder $use_pretrain_encoder \
    --use_finetune_encoder $use_finetune_encoder \
    --use_pretrain_in_ft $use_pretrain_in_ft \
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
> ./run_log/log_202601252150012_win/ETTh2/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1


done


done




layer_m1=1
# ETTM1 noise
use_geo_mask=1
use_geo_mask=0
predict_eps=1
pretrain_mode=noise

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
    --use_pretrain_encoder $use_pretrain_encoder \
    --use_finetune_encoder $use_finetune_encoder \
    --use_pretrain_in_ft $use_pretrain_in_ft \
    --patch_len 2 \
    --stride 2 \
    --head_dropout 0.1 \
    --dropout 0.2 \
    --time_steps 1000 \
    --scheduler cosine \
    --lr_decay 0.95 \
    --learning_rate 0.0001 \
    --batch_size 64 \
    --train_epochs 20 \
    --del_orth_loss $del_orth_loss \
    --mask_ratio $mask_ratio \
    --use_geo_mask $use_geo_mask \
    --pretrain_mode $pretrain_mode \
    --predict_eps $predict_eps \
    --del_season_freq_loss $del_season_freq_loss \
    --del_smoothness_loss $del_smoothness_loss \
    --del_freq_loss $del_freq_loss \
    --del_recon_loss $del_recon_loss \
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
> ./run_log/log_202601252150012_win/ETTm1/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1

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
    --use_pretrain_encoder $use_pretrain_encoder \
    --use_finetune_encoder $use_finetune_encoder \
    --use_pretrain_in_ft $use_pretrain_in_ft \
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
> ./run_log/log_202601252150012_win/ETTm1/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1

done
done


# ETTm2 mask
use_geo_mask=1
use_geo_mask=0
predict_eps=0
pretrain_mode=noise
mask_ratio=0.3
layer_m2=1
for denoise_layers_num in 1;do


# -------------- ETTm2 script
echo "ETTm2 $denoise_layers_num "

python -u run.py \
    --task_name pretrain \
    --root_path ./datasets/ETT-small/ \
    --data_path ETTm2.csv \
    --model_id ETTm2 \
    --model TimeDART \
    --data ETTm2 \
    --features M \
    --input_len 96 \
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
    --denoise_layers_num $denoise_layers_num \
    --use_pretrain_encoder $use_pretrain_encoder \
    --use_finetune_encoder $use_finetune_encoder \
    --use_pretrain_in_ft $use_pretrain_in_ft \
    --dropout 0.2 \
    --time_steps 1000 \
    --scheduler cosine \
    --lr_decay 0.8 \
    --learning_rate 0.001 \
    --batch_size 64 \
    --del_orth_loss $del_orth_loss \
    --mask_ratio $mask_ratio \
    --use_geo_mask $use_geo_mask \
    --pretrain_mode $pretrain_mode \
    --predict_eps $predict_eps \
    --del_season_freq_loss $del_season_freq_loss \
    --del_smoothness_loss $del_smoothness_loss \
    --del_recon_loss $del_recon_loss \
    --del_freq_loss $del_freq_loss \
    --train_epochs 20 \
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
> ./run_log/log_202601252150012_win/ETTm2/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1

d_model=8
n_heads=8 # 老
#n_heads=4

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
        --input_len 96 \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 2 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --n_heads $n_heads \
        --d_model $d_model \
        --d_ff 16 \
        --patch_len 2 \
        --stride 2 \
        --dropout 0.4 \
        --head_dropout 0.1 \
        --denoise_layers_num $denoise_layers_num \
    --use_pretrain_encoder $use_pretrain_encoder \
    --use_finetune_encoder $use_finetune_encoder \
    --use_pretrain_in_ft $use_pretrain_in_ft \
        --batch_size 64 \
        --lr_decay 0.5 \
        --lradj step \
        --time_steps 1000 \
        --scheduler cosine \
        --patience 3 \
        --learning_rate 0.0001 \
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
        --pct_start 0.2 \
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
> ./run_log/log_202601252150012_win/ETTm2/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1

done



done



#---------------- Exchange script
# Exchange mask
use_geo_mask=1
use_geo_mask=0
predict_eps=0
pretrain_mode=noise
mask_ratio=0.5
layer_exchange=1
for denoise_layers_num in 1 ;do


d_model=64
d_model=32
n_heads=8
batch_size=16

echo "Exchange $denoise_layers_num"

python -u run.py \
    --task_name pretrain \
    --root_path ./datasets/exchange_rate/ \
    --data_path exchange_rate.csv \
    --model_id Exchange \
    --model TimeDART \
    --data Exchange \
    --features M \
    --input_len 96 \
    --e_layers 2 \
    --d_layers 1 \
    --enc_in 8 \
    --dec_in 8 \
    --denoise_layers_num $denoise_layers_num \
    --use_pretrain_encoder $use_pretrain_encoder \
    --use_finetune_encoder $use_finetune_encoder \
    --use_pretrain_in_ft $use_pretrain_in_ft \
    --c_out 8 \
    --n_heads $n_heads \
    --d_model $d_model \
    --d_ff 64 \
    --patch_len 2 \
    --stride 2 \
    --head_dropout 0.1 \
    --dropout 0.2 \
    --time_steps 1000 \
    --scheduler cosine \
    --lr_decay 0.95 \
    --learning_rate 0.001 \
    --batch_size $batch_size \
    --train_epochs 20 \
    --del_orth_loss $del_orth_loss \
    --mask_ratio $mask_ratio \
    --use_geo_mask $use_geo_mask \
    --pretrain_mode $pretrain_mode \
    --predict_eps $predict_eps \
    --del_season_freq_loss $del_season_freq_loss \
    --del_smoothness_loss $del_smoothness_loss \
    --del_freq_loss $del_freq_loss \
    --del_recon_loss $del_recon_loss \
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
> ./run_log/log_202601252150012_win/Exchange/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1

#d_model=64
#n_heads=8
#batch_size=16

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
        --input_len 96 \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 2 \
        --denoise_layers_num $denoise_layers_num \
    --use_pretrain_encoder $use_pretrain_encoder \
    --use_finetune_encoder $use_finetune_encoder \
    --use_pretrain_in_ft $use_pretrain_in_ft \
        --enc_in 8 \
        --dec_in 8 \
        --c_out 8 \
        --n_heads $n_heads \
        --d_model $d_model \
        --d_ff 64 \
        --patch_len 2 \
        --stride 2 \
        --dropout 0.2 \
        --head_dropout 0.1 \
        --batch_size $batch_size \
        --lr_decay 0.8 \
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
> ./run_log/log_202601252150012_win/Exchange/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1

done
done



#----------------- WTH script
# WTH mask
use_geo_mask=1
use_geo_mask=0
predict_eps=0

#原始32dmodel 16batch
d_model=32
d_model=64
batch_size=16
layer_wth=1
pretrain_mode=noise
mask_ratio=0.7
for denoise_layers_num in 1;do

echo "Weather $denoise_layers_num "


python -u run.py \
    --task_name pretrain \
    --root_path ./datasets/weather/ \
    --data_path weather.csv \
    --model_id Weather \
    --model TimeDART \
    --data Weather \
    --features M \
    --input_len 96 \
    --e_layers 2 \
    --d_layers 1 \
    --enc_in 21 \
    --dec_in 21 \
    --denoise_layers_num $denoise_layers_num \
    --use_pretrain_encoder $use_pretrain_encoder \
    --use_finetune_encoder $use_finetune_encoder \
    --use_pretrain_in_ft $use_pretrain_in_ft \
    --c_out 21 \
    --n_heads 8 \
    --d_model $d_model \
    --d_ff 64 \
    --patch_len 2 \
    --stride 2 \
    --head_dropout 0.1 \
    --dropout 0.2 \
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
    --lr_decay 0.95 \
    --learning_rate 0.001 \
    --batch_size $batch_size \
    --train_epochs 20 \
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
> ./run_log/log_202601252150012_win/weather/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1



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
    --use_pretrain_encoder $use_pretrain_encoder \
    --use_finetune_encoder $use_finetune_encoder \
    --use_pretrain_in_ft $use_pretrain_in_ft \
        --features M \
        --input_len 96 \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 2 \
        --enc_in 21 \
        --dec_in 21 \
        --c_out 21 \
        --n_heads 8 \
        --d_model $d_model \
        --d_ff 64 \
        --patch_len 2 \
        --stride 2 \
        --dropout 0.2 \
        --head_dropout 0.1 \
        --batch_size $batch_size \
        --del_orth_loss $del_orth_loss \
    --mask_ratio $mask_ratio \
    --use_geo_mask $use_geo_mask \
    --pretrain_mode $pretrain_mode \
    --predict_eps $predict_eps \
        --del_season_freq_loss $del_season_freq_loss \
        --del_smoothness_loss $del_smoothness_loss \
        --del_freq_loss $del_freq_loss \
        --del_recon_loss $del_recon_loss \
        --lr_decay 0.5 \
        --lradj step \
        --time_steps 1000 \
        --scheduler cosine \
        --patience 3 \
        --learning_rate 0.0004 \
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
> ./run_log/log_202601252150012_win/weather/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1

done





done



#---------------- ELE script

# ELE mask
use_geo_mask=1
use_geo_mask=0
predict_eps=0
pretrain_mode=noise
mask_ratio=0.9
layer_ecl=1
for denoise_layers_num in 1;do



#
n_heads=16
d_model=128
batch_size=16

echo "ECL $denoise_layers_num "


python -u run.py \
    --task_name pretrain \
    --root_path ./datasets/electricity/ \
    --data_path electricity.csv \
    --model_id Electricity \
    --model TimeDART \
    --data Electricity \
    --features M \
    --input_len 96 \
    --e_layers 2 \
    --d_layers 1 \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
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
    --train_epochs 20 \
> ./run_log/log_202601252150012_win/electricity/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1

n_heads = 16
d_model=128
batch_size=16
#batch_size=8
#d_model=64

for pred_len in 96 192 336 720; do
echo "ECL $denoise_layers_num _ $pred_len"
    python -u run.py \
        --task_name finetune \
        --is_training 1 \
        --root_path ./datasets/electricity/ \
        --data_path electricity.csv \
        --model_id Electricity \
        --model TimeDART \
        --data Electricity \
        --features M \
        --input_len 96 \
        --label_len 48 \
        --denoise_layers_num $denoise_layers_num \
    --use_pretrain_encoder $use_pretrain_encoder \
    --use_finetune_encoder $use_finetune_encoder \
    --use_pretrain_in_ft $use_pretrain_in_ft \
        --pred_len $pred_len \
        --e_layers 2 \
        --enc_in 321 \
        --dec_in 321 \
        --c_out 321 \
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
> ./run_log/log_202601252150012_win/electricity/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1


done
done

done


use_pretrain_encoder=1
use_finetune_encoder=1
destroy_mode=season
echo "use_new_decomp=0  老分解）"
use_new_decomp=0
use_pretrain_in_ft=0
film_mode=full
for _ in 1;do

del_orth_loss=1
del_season_freq_loss=1
del_freq_loss=1
del_smoothness_loss=1
del_recon_loss=1


use_init_loss=1





layer_h1=1
# ETTh1 noise
use_geo_mask=1
use_geo_mask=0
predict_eps=1
pretrain_mode=noise
mask_ratio=0.41
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
    --use_pretrain_encoder $use_pretrain_encoder \
    --use_finetune_encoder $use_finetune_encoder \
    --use_pretrain_in_ft $use_pretrain_in_ft \
    --film_mode $film_mode \
    --patch_len 2 \
    --stride 2 \
    --head_dropout 0.1 \
    --dropout 0.2 \
    --time_steps 1000 \
    --scheduler cosine \
    --lr_decay 0.9 \
    --learning_rate 0.0001 \
    --batch_size 16 \
    --train_epochs 20 \
    --del_orth_loss $del_orth_loss \
    --mask_ratio $mask_ratio \
    --use_geo_mask $use_geo_mask \
    --pretrain_mode $pretrain_mode \
    --predict_eps $predict_eps \
    --del_season_freq_loss $del_season_freq_loss \
    --del_smoothness_loss $del_smoothness_loss \
    --del_freq_loss $del_freq_loss \
    --del_recon_loss $del_recon_loss \
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
> ./run_log/log_202601252150012_win/ETTh1/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1
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
    --use_pretrain_encoder $use_pretrain_encoder \
    --use_finetune_encoder $use_finetune_encoder \
    --use_pretrain_in_ft $use_pretrain_in_ft \
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
        --del_freq_loss $del_freq_loss \
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
> ./run_log/log_202601252150012_win/ETTh1/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1

done

done


# ETTh2 mask
layer_h2=2
use_geo_mask=1
use_geo_mask=0
predict_eps=0
pretrain_mode=noise
mask_ratio=0.7
for denoise_layers_num in 1;do




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
    --use_pretrain_encoder $use_pretrain_encoder \
    --use_finetune_encoder $use_finetune_encoder \
    --use_pretrain_in_ft $use_pretrain_in_ft \
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
    --train_epochs 20 \
    --del_orth_loss $del_orth_loss \
    --mask_ratio $mask_ratio \
    --use_geo_mask $use_geo_mask \
    --pretrain_mode $pretrain_mode \
    --predict_eps $predict_eps \
    --del_season_freq_loss $del_season_freq_loss \
    --del_smoothness_loss $del_smoothness_loss \
    --del_freq_loss $del_freq_loss \
    --del_recon_loss $del_recon_loss \
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
> ./run_log/log_202601252150012_win/ETTh2/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1


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
    --use_pretrain_encoder $use_pretrain_encoder \
    --use_finetune_encoder $use_finetune_encoder \
    --use_pretrain_in_ft $use_pretrain_in_ft \
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
> ./run_log/log_202601252150012_win/ETTh2/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1


done


done




layer_m1=1
# ETTM1 noise
use_geo_mask=1
use_geo_mask=0
predict_eps=1
pretrain_mode=noise

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
    --use_pretrain_encoder $use_pretrain_encoder \
    --use_finetune_encoder $use_finetune_encoder \
    --use_pretrain_in_ft $use_pretrain_in_ft \
    --patch_len 2 \
    --stride 2 \
    --head_dropout 0.1 \
    --dropout 0.2 \
    --time_steps 1000 \
    --scheduler cosine \
    --lr_decay 0.95 \
    --learning_rate 0.0001 \
    --batch_size 64 \
    --train_epochs 20 \
    --del_orth_loss $del_orth_loss \
    --mask_ratio $mask_ratio \
    --use_geo_mask $use_geo_mask \
    --pretrain_mode $pretrain_mode \
    --predict_eps $predict_eps \
    --del_season_freq_loss $del_season_freq_loss \
    --del_smoothness_loss $del_smoothness_loss \
    --del_freq_loss $del_freq_loss \
    --del_recon_loss $del_recon_loss \
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
> ./run_log/log_202601252150012_win/ETTm1/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1

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
    --use_pretrain_encoder $use_pretrain_encoder \
    --use_finetune_encoder $use_finetune_encoder \
    --use_pretrain_in_ft $use_pretrain_in_ft \
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
> ./run_log/log_202601252150012_win/ETTm1/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1

done
done


# ETTm2 mask
use_geo_mask=1
use_geo_mask=0
predict_eps=0
pretrain_mode=noise
mask_ratio=0.3
layer_m2=1
for denoise_layers_num in 1;do


# -------------- ETTm2 script
echo "ETTm2 $denoise_layers_num "

python -u run.py \
    --task_name pretrain \
    --root_path ./datasets/ETT-small/ \
    --data_path ETTm2.csv \
    --model_id ETTm2 \
    --model TimeDART \
    --data ETTm2 \
    --features M \
    --input_len 96 \
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
    --denoise_layers_num $denoise_layers_num \
    --use_pretrain_encoder $use_pretrain_encoder \
    --use_finetune_encoder $use_finetune_encoder \
    --use_pretrain_in_ft $use_pretrain_in_ft \
    --dropout 0.2 \
    --time_steps 1000 \
    --scheduler cosine \
    --lr_decay 0.8 \
    --learning_rate 0.001 \
    --batch_size 64 \
    --del_orth_loss $del_orth_loss \
    --mask_ratio $mask_ratio \
    --use_geo_mask $use_geo_mask \
    --pretrain_mode $pretrain_mode \
    --predict_eps $predict_eps \
    --del_season_freq_loss $del_season_freq_loss \
    --del_smoothness_loss $del_smoothness_loss \
    --del_recon_loss $del_recon_loss \
    --del_freq_loss $del_freq_loss \
    --train_epochs 20 \
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
> ./run_log/log_202601252150012_win/ETTm2/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1

d_model=8
n_heads=8 # 老
#n_heads=4

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
        --input_len 96 \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 2 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --n_heads $n_heads \
        --d_model $d_model \
        --d_ff 16 \
        --patch_len 2 \
        --stride 2 \
        --dropout 0.4 \
        --head_dropout 0.1 \
        --denoise_layers_num $denoise_layers_num \
    --use_pretrain_encoder $use_pretrain_encoder \
    --use_finetune_encoder $use_finetune_encoder \
    --use_pretrain_in_ft $use_pretrain_in_ft \
        --batch_size 64 \
        --lr_decay 0.5 \
        --lradj step \
        --time_steps 1000 \
        --scheduler cosine \
        --patience 3 \
        --learning_rate 0.0001 \
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
        --pct_start 0.2 \
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
> ./run_log/log_202601252150012_win/ETTm2/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1

done



done



#---------------- Exchange script
# Exchange mask
use_geo_mask=1
use_geo_mask=0
predict_eps=0
pretrain_mode=noise
mask_ratio=0.5
layer_exchange=1
for denoise_layers_num in 1 ;do


d_model=64
d_model=32
n_heads=8
batch_size=16

echo "Exchange $denoise_layers_num"

python -u run.py \
    --task_name pretrain \
    --root_path ./datasets/exchange_rate/ \
    --data_path exchange_rate.csv \
    --model_id Exchange \
    --model TimeDART \
    --data Exchange \
    --features M \
    --input_len 96 \
    --e_layers 2 \
    --d_layers 1 \
    --enc_in 8 \
    --dec_in 8 \
    --denoise_layers_num $denoise_layers_num \
    --use_pretrain_encoder $use_pretrain_encoder \
    --use_finetune_encoder $use_finetune_encoder \
    --use_pretrain_in_ft $use_pretrain_in_ft \
    --c_out 8 \
    --n_heads $n_heads \
    --d_model $d_model \
    --d_ff 64 \
    --patch_len 2 \
    --stride 2 \
    --head_dropout 0.1 \
    --dropout 0.2 \
    --time_steps 1000 \
    --scheduler cosine \
    --lr_decay 0.95 \
    --learning_rate 0.001 \
    --batch_size $batch_size \
    --train_epochs 20 \
    --del_orth_loss $del_orth_loss \
    --mask_ratio $mask_ratio \
    --use_geo_mask $use_geo_mask \
    --pretrain_mode $pretrain_mode \
    --predict_eps $predict_eps \
    --del_season_freq_loss $del_season_freq_loss \
    --del_smoothness_loss $del_smoothness_loss \
    --del_freq_loss $del_freq_loss \
    --del_recon_loss $del_recon_loss \
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
> ./run_log/log_202601252150012_win/Exchange/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1

#d_model=64
#n_heads=8
#batch_size=16

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
        --input_len 96 \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 2 \
        --denoise_layers_num $denoise_layers_num \
    --use_pretrain_encoder $use_pretrain_encoder \
    --use_finetune_encoder $use_finetune_encoder \
    --use_pretrain_in_ft $use_pretrain_in_ft \
        --enc_in 8 \
        --dec_in 8 \
        --c_out 8 \
        --n_heads $n_heads \
        --d_model $d_model \
        --d_ff 64 \
        --patch_len 2 \
        --stride 2 \
        --dropout 0.2 \
        --head_dropout 0.1 \
        --batch_size $batch_size \
        --lr_decay 0.8 \
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
> ./run_log/log_202601252150012_win/Exchange/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1

done
done



#----------------- WTH script
# WTH mask
use_geo_mask=1
use_geo_mask=0
predict_eps=0

#原始32dmodel 16batch
d_model=32
d_model=64
batch_size=16
layer_wth=1
pretrain_mode=noise
mask_ratio=0.7
for denoise_layers_num in 1;do

echo "Weather $denoise_layers_num "


python -u run.py \
    --task_name pretrain \
    --root_path ./datasets/weather/ \
    --data_path weather.csv \
    --model_id Weather \
    --model TimeDART \
    --data Weather \
    --features M \
    --input_len 96 \
    --e_layers 2 \
    --d_layers 1 \
    --enc_in 21 \
    --dec_in 21 \
    --denoise_layers_num $denoise_layers_num \
    --use_pretrain_encoder $use_pretrain_encoder \
    --use_finetune_encoder $use_finetune_encoder \
    --use_pretrain_in_ft $use_pretrain_in_ft \
    --c_out 21 \
    --n_heads 8 \
    --d_model $d_model \
    --d_ff 64 \
    --patch_len 2 \
    --stride 2 \
    --head_dropout 0.1 \
    --dropout 0.2 \
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
    --lr_decay 0.95 \
    --learning_rate 0.001 \
    --batch_size $batch_size \
    --train_epochs 20 \
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
> ./run_log/log_202601252150012_win/weather/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1



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
    --use_pretrain_encoder $use_pretrain_encoder \
    --use_finetune_encoder $use_finetune_encoder \
    --use_pretrain_in_ft $use_pretrain_in_ft \
        --features M \
        --input_len 96 \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 2 \
        --enc_in 21 \
        --dec_in 21 \
        --c_out 21 \
        --n_heads 8 \
        --d_model $d_model \
        --d_ff 64 \
        --patch_len 2 \
        --stride 2 \
        --dropout 0.2 \
        --head_dropout 0.1 \
        --batch_size $batch_size \
        --del_orth_loss $del_orth_loss \
    --mask_ratio $mask_ratio \
    --use_geo_mask $use_geo_mask \
    --pretrain_mode $pretrain_mode \
    --predict_eps $predict_eps \
        --del_season_freq_loss $del_season_freq_loss \
        --del_smoothness_loss $del_smoothness_loss \
        --del_freq_loss $del_freq_loss \
        --del_recon_loss $del_recon_loss \
        --lr_decay 0.5 \
        --lradj step \
        --time_steps 1000 \
        --scheduler cosine \
        --patience 3 \
        --learning_rate 0.0004 \
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
> ./run_log/log_202601252150012_win/weather/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1

done





done



#---------------- ELE script

# ELE mask
use_geo_mask=1
use_geo_mask=0
predict_eps=0
pretrain_mode=noise
mask_ratio=0.9
layer_ecl=1
for denoise_layers_num in 1;do



#
n_heads=16
d_model=128
batch_size=16

echo "ECL $denoise_layers_num "


python -u run.py \
    --task_name pretrain \
    --root_path ./datasets/electricity/ \
    --data_path electricity.csv \
    --model_id Electricity \
    --model TimeDART \
    --data Electricity \
    --features M \
    --input_len 96 \
    --e_layers 2 \
    --d_layers 1 \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
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
    --train_epochs 20 \
> ./run_log/log_202601252150012_win/electricity/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1

n_heads = 16
d_model=128
batch_size=16
#batch_size=8
#d_model=64

for pred_len in 96 192 336 720; do
echo "ECL $denoise_layers_num _ $pred_len"
    python -u run.py \
        --task_name finetune \
        --is_training 1 \
        --root_path ./datasets/electricity/ \
        --data_path electricity.csv \
        --model_id Electricity \
        --model TimeDART \
        --data Electricity \
        --features M \
        --input_len 96 \
        --label_len 48 \
        --denoise_layers_num $denoise_layers_num \
    --use_pretrain_encoder $use_pretrain_encoder \
    --use_finetune_encoder $use_finetune_encoder \
    --use_pretrain_in_ft $use_pretrain_in_ft \
        --pred_len $pred_len \
        --e_layers 2 \
        --enc_in 321 \
        --dec_in 321 \
        --c_out 321 \
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
> ./run_log/log_202601252150012_win/electricity/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1


done
done

done




use_pretrain_encoder=1
use_finetune_encoder=1
destroy_mode=season
echo "use_new_decomp=2  表示不分解）"
use_new_decomp=2
use_pretrain_in_ft=0
film_mode=full
for _ in 1;do

del_orth_loss=1
del_season_freq_loss=1
del_freq_loss=1
del_smoothness_loss=1
del_recon_loss=1


use_init_loss=1





layer_h1=1
# ETTh1 noise
use_geo_mask=1
use_geo_mask=0
predict_eps=1
pretrain_mode=noise
mask_ratio=0.41
for denoise_layers_num in 1;do


echo "ETTh1 $denoise_layers_num"
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
#    --use_pretrain_encoder $use_pretrain_encoder \
#    --use_finetune_encoder $use_finetune_encoder \
#    --use_pretrain_in_ft $use_pretrain_in_ft \
#    --film_mode $film_mode \
#    --patch_len 2 \
#    --stride 2 \
#    --head_dropout 0.1 \
#    --dropout 0.2 \
#    --time_steps 1000 \
#    --scheduler cosine \
#    --lr_decay 0.9 \
#    --learning_rate 0.0001 \
#    --batch_size 16 \
#    --train_epochs 20 \
#    --del_orth_loss $del_orth_loss \
#    --mask_ratio $mask_ratio \
#    --use_geo_mask $use_geo_mask \
#    --pretrain_mode $pretrain_mode \
#    --predict_eps $predict_eps \
#    --del_season_freq_loss $del_season_freq_loss \
#    --del_smoothness_loss $del_smoothness_loss \
#    --del_freq_loss $del_freq_loss \
#    --del_recon_loss $del_recon_loss \
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
#> ./run_log/log_202601252150012_win/ETTh1/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1
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
    --use_pretrain_encoder $use_pretrain_encoder \
    --use_finetune_encoder $use_finetune_encoder \
    --use_pretrain_in_ft $use_pretrain_in_ft \
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
        --del_freq_loss $del_freq_loss \
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
> ./run_log/log_202601252150012_win/ETTh1/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1

done

done


# ETTh2 mask
layer_h2=2
use_geo_mask=1
use_geo_mask=0
predict_eps=0
pretrain_mode=noise
mask_ratio=0.7
for denoise_layers_num in 1;do




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
    --use_pretrain_encoder $use_pretrain_encoder \
    --use_finetune_encoder $use_finetune_encoder \
    --use_pretrain_in_ft $use_pretrain_in_ft \
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
    --train_epochs 20 \
    --del_orth_loss $del_orth_loss \
    --mask_ratio $mask_ratio \
    --use_geo_mask $use_geo_mask \
    --pretrain_mode $pretrain_mode \
    --predict_eps $predict_eps \
    --del_season_freq_loss $del_season_freq_loss \
    --del_smoothness_loss $del_smoothness_loss \
    --del_freq_loss $del_freq_loss \
    --del_recon_loss $del_recon_loss \
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
> ./run_log/log_202601252150012_win/ETTh2/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1


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
    --use_pretrain_encoder $use_pretrain_encoder \
    --use_finetune_encoder $use_finetune_encoder \
    --use_pretrain_in_ft $use_pretrain_in_ft \
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
> ./run_log/log_202601252150012_win/ETTh2/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1


done


done




layer_m1=1
# ETTM1 noise
use_geo_mask=1
use_geo_mask=0
predict_eps=1
pretrain_mode=noise

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
    --use_pretrain_encoder $use_pretrain_encoder \
    --use_finetune_encoder $use_finetune_encoder \
    --use_pretrain_in_ft $use_pretrain_in_ft \
    --patch_len 2 \
    --stride 2 \
    --head_dropout 0.1 \
    --dropout 0.2 \
    --time_steps 1000 \
    --scheduler cosine \
    --lr_decay 0.95 \
    --learning_rate 0.0001 \
    --batch_size 64 \
    --train_epochs 20 \
    --del_orth_loss $del_orth_loss \
    --mask_ratio $mask_ratio \
    --use_geo_mask $use_geo_mask \
    --pretrain_mode $pretrain_mode \
    --predict_eps $predict_eps \
    --del_season_freq_loss $del_season_freq_loss \
    --del_smoothness_loss $del_smoothness_loss \
    --del_freq_loss $del_freq_loss \
    --del_recon_loss $del_recon_loss \
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
> ./run_log/log_202601252150012_win/ETTm1/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1

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
    --use_pretrain_encoder $use_pretrain_encoder \
    --use_finetune_encoder $use_finetune_encoder \
    --use_pretrain_in_ft $use_pretrain_in_ft \
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
> ./run_log/log_202601252150012_win/ETTm1/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1

done
done


# ETTm2 mask
use_geo_mask=1
use_geo_mask=0
predict_eps=0
pretrain_mode=noise
mask_ratio=0.3
layer_m2=1
for denoise_layers_num in 1;do


# -------------- ETTm2 script
echo "ETTm2 $denoise_layers_num "

python -u run.py \
    --task_name pretrain \
    --root_path ./datasets/ETT-small/ \
    --data_path ETTm2.csv \
    --model_id ETTm2 \
    --model TimeDART \
    --data ETTm2 \
    --features M \
    --input_len 96 \
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
    --denoise_layers_num $denoise_layers_num \
    --use_pretrain_encoder $use_pretrain_encoder \
    --use_finetune_encoder $use_finetune_encoder \
    --use_pretrain_in_ft $use_pretrain_in_ft \
    --dropout 0.2 \
    --time_steps 1000 \
    --scheduler cosine \
    --lr_decay 0.8 \
    --learning_rate 0.001 \
    --batch_size 64 \
    --del_orth_loss $del_orth_loss \
    --mask_ratio $mask_ratio \
    --use_geo_mask $use_geo_mask \
    --pretrain_mode $pretrain_mode \
    --predict_eps $predict_eps \
    --del_season_freq_loss $del_season_freq_loss \
    --del_smoothness_loss $del_smoothness_loss \
    --del_recon_loss $del_recon_loss \
    --del_freq_loss $del_freq_loss \
    --train_epochs 20 \
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
> ./run_log/log_202601252150012_win/ETTm2/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1

d_model=8
n_heads=8 # 老
#n_heads=4

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
        --input_len 96 \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 2 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --n_heads $n_heads \
        --d_model $d_model \
        --d_ff 16 \
        --patch_len 2 \
        --stride 2 \
        --dropout 0.4 \
        --head_dropout 0.1 \
        --denoise_layers_num $denoise_layers_num \
    --use_pretrain_encoder $use_pretrain_encoder \
    --use_finetune_encoder $use_finetune_encoder \
    --use_pretrain_in_ft $use_pretrain_in_ft \
        --batch_size 64 \
        --lr_decay 0.5 \
        --lradj step \
        --time_steps 1000 \
        --scheduler cosine \
        --patience 3 \
        --learning_rate 0.0001 \
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
        --pct_start 0.2 \
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
> ./run_log/log_202601252150012_win/ETTm2/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1

done



done



#---------------- Exchange script
# Exchange mask
use_geo_mask=1
use_geo_mask=0
predict_eps=0
pretrain_mode=noise
mask_ratio=0.5
layer_exchange=1
for denoise_layers_num in 1 ;do


d_model=64
d_model=32
n_heads=8
batch_size=16

echo "Exchange $denoise_layers_num"

python -u run.py \
    --task_name pretrain \
    --root_path ./datasets/exchange_rate/ \
    --data_path exchange_rate.csv \
    --model_id Exchange \
    --model TimeDART \
    --data Exchange \
    --features M \
    --input_len 96 \
    --e_layers 2 \
    --d_layers 1 \
    --enc_in 8 \
    --dec_in 8 \
    --denoise_layers_num $denoise_layers_num \
    --use_pretrain_encoder $use_pretrain_encoder \
    --use_finetune_encoder $use_finetune_encoder \
    --use_pretrain_in_ft $use_pretrain_in_ft \
    --c_out 8 \
    --n_heads $n_heads \
    --d_model $d_model \
    --d_ff 64 \
    --patch_len 2 \
    --stride 2 \
    --head_dropout 0.1 \
    --dropout 0.2 \
    --time_steps 1000 \
    --scheduler cosine \
    --lr_decay 0.95 \
    --learning_rate 0.001 \
    --batch_size $batch_size \
    --train_epochs 20 \
    --del_orth_loss $del_orth_loss \
    --mask_ratio $mask_ratio \
    --use_geo_mask $use_geo_mask \
    --pretrain_mode $pretrain_mode \
    --predict_eps $predict_eps \
    --del_season_freq_loss $del_season_freq_loss \
    --del_smoothness_loss $del_smoothness_loss \
    --del_freq_loss $del_freq_loss \
    --del_recon_loss $del_recon_loss \
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
> ./run_log/log_202601252150012_win/Exchange/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1

#d_model=64
#n_heads=8
#batch_size=16

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
        --input_len 96 \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 2 \
        --denoise_layers_num $denoise_layers_num \
    --use_pretrain_encoder $use_pretrain_encoder \
    --use_finetune_encoder $use_finetune_encoder \
    --use_pretrain_in_ft $use_pretrain_in_ft \
        --enc_in 8 \
        --dec_in 8 \
        --c_out 8 \
        --n_heads $n_heads \
        --d_model $d_model \
        --d_ff 64 \
        --patch_len 2 \
        --stride 2 \
        --dropout 0.2 \
        --head_dropout 0.1 \
        --batch_size $batch_size \
        --lr_decay 0.8 \
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
> ./run_log/log_202601252150012_win/Exchange/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1

done
done



#----------------- WTH script
# WTH mask
use_geo_mask=1
use_geo_mask=0
predict_eps=0

#原始32dmodel 16batch
d_model=32
d_model=64
batch_size=16
layer_wth=1
pretrain_mode=noise
mask_ratio=0.7
for denoise_layers_num in 1;do

echo "Weather $denoise_layers_num "


python -u run.py \
    --task_name pretrain \
    --root_path ./datasets/weather/ \
    --data_path weather.csv \
    --model_id Weather \
    --model TimeDART \
    --data Weather \
    --features M \
    --input_len 96 \
    --e_layers 2 \
    --d_layers 1 \
    --enc_in 21 \
    --dec_in 21 \
    --denoise_layers_num $denoise_layers_num \
    --use_pretrain_encoder $use_pretrain_encoder \
    --use_finetune_encoder $use_finetune_encoder \
    --use_pretrain_in_ft $use_pretrain_in_ft \
    --c_out 21 \
    --n_heads 8 \
    --d_model $d_model \
    --d_ff 64 \
    --patch_len 2 \
    --stride 2 \
    --head_dropout 0.1 \
    --dropout 0.2 \
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
    --lr_decay 0.95 \
    --learning_rate 0.001 \
    --batch_size $batch_size \
    --train_epochs 20 \
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
> ./run_log/log_202601252150012_win/weather/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1



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
    --use_pretrain_encoder $use_pretrain_encoder \
    --use_finetune_encoder $use_finetune_encoder \
    --use_pretrain_in_ft $use_pretrain_in_ft \
        --features M \
        --input_len 96 \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 2 \
        --enc_in 21 \
        --dec_in 21 \
        --c_out 21 \
        --n_heads 8 \
        --d_model $d_model \
        --d_ff 64 \
        --patch_len 2 \
        --stride 2 \
        --dropout 0.2 \
        --head_dropout 0.1 \
        --batch_size $batch_size \
        --del_orth_loss $del_orth_loss \
    --mask_ratio $mask_ratio \
    --use_geo_mask $use_geo_mask \
    --pretrain_mode $pretrain_mode \
    --predict_eps $predict_eps \
        --del_season_freq_loss $del_season_freq_loss \
        --del_smoothness_loss $del_smoothness_loss \
        --del_freq_loss $del_freq_loss \
        --del_recon_loss $del_recon_loss \
        --lr_decay 0.5 \
        --lradj step \
        --time_steps 1000 \
        --scheduler cosine \
        --patience 3 \
        --learning_rate 0.0004 \
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
> ./run_log/log_202601252150012_win/weather/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1

done





done



#---------------- ELE script

# ELE mask
use_geo_mask=1
use_geo_mask=0
predict_eps=0
pretrain_mode=noise
mask_ratio=0.9
layer_ecl=1
for denoise_layers_num in 1;do



#
n_heads=16
d_model=128
batch_size=16

echo "ECL $denoise_layers_num "


python -u run.py \
    --task_name pretrain \
    --root_path ./datasets/electricity/ \
    --data_path electricity.csv \
    --model_id Electricity \
    --model TimeDART \
    --data Electricity \
    --features M \
    --input_len 96 \
    --e_layers 2 \
    --d_layers 1 \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
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
    --train_epochs 20 \
> ./run_log/log_202601252150012_win/electricity/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1

n_heads = 16
d_model=128
batch_size=16
#batch_size=8
#d_model=64

for pred_len in 96 192 336 720; do
echo "ECL $denoise_layers_num _ $pred_len"
    python -u run.py \
        --task_name finetune \
        --is_training 1 \
        --root_path ./datasets/electricity/ \
        --data_path electricity.csv \
        --model_id Electricity \
        --model TimeDART \
        --data Electricity \
        --features M \
        --input_len 96 \
        --label_len 48 \
        --denoise_layers_num $denoise_layers_num \
    --use_pretrain_encoder $use_pretrain_encoder \
    --use_finetune_encoder $use_finetune_encoder \
    --use_pretrain_in_ft $use_pretrain_in_ft \
        --pred_len $pred_len \
        --e_layers 2 \
        --enc_in 321 \
        --dec_in 321 \
        --c_out 321 \
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
> ./run_log/log_202601252150012_win/electricity/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1


done
done

done
fi


use_pretrain_encoder=1
use_finetune_encoder=1
destroy_mode=season
echo "不用film"
use_new_decomp=1
use_pretrain_in_ft=0
film_mode=full
no_film=0
for _ in 1;do

del_orth_loss=1
del_season_freq_loss=1
del_freq_loss=1
del_smoothness_loss=1
del_recon_loss=1


use_init_loss=1





layer_h1=1
# ETTh1 noise
use_geo_mask=1
use_geo_mask=0
predict_eps=1
pretrain_mode=noise
mask_ratio=0.41
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
    --use_pretrain_encoder $use_pretrain_encoder \
    --use_finetune_encoder $use_finetune_encoder \
    --use_pretrain_in_ft $use_pretrain_in_ft \
    --film_mode $film_mode \
    --no_film $no_film \
    --patch_len 2 \
    --stride 2 \
    --head_dropout 0.1 \
    --dropout 0.2 \
    --time_steps 1000 \
    --scheduler cosine \
    --lr_decay 0.9 \
    --learning_rate 0.0001 \
    --batch_size 16 \
    --train_epochs 20 \
    --del_orth_loss $del_orth_loss \
    --mask_ratio $mask_ratio \
    --use_geo_mask $use_geo_mask \
    --pretrain_mode $pretrain_mode \
    --predict_eps $predict_eps \
    --del_season_freq_loss $del_season_freq_loss \
    --del_smoothness_loss $del_smoothness_loss \
    --del_freq_loss $del_freq_loss \
    --del_recon_loss $del_recon_loss \
    --use_init_loss $use_init_loss \
    --use_new_decomp $use_new_decomp \
      --no_film $no_film \
    --use_loss_compute $use_loss_compute \
    --use_denoise $use_denoise \
    --use_inner_new_decomp $use_inner_new_decomp \
    --use_defire_noise $use_defire_noise \
    --use_positional_encoding $use_positional_encoding \
    --use_sostoken $use_sostoken \
    --use_inner_encoder $use_inner_encoder \
    --down_sampling_window 2 \
> ./run_log/log_202601252150012_win/ETTh1/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1
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
    --use_pretrain_encoder $use_pretrain_encoder \
    --use_finetune_encoder $use_finetune_encoder \
    --use_pretrain_in_ft $use_pretrain_in_ft \
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
        --del_freq_loss $del_freq_loss \
        --use_init_loss $use_init_loss \
      --use_new_decomp $use_new_decomp \
      --no_film $no_film \
      --no_film $no_film \
      --use_loss_compute $use_loss_compute \
      --use_denoise $use_denoise \
      --use_inner_new_decomp $use_inner_new_decomp \
      --use_defire_noise $use_defire_noise \
      --use_positional_encoding $use_positional_encoding \
      --use_sostoken $use_sostoken \
      --use_inner_encoder $use_inner_encoder \
        --down_sampling_window 2 \
> ./run_log/log_202601252150012_win/ETTh1/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1

done

done


# ETTh2 mask
layer_h2=2
use_geo_mask=1
use_geo_mask=0
predict_eps=0
pretrain_mode=noise
mask_ratio=0.7
for denoise_layers_num in 1;do




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
    --use_pretrain_encoder $use_pretrain_encoder \
    --use_finetune_encoder $use_finetune_encoder \
    --use_pretrain_in_ft $use_pretrain_in_ft \
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
    --train_epochs 20 \
    --del_orth_loss $del_orth_loss \
    --mask_ratio $mask_ratio \
    --use_geo_mask $use_geo_mask \
    --pretrain_mode $pretrain_mode \
    --predict_eps $predict_eps \
    --del_season_freq_loss $del_season_freq_loss \
    --del_smoothness_loss $del_smoothness_loss \
    --del_freq_loss $del_freq_loss \
    --del_recon_loss $del_recon_loss \
    --use_init_loss $use_init_loss \
    --use_new_decomp $use_new_decomp \
      --no_film $no_film \
    --use_loss_compute $use_loss_compute \
    --use_denoise $use_denoise \
    --use_inner_new_decomp $use_inner_new_decomp \
    --use_defire_noise $use_defire_noise \
    --use_positional_encoding $use_positional_encoding \
    --use_sostoken $use_sostoken \
    --use_inner_encoder $use_inner_encoder \
    --down_sampling_window 2 \
> ./run_log/log_202601252150012_win/ETTh2/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1


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
    --use_pretrain_encoder $use_pretrain_encoder \
    --use_finetune_encoder $use_finetune_encoder \
    --use_pretrain_in_ft $use_pretrain_in_ft \
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
        --learning_rate 0.0001 \
        --pct_start 0.3 \
        --use_init_loss $use_init_loss \
    --use_new_decomp $use_new_decomp \
      --no_film $no_film \
    --use_loss_compute $use_loss_compute \
    --use_denoise $use_denoise \
    --use_inner_new_decomp $use_inner_new_decomp \
    --use_defire_noise $use_defire_noise \
    --use_positional_encoding $use_positional_encoding \
    --use_sostoken $use_sostoken \
    --use_inner_encoder $use_inner_encoder \
        --down_sampling_window 2 \
> ./run_log/log_202601252150012_win/ETTh2/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1


done


done




layer_m1=1
# ETTM1 noise
use_geo_mask=1
use_geo_mask=0
predict_eps=1
pretrain_mode=noise

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
    --use_pretrain_encoder $use_pretrain_encoder \
    --use_finetune_encoder $use_finetune_encoder \
    --use_pretrain_in_ft $use_pretrain_in_ft \
    --patch_len 2 \
    --stride 2 \
    --head_dropout 0.1 \
    --dropout 0.2 \
    --time_steps 1000 \
    --scheduler cosine \
    --lr_decay 0.95 \
    --learning_rate 0.0001 \
    --batch_size 64 \
    --train_epochs 20 \
    --del_orth_loss $del_orth_loss \
    --mask_ratio $mask_ratio \
    --use_geo_mask $use_geo_mask \
    --pretrain_mode $pretrain_mode \
    --predict_eps $predict_eps \
    --del_season_freq_loss $del_season_freq_loss \
    --del_smoothness_loss $del_smoothness_loss \
    --del_freq_loss $del_freq_loss \
    --del_recon_loss $del_recon_loss \
    --use_init_loss $use_init_loss \
    --use_new_decomp $use_new_decomp \
      --no_film $no_film \
    --use_loss_compute $use_loss_compute \
    --use_denoise $use_denoise \
    --use_inner_new_decomp $use_inner_new_decomp \
    --use_defire_noise $use_defire_noise \
    --use_positional_encoding $use_positional_encoding \
    --use_sostoken $use_sostoken \
    --use_inner_encoder $use_inner_encoder \
    --down_sampling_window 2 \
> ./run_log/log_202601252150012_win/ETTm1/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1

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
    --use_pretrain_encoder $use_pretrain_encoder \
    --use_finetune_encoder $use_finetune_encoder \
    --use_pretrain_in_ft $use_pretrain_in_ft \
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
        --patience 3 \
        --learning_rate 0.0001 \
        --pct_start 0.3 \
       --use_init_loss $use_init_loss \
    --use_new_decomp $use_new_decomp \
      --no_film $no_film \
    --use_loss_compute $use_loss_compute \
    --use_denoise $use_denoise \
    --use_inner_new_decomp $use_inner_new_decomp \
    --use_defire_noise $use_defire_noise \
    --use_positional_encoding $use_positional_encoding \
    --use_sostoken $use_sostoken \
    --use_inner_encoder $use_inner_encoder \
       --down_sampling_window 2 \
> ./run_log/log_202601252150012_win/ETTm1/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1

done
done


# ETTm2 mask
use_geo_mask=1
use_geo_mask=0
predict_eps=0
pretrain_mode=noise
mask_ratio=0.3
layer_m2=1
for denoise_layers_num in 1;do


# -------------- ETTm2 script
echo "ETTm2 $denoise_layers_num "

python -u run.py \
    --task_name pretrain \
    --root_path ./datasets/ETT-small/ \
    --data_path ETTm2.csv \
    --model_id ETTm2 \
    --model TimeDART \
    --data ETTm2 \
    --features M \
    --input_len 96 \
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
    --denoise_layers_num $denoise_layers_num \
    --use_pretrain_encoder $use_pretrain_encoder \
    --use_finetune_encoder $use_finetune_encoder \
    --use_pretrain_in_ft $use_pretrain_in_ft \
    --dropout 0.2 \
    --time_steps 1000 \
    --scheduler cosine \
    --lr_decay 0.8 \
    --learning_rate 0.001 \
    --batch_size 64 \
    --del_orth_loss $del_orth_loss \
    --mask_ratio $mask_ratio \
    --use_geo_mask $use_geo_mask \
    --pretrain_mode $pretrain_mode \
    --predict_eps $predict_eps \
    --del_season_freq_loss $del_season_freq_loss \
    --del_smoothness_loss $del_smoothness_loss \
    --del_recon_loss $del_recon_loss \
    --del_freq_loss $del_freq_loss \
    --train_epochs 20 \
    --use_init_loss $use_init_loss \
    --use_new_decomp $use_new_decomp \
      --no_film $no_film \
    --use_loss_compute $use_loss_compute \
    --use_denoise $use_denoise \
    --use_inner_new_decomp $use_inner_new_decomp \
    --use_defire_noise $use_defire_noise \
    --use_positional_encoding $use_positional_encoding \
    --use_sostoken $use_sostoken \
    --use_inner_encoder $use_inner_encoder \
    --down_sampling_window 2 \
> ./run_log/log_202601252150012_win/ETTm2/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1

d_model=8
n_heads=8 # 老
#n_heads=4

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
        --input_len 96 \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 2 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --n_heads $n_heads \
        --d_model $d_model \
        --d_ff 16 \
        --patch_len 2 \
        --stride 2 \
        --dropout 0.4 \
        --head_dropout 0.1 \
        --denoise_layers_num $denoise_layers_num \
    --use_pretrain_encoder $use_pretrain_encoder \
    --use_finetune_encoder $use_finetune_encoder \
    --use_pretrain_in_ft $use_pretrain_in_ft \
        --batch_size 64 \
        --lr_decay 0.5 \
        --lradj step \
        --time_steps 1000 \
        --scheduler cosine \
        --patience 3 \
        --learning_rate 0.0001 \
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
        --pct_start 0.2 \
       --use_init_loss $use_init_loss \
    --use_new_decomp $use_new_decomp \
      --no_film $no_film \
    --use_loss_compute $use_loss_compute \
    --use_denoise $use_denoise \
    --use_inner_new_decomp $use_inner_new_decomp \
    --use_defire_noise $use_defire_noise \
    --use_positional_encoding $use_positional_encoding \
    --use_sostoken $use_sostoken \
    --use_inner_encoder $use_inner_encoder \
       --down_sampling_window 2 \
> ./run_log/log_202601252150012_win/ETTm2/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1

done



done



#---------------- Exchange script
# Exchange mask
use_geo_mask=1
use_geo_mask=0
predict_eps=0
pretrain_mode=noise
mask_ratio=0.5
layer_exchange=1
for denoise_layers_num in 1 ;do


d_model=64
d_model=32
n_heads=8
batch_size=16

echo "Exchange $denoise_layers_num"

python -u run.py \
    --task_name pretrain \
    --root_path ./datasets/exchange_rate/ \
    --data_path exchange_rate.csv \
    --model_id Exchange \
    --model TimeDART \
    --data Exchange \
    --features M \
    --input_len 96 \
    --e_layers 2 \
    --d_layers 1 \
    --enc_in 8 \
    --dec_in 8 \
    --denoise_layers_num $denoise_layers_num \
    --use_pretrain_encoder $use_pretrain_encoder \
    --use_finetune_encoder $use_finetune_encoder \
    --use_pretrain_in_ft $use_pretrain_in_ft \
    --c_out 8 \
    --n_heads $n_heads \
    --d_model $d_model \
    --d_ff 64 \
    --patch_len 2 \
    --stride 2 \
    --head_dropout 0.1 \
    --dropout 0.2 \
    --time_steps 1000 \
    --scheduler cosine \
    --lr_decay 0.95 \
    --learning_rate 0.001 \
    --batch_size $batch_size \
    --train_epochs 20 \
    --del_orth_loss $del_orth_loss \
    --mask_ratio $mask_ratio \
    --use_geo_mask $use_geo_mask \
    --pretrain_mode $pretrain_mode \
    --predict_eps $predict_eps \
    --del_season_freq_loss $del_season_freq_loss \
    --del_smoothness_loss $del_smoothness_loss \
    --del_freq_loss $del_freq_loss \
    --del_recon_loss $del_recon_loss \
    --use_init_loss $use_init_loss \
    --use_new_decomp $use_new_decomp \
      --no_film $no_film \
    --use_loss_compute $use_loss_compute \
    --use_denoise $use_denoise \
    --use_inner_new_decomp $use_inner_new_decomp \
    --use_defire_noise $use_defire_noise \
    --use_positional_encoding $use_positional_encoding \
    --use_sostoken $use_sostoken \
    --use_inner_encoder $use_inner_encoder \
    --down_sampling_window 2 \
> ./run_log/log_202601252150012_win/Exchange/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1

#d_model=64
#n_heads=8
#batch_size=16

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
        --input_len 96 \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 2 \
        --denoise_layers_num $denoise_layers_num \
    --use_pretrain_encoder $use_pretrain_encoder \
    --use_finetune_encoder $use_finetune_encoder \
    --use_pretrain_in_ft $use_pretrain_in_ft \
        --enc_in 8 \
        --dec_in 8 \
        --c_out 8 \
        --n_heads $n_heads \
        --d_model $d_model \
        --d_ff 64 \
        --patch_len 2 \
        --stride 2 \
        --dropout 0.2 \
        --head_dropout 0.1 \
        --batch_size $batch_size \
        --lr_decay 0.8 \
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
        --patience 3 \
        --learning_rate 0.0001 \
        --pct_start 0.3 \
       --use_init_loss $use_init_loss \
    --use_new_decomp $use_new_decomp \
      --no_film $no_film \
    --use_loss_compute $use_loss_compute \
    --use_denoise $use_denoise \
    --use_inner_new_decomp $use_inner_new_decomp \
    --use_defire_noise $use_defire_noise \
    --use_positional_encoding $use_positional_encoding \
    --use_sostoken $use_sostoken \
    --use_inner_encoder $use_inner_encoder \
       --down_sampling_window 2 \
> ./run_log/log_202601252150012_win/Exchange/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1

done
done



#----------------- WTH script
# WTH mask
use_geo_mask=1
use_geo_mask=0
predict_eps=0

#原始32dmodel 16batch
d_model=32
d_model=64
batch_size=16
layer_wth=1
pretrain_mode=noise
mask_ratio=0.7
for denoise_layers_num in 1;do

echo "Weather $denoise_layers_num "


python -u run.py \
    --task_name pretrain \
    --root_path ./datasets/weather/ \
    --data_path weather.csv \
    --model_id Weather \
    --model TimeDART \
    --data Weather \
    --features M \
    --input_len 96 \
    --e_layers 2 \
    --d_layers 1 \
    --enc_in 21 \
    --dec_in 21 \
    --denoise_layers_num $denoise_layers_num \
    --use_pretrain_encoder $use_pretrain_encoder \
    --use_finetune_encoder $use_finetune_encoder \
    --use_pretrain_in_ft $use_pretrain_in_ft \
    --c_out 21 \
    --n_heads 8 \
    --d_model $d_model \
    --d_ff 64 \
    --patch_len 2 \
    --stride 2 \
    --head_dropout 0.1 \
    --dropout 0.2 \
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
    --lr_decay 0.95 \
    --learning_rate 0.001 \
    --batch_size $batch_size \
    --train_epochs 20 \
    --use_init_loss $use_init_loss \
    --use_new_decomp $use_new_decomp \
      --no_film $no_film \
    --use_loss_compute $use_loss_compute \
    --use_denoise $use_denoise \
    --use_inner_new_decomp $use_inner_new_decomp \
    --use_defire_noise $use_defire_noise \
    --use_positional_encoding $use_positional_encoding \
    --use_sostoken $use_sostoken \
    --use_inner_encoder $use_inner_encoder \
    --down_sampling_window 2 \
> ./run_log/log_202601252150012_win/weather/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1



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
    --use_pretrain_encoder $use_pretrain_encoder \
    --use_finetune_encoder $use_finetune_encoder \
    --use_pretrain_in_ft $use_pretrain_in_ft \
        --features M \
        --input_len 96 \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 2 \
        --enc_in 21 \
        --dec_in 21 \
        --c_out 21 \
        --n_heads 8 \
        --d_model $d_model \
        --d_ff 64 \
        --patch_len 2 \
        --stride 2 \
        --dropout 0.2 \
        --head_dropout 0.1 \
        --batch_size $batch_size \
        --del_orth_loss $del_orth_loss \
    --mask_ratio $mask_ratio \
    --use_geo_mask $use_geo_mask \
    --pretrain_mode $pretrain_mode \
    --predict_eps $predict_eps \
        --del_season_freq_loss $del_season_freq_loss \
        --del_smoothness_loss $del_smoothness_loss \
        --del_freq_loss $del_freq_loss \
        --del_recon_loss $del_recon_loss \
        --lr_decay 0.5 \
        --lradj step \
        --time_steps 1000 \
        --scheduler cosine \
        --patience 3 \
        --learning_rate 0.0004 \
        --pct_start 0.3 \
    --use_init_loss $use_init_loss \
    --use_new_decomp $use_new_decomp \
      --no_film $no_film \
    --use_loss_compute $use_loss_compute \
    --use_denoise $use_denoise \
    --use_inner_new_decomp $use_inner_new_decomp \
    --use_defire_noise $use_defire_noise \
    --use_positional_encoding $use_positional_encoding \
    --use_sostoken $use_sostoken \
    --use_inner_encoder $use_inner_encoder \
    --down_sampling_window 2 \
> ./run_log/log_202601252150012_win/weather/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1

done





done



#---------------- ELE script

# ELE mask
use_geo_mask=1
use_geo_mask=0
predict_eps=0
pretrain_mode=noise
mask_ratio=0.9
layer_ecl=1
for denoise_layers_num in 1;do



#
n_heads=16
d_model=128
batch_size=16

echo "ECL $denoise_layers_num "


python -u run.py \
    --task_name pretrain \
    --root_path ./datasets/electricity/ \
    --data_path electricity.csv \
    --model_id Electricity \
    --model TimeDART \
    --data Electricity \
    --features M \
    --input_len 96 \
    --e_layers 2 \
    --d_layers 1 \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
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
      --no_film $no_film \
    --use_loss_compute $use_loss_compute \
    --use_denoise $use_denoise \
    --use_inner_new_decomp $use_inner_new_decomp \
    --use_defire_noise $use_defire_noise \
    --use_positional_encoding $use_positional_encoding \
    --use_sostoken $use_sostoken \
    --use_inner_encoder $use_inner_encoder \
    --down_sampling_window 2 \
    --train_epochs 20 \
> ./run_log/log_202601252150012_win/electricity/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1

n_heads = 16
d_model=128
batch_size=16
#batch_size=8
#d_model=64

for pred_len in 96 192 336 720; do
echo "ECL $denoise_layers_num _ $pred_len"
    python -u run.py \
        --task_name finetune \
        --is_training 1 \
        --root_path ./datasets/electricity/ \
        --data_path electricity.csv \
        --model_id Electricity \
        --model TimeDART \
        --data Electricity \
        --features M \
        --input_len 96 \
        --label_len 48 \
        --denoise_layers_num $denoise_layers_num \
    --use_pretrain_encoder $use_pretrain_encoder \
    --use_finetune_encoder $use_finetune_encoder \
    --use_pretrain_in_ft $use_pretrain_in_ft \
        --pred_len $pred_len \
        --e_layers 2 \
        --enc_in 321 \
        --dec_in 321 \
        --c_out 321 \
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
      --no_film $no_film \
    --use_loss_compute $use_loss_compute \
    --use_denoise $use_denoise \
    --use_inner_new_decomp $use_inner_new_decomp \
    --use_defire_noise $use_defire_noise \
    --use_positional_encoding $use_positional_encoding \
    --use_sostoken $use_sostoken \
    --use_inner_encoder $use_inner_encoder \
        --down_sampling_window 2 \
        --pct_start 0.3 \
> ./run_log/log_202601252150012_win/electricity/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1


done
done

done
