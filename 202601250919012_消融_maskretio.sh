


if [ ! -d "./run_log" ]; then
    mkdir ./run_log
fi
if [ ! -d "./run_log/log_202601250900012_win" ]; then
    mkdir ./run_log/log_202601250900012_win
fi
if [ ! -d "./run_log/log_202601250900012_win/ETTm1" ]; then
    mkdir ./run_log/log_202601250900012_win/ETTm1
fi
if [ ! -d "./run_log/log_202601250900012_win/ETTh1" ]; then
    mkdir ./run_log/log_202601250900012_win/ETTh1
fi
if [ ! -d "./run_log/log_202601250900012_win/ETTm2" ]; then
    mkdir ./run_log/log_202601250900012_win/ETTm2
fi

if [ ! -d "./run_log/log_202601250900012_win/ETTh2" ]; then
    mkdir ./run_log/log_202601250900012_win/ETTh2
fi
if [ ! -d "./run_log/log_202601250900012_win/electricity" ]; then
    mkdir ./run_log/log_202601250900012_win/electricity
fi

if [ ! -d "./run_log/log_202601250900012_win/Exchange" ]; then
    mkdir ./run_log/log_202601250900012_win/Exchange
fi

#if [ ! -d "./run_log/log_202601250900012_win/Solar" ]; then
#    mkdir ./run_log/log_202601250900012_win/Solar
#fi

if [ ! -d "./run_log/log_202601250900012_win/weather" ]; then
    mkdir ./run_log/log_202601250900012_win/weather
fi

if [ ! -d "./run_log/log_202601250900012_win/Traffic" ]; then
    mkdir ./run_log/log_202601250900012_win/Traffic
fi
#
#if [ ! -d "./run_log/log_202601250900012_win/PEMS03" ]; then
#    mkdir ./run_log/log_202601250900012_win/PEMS03
#fi
#
#if [ ! -d "./run_log/log_202601250900012_win/PEMS04" ]; then
#    mkdir ./run_log/log_202601250900012_win/PEMS04
#fi
#
#if [ ! -d "./run_log/log_202601250900012_win/PEMS07" ]; then
#    mkdir ./run_log/log_202601250900012_win/PEMS07
#fi
#if [ ! -d "./run_log/log_202601250900012_win/PEMS08" ]; then
#    mkdir ./run_log/log_202601250900012_win/PEMS08
#fi

#-------------------------------------------------

#----------------------------------------------- version


#




#96->96, 0.370, 0.396,2026-01-24 21:13:10,32,16,16,log_var_orth=6.0,log_var_season_freq=4.0,log_var_freq=0.2,use_loss_compute=1,use_new_decomp=1,use_denoise=1,log_var_season_freq=1,params=models: TimeDART,datasets:ETTh1,seq_len:96,pred_len: 96,macs:102.675M, params:173.684K
#96->192, 0.413, 0.425,2026-01-24 21:15:30,32,16,16,log_var_orth=6.0,log_var_season_freq=4.0,log_var_freq=0.2,use_loss_compute=1,use_new_decomp=1,use_denoise=1,log_var_season_freq=1,params=models: TimeDART,datasets:ETTh1,seq_len:96,pred_len: 192,macs:120.223M, params:330.548K
#96->336, 0.440, 0.442,2026-01-24 21:17:53,32,16,16,log_var_orth=6.0,log_var_season_freq=4.0,log_var_freq=0.2,use_loss_compute=1,use_new_decomp=1,use_denoise=1,log_var_season_freq=1,params=models: TimeDART,datasets:ETTh1,seq_len:96,pred_len: 336,macs:146.543M, params:565.844K
#96->720, 0.445, 0.459,2026-01-24 21:20:12,32,16,16,log_var_orth=6.0,log_var_season_freq=4.0,log_var_freq=0.2,use_loss_compute=1,use_new_decomp=1,use_denoise=1,log_var_season_freq=1,params=models: TimeDART,datasets:ETTh1,seq_len:96,pred_len: 720,macs:216.733M, params:1.193M
#
#96->96, 0.284, 0.336,2026-01-24 21:25:36,8,16,8,log_var_orth=6.0,log_var_season_freq=4.0,log_var_freq=0.2,use_loss_compute=1,use_new_decomp=1,use_denoise=1,log_var_season_freq=1,params=models: TimeDART,datasets:ETTh2,seq_len:96,pred_len: 96,macs:16.401M, params:49.572K
#96->192, 0.362, 0.386,2026-01-24 21:28:10,8,16,8,log_var_orth=6.0,log_var_season_freq=4.0,log_var_freq=0.2,use_loss_compute=1,use_new_decomp=1,use_denoise=1,log_var_season_freq=1,params=models: TimeDART,datasets:ETTh2,seq_len:96,pred_len: 192,macs:21.562M, params:95.844K
#96->336, 0.406, 0.422,2026-01-24 21:30:44,8,16,8,log_var_orth=6.0,log_var_season_freq=4.0,log_var_freq=0.2,use_loss_compute=1,use_new_decomp=1,use_denoise=1,log_var_season_freq=1,params=models: TimeDART,datasets:ETTh2,seq_len:96,pred_len: 336,macs:29.304M, params:165.252K
#96->720, 0.418, 0.439,2026-01-24 21:32:48,8,16,8,log_var_orth=6.0,log_var_season_freq=4.0,log_var_freq=0.2,use_loss_compute=1,use_new_decomp=1,use_denoise=1,log_var_season_freq=1,params=models: TimeDART,datasets:ETTh2,seq_len:96,pred_len: 720,macs:49.948M, params:350.340K
#
#96->96, 0.325, 0.364,2026-01-24 21:46:10,32,64,8,log_var_orth=6.0,log_var_season_freq=4.0,log_var_freq=0.2,use_loss_compute=1,use_new_decomp=1,use_denoise=1,log_var_season_freq=1,params=models: TimeDART,datasets:ETTm1,seq_len:96,pred_len: 96,macs:410.701M, params:173.684K
#96->192, 0.363, 0.385,2026-01-24 21:54:16,32,64,8,log_var_orth=6.0,log_var_season_freq=4.0,log_var_freq=0.2,use_loss_compute=1,use_new_decomp=1,use_denoise=1,log_var_season_freq=1,params=models: TimeDART,datasets:ETTm1,seq_len:96,pred_len: 192,macs:480.890M, params:330.548K
#96->336, 0.389, 0.403,2026-01-24 22:02:12,32,64,8,log_var_orth=6.0,log_var_season_freq=4.0,log_var_freq=0.2,use_loss_compute=1,use_new_decomp=1,use_denoise=1,log_var_season_freq=1,params=models: TimeDART,datasets:ETTm1,seq_len:96,pred_len: 336,macs:586.174M, params:565.844K
#96->720, 0.450, 0.438,2026-01-24 22:10:32,32,64,8,log_var_orth=6.0,log_var_season_freq=4.0,log_var_freq=0.2,use_loss_compute=1,use_new_decomp=1,use_denoise=1,log_var_season_freq=1,params=models: TimeDART,datasets:ETTm1,seq_len:96,pred_len: 720,macs:866.930M, params:1.193M
#
#96->96, 0.179, 0.265,2026-01-24 22:24:09,8,64,8,log_var_orth=6.0,log_var_season_freq=4.0,log_var_freq=0.2,use_loss_compute=1,use_new_decomp=1,use_denoise=1,log_var_season_freq=1,params=models: TimeDART,datasets:ETTm2,seq_len:96,pred_len: 96,macs:54.595M, params:49.028K
#96->192, 0.243, 0.305,2026-01-24 22:32:19,8,64,8,log_var_orth=6.0,log_var_season_freq=4.0,log_var_freq=0.2,use_loss_compute=1,use_new_decomp=1,use_denoise=1,log_var_season_freq=1,params=models: TimeDART,datasets:ETTm2,seq_len:96,pred_len: 192,macs:75.239M, params:95.300K
#96->336, 0.302, 0.343,2026-01-24 22:40:13,8,64,8,log_var_orth=6.0,log_var_season_freq=4.0,log_var_freq=0.2,use_loss_compute=1,use_new_decomp=1,use_denoise=1,log_var_season_freq=1,params=models: TimeDART,datasets:ETTm2,seq_len:96,pred_len: 336,macs:106.205M, params:164.708K
#96->720, 0.401, 0.399,2026-01-24 22:47:53,8,64,8,log_var_orth=6.0,log_var_season_freq=4.0,log_var_freq=0.2,use_loss_compute=1,use_new_decomp=1,use_denoise=1,log_var_season_freq=1,params=models: TimeDART,datasets:ETTm2,seq_len:96,pred_len: 720,macs:188.780M, params:349.796K
#
#96->96, 0.085, 0.204,2026-01-24 22:50:07,32,16,8,log_var_orth=6.0,log_var_season_freq=4.0,log_var_freq=0.2,use_loss_compute=1,use_new_decomp=1,use_denoise=1,log_var_season_freq=1,params=models: TimeDART,datasets:Exchange,seq_len:96,pred_len: 96,macs:117.344M, params:173.952K
#96->192, 0.184, 0.306,2026-01-24 22:50:52,32,16,8,log_var_orth=6.0,log_var_season_freq=4.0,log_var_freq=0.2,use_loss_compute=1,use_new_decomp=1,use_denoise=1,log_var_season_freq=1,params=models: TimeDART,datasets:Exchange,seq_len:96,pred_len: 192,macs:137.398M, params:330.816K
#96->336, 0.333, 0.418,2026-01-24 22:51:51,32,16,8,log_var_orth=6.0,log_var_season_freq=4.0,log_var_freq=0.2,use_loss_compute=1,use_new_decomp=1,use_denoise=1,log_var_season_freq=1,params=models: TimeDART,datasets:Exchange,seq_len:96,pred_len: 336,macs:167.479M, params:566.112K
#96->720, 0.924, 0.726,2026-01-24 22:52:34,32,16,8,log_var_orth=6.0,log_var_season_freq=4.0,log_var_freq=0.2,use_loss_compute=1,use_new_decomp=1,use_denoise=1,log_var_season_freq=1,params=models: TimeDART,datasets:Exchange,seq_len:96,pred_len: 720,macs:247.695M, params:1.194M
#
#96->96, 0.170, 0.212,2026-01-24 23:20:13,64,16,8,log_var_orth=6.0,log_var_season_freq=4.0,log_var_freq=0.2,use_loss_compute=1,use_new_decomp=1,use_denoise=1,log_var_season_freq=1,params=models: TimeDART,datasets:Weather,seq_len:96,pred_len: 96,macs:804.529M, params:352.948K
#96->192, 0.216, 0.254,2026-01-24 23:32:51,64,16,8,log_var_orth=6.0,log_var_season_freq=4.0,log_var_freq=0.2,use_loss_compute=1,use_new_decomp=1,use_denoise=1,log_var_season_freq=1,params=models: TimeDART,datasets:Weather,seq_len:96,pred_len: 192,macs:906.716M, params:657.268K
#96->336, 0.273, 0.294,2026-01-24 23:45:18,64,16,8,log_var_orth=6.0,log_var_season_freq=4.0,log_var_freq=0.2,use_loss_compute=1,use_new_decomp=1,use_denoise=1,log_var_season_freq=1,params=models: TimeDART,datasets:Weather,seq_len:96,pred_len: 336,macs:1.060G, params:1.114M
#96->720, 0.351, 0.345,2026-01-24 23:58:04,64,16,8,log_var_orth=6.0,log_var_season_freq=4.0,log_var_freq=0.2,use_loss_compute=1,use_new_decomp=1,use_denoise=1,log_var_season_freq=1,params=models: TimeDART,datasets:Weather,seq_len:96,pred_len: 720,macs:1.469G, params:2.331M
#
#
#96->96, 0.161, 0.249,2026-01-25 01:26:17,128,16,16,log_var_orth=6.0,log_var_season_freq=4.0,log_var_freq=0.2,use_loss_compute=1,use_new_decomp=1,use_denoise=1,log_var_season_freq=1,params=models: TimeDART,datasets:Electricity,seq_len:96,pred_len: 96,macs:15.418G, params:843.072K
#96->192, 0.171, 0.259,2026-01-25 01:55:24,128,16,16,log_var_orth=6.0,log_var_season_freq=4.0,log_var_freq=0.2,use_loss_compute=1,use_new_decomp=1,use_denoise=1,log_var_season_freq=1,params=models: TimeDART,datasets:Electricity,seq_len:96,pred_len: 192,macs:16.223G, params:999.936K
#96->336, 0.187, 0.276,2026-01-25 02:24:40,128,16,16,log_var_orth=6.0,log_var_season_freq=4.0,log_var_freq=0.2,use_loss_compute=1,use_new_decomp=1,use_denoise=1,log_var_season_freq=1,params=models: TimeDART,datasets:Electricity,seq_len:96,pred_len: 336,macs:17.430G, params:1.235M
#96->720, 0.226, 0.309,2026-01-25 02:54:57,128,16,16,log_var_orth=6.0,log_var_season_freq=4.0,log_var_freq=0.2,use_loss_compute=1,use_new_decomp=1,use_denoise=1,log_var_season_freq=1,params=models: TimeDART,datasets:Electricity,seq_len:96,pred_len: 720,macs:20.648G, params:1.863M


use_defire_noise=0
use_inner_encoder=0
use_positional_encoding=1
use_sostoken=1




use_loss_compute=1
use_new_decomp=1
use_denoise=1
use_inner_new_decomp=1


#mask_ratio=0.3
#mask_ratio=0.8



# 消融  参数：
# 1. use_pretrain_in_ft=1 # 用去噪网络
# 2. film_mode=full、none、random、trend_only、t_only
# 3. use_pretrain_in_ft=0 # 不用去噪网络
# 4. use_new_decomp=1、0
# 5. use_geo_mask=1  使用几何掩码
# 6. predict_eps=1、0 噪声阶段是否用预测噪声





not_context=0
not_cond=1
use_finetune_encoder=1
use_pretrain_encoder=1
use_pretrain_in_ft=0
destroy_mode=season



echo "maskretio"
use_new_decomp=1
film_mode=full

pretrain_mode=mask
mask_ratio=0.41
for mask_ratio in 0.7;do

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
    --not_context $not_context \
    --not_cond $not_cond \
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
> ./run_log/log_202601250900012_win/ETTh1/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1
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
    --not_context $not_context \
    --not_cond $not_cond \
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
> ./run_log/log_202601250900012_win/ETTh1/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1

done

done


# ETTh2 mask
layer_h2=2
use_geo_mask=1
use_geo_mask=0
predict_eps=0

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
    --not_context $not_context \
    --not_cond $not_cond \
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
> ./run_log/log_202601250900012_win/ETTh2/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1


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
    --not_context $not_context \
    --not_cond $not_cond \
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
> ./run_log/log_202601250900012_win/ETTh2/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1


done


done




layer_m1=1
# ETTM1 noise
use_geo_mask=1
use_geo_mask=0
predict_eps=1


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
    --not_context $not_context \
    --not_cond $not_cond \
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
> ./run_log/log_202601250900012_win/ETTm1/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1

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
    --not_context $not_context \
    --not_cond $not_cond \
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
> ./run_log/log_202601250900012_win/ETTm1/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1

done
done


# ETTm2 mask
use_geo_mask=1
use_geo_mask=0
predict_eps=0

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
    --not_context $not_context \
    --not_cond $not_cond \
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
> ./run_log/log_202601250900012_win/ETTm2/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1

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
    --not_context $not_context \
    --not_cond $not_cond \
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
> ./run_log/log_202601250900012_win/ETTm2/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1

done



done



#---------------- Exchange script
# Exchange mask
use_geo_mask=1
use_geo_mask=0
predict_eps=0

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
    --not_context $not_context \
    --not_cond $not_cond \
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
> ./run_log/log_202601250900012_win/Exchange/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1

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
    --not_context $not_context \
    --not_cond $not_cond \
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
> ./run_log/log_202601250900012_win/Exchange/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1

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
    --not_context $not_context \
    --not_cond $not_cond \
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
> ./run_log/log_202601250900012_win/weather/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1



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
    --not_context $not_context \
    --not_cond $not_cond \
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
> ./run_log/log_202601250900012_win/weather/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1

done





done



#---------------- ELE script

# ELE mask
use_geo_mask=1
use_geo_mask=0
predict_eps=0

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
    --not_context $not_context \
    --not_cond $not_cond \
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
> ./run_log/log_202601250900012_win/electricity/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1

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
    --not_context $not_context \
    --not_cond $not_cond \
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
> ./run_log/log_202601250900012_win/electricity/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_mask_ratio'$mask_ratio'_'0.01.log 2>&1


done
done

done







