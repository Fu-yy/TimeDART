if [ ! -d "./run_log" ]; then
    mkdir ./run_log
fi
if [ ! -d "./run_log/log_20257062041_win" ]; then
    mkdir ./run_log/log_20257062041_win
fi
if [ ! -d "./run_log/log_20257062041_win/ETTm1" ]; then
    mkdir ./run_log/log_20257062041_win/ETTm1
fi
if [ ! -d "./run_log/log_20257062041_win/ETTh1" ]; then
    mkdir ./run_log/log_20257062041_win/ETTh1
fi
if [ ! -d "./run_log/log_20257062041_win/ETTm2" ]; then
    mkdir ./run_log/log_20257062041_win/ETTm2
fi

if [ ! -d "./run_log/log_20257062041_win/ETTh2" ]; then
    mkdir ./run_log/log_20257062041_win/ETTh2
fi
if [ ! -d "./run_log/log_20257062041_win/electricity" ]; then
    mkdir ./run_log/log_20257062041_win/electricity
fi

if [ ! -d "./run_log/log_20257062041_win/Exchange" ]; then
    mkdir ./run_log/log_20257062041_win/Exchange
fi

#if [ ! -d "./run_log/log_20257062041_win/Solar" ]; then
#    mkdir ./run_log/log_20257062041_win/Solar
#fi

if [ ! -d "./run_log/log_20257062041_win/weather" ]; then
    mkdir ./run_log/log_20257062041_win/weather
fi

if [ ! -d "./run_log/log_20257062041_win/Traffic" ]; then
    mkdir ./run_log/log_20257062041_win/Traffic
fi
#
#if [ ! -d "./run_log/log_20257062041_win/PEMS03" ]; then
#    mkdir ./run_log/log_20257062041_win/PEMS03
#fi
#
#if [ ! -d "./run_log/log_20257062041_win/PEMS04" ]; then
#    mkdir ./run_log/log_20257062041_win/PEMS04
#fi
#
#if [ ! -d "./run_log/log_20257062041_win/PEMS07" ]; then
#    mkdir ./run_log/log_20257062041_win/PEMS07
#fi
#if [ ! -d "./run_log/log_20257062041_win/PEMS08" ]; then
#    mkdir ./run_log/log_20257062041_win/PEMS08
#fi




#上个个版本  2213
#这个版本 3322










use_init_loss=1
#------------------------
#
del_orth_loss=0
del_season_freq_loss=1
del_smoothness_loss=1
del_freq_loss=1
del_recon_loss=1
n_heads=16

for denoise_layers_num in 1;do





log_var_freq=0.2
log_var_orth=6.0
log_var_smooth=0.0
log_var_season_freq=4.0
log_var_recon=0.0

log_var_freq=0.3
log_var_orth=0.01
log_var_smooth=0.0
log_var_season_freq=0.05
log_var_recon=0.0
##ecl
#
#batch_size=16
#d_model=128
for log_var_orth in 0.0001 0.01 1 5 10 100;do
batch_size=8
d_model=64
echo "ECL $denoise_layers_num "

python -u run.py \
    --task_name pretrain \
    --root_path ./datasets/electricity/ \
    --data_path electricity.csv \
    --model_id Electricity \
    --model TimeDART \
    --data Electricity \
    --features M \
    --input_len 336 \
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
    --scheduler cosine \
    --lr_decay 0.95 \
    --learning_rate 0.0001 \
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
    --batch_size $batch_size \
    --use_init_loss $use_init_loss \
    --down_sampling_window 2 \
    --train_epochs 10 \
> ./run_log/log_20257062041_win/electricity/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_'0.01.log 2>&1

#batch_size=16
#d_model=128
batch_size=8
d_model=64
#n_heads = 16

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
        --input_len 336 \
        --label_len 48 \
        --denoise_layers_num $denoise_layers_num \
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
        --del_season_freq_loss $del_season_freq_loss \
        --del_smoothness_loss $del_smoothness_loss \
        --del_freq_loss $del_freq_loss \
        --del_recon_loss $del_recon_loss \
        --log_var_recon $log_var_recon \
        --log_var_freq $log_var_freq \
        --log_var_orth $log_var_orth \
        --log_var_smooth $log_var_smooth \
        --log_var_season_freq $log_var_season_freq \
        --scheduler cosine \
        --patience 3 \
        --learning_rate 0.0004 \
        --use_init_loss $use_init_loss \
        --down_sampling_window 2 \
        --pct_start 0.3 \
> ./run_log/log_20257062041_win/electricity/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_'0.01.log 2>&1

done

done
done

