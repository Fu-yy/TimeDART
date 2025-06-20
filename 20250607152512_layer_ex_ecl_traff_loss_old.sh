if [ ! -d "./run_log" ]; then
    mkdir ./run_log
fi
if [ ! -d "./run_log/log_20250607152945_win" ]; then
    mkdir ./run_log/log_20250607152945_win
fi
if [ ! -d "./run_log/log_20250607152945_win/ETTm1" ]; then
    mkdir ./run_log/log_20250607152945_win/ETTm1
fi
if [ ! -d "./run_log/log_20250607152945_win/ETTh1" ]; then
    mkdir ./run_log/log_20250607152945_win/ETTh1
fi
if [ ! -d "./run_log/log_20250607152945_win/ETTm2" ]; then
    mkdir ./run_log/log_20250607152945_win/ETTm2
fi

if [ ! -d "./run_log/log_20250607152945_win/ETTh2" ]; then
    mkdir ./run_log/log_20250607152945_win/ETTh2
fi
if [ ! -d "./run_log/log_20250607152945_win/electricity" ]; then
    mkdir ./run_log/log_20250607152945_win/electricity
fi

if [ ! -d "./run_log/log_20250607152945_win/Exchange" ]; then
    mkdir ./run_log/log_20250607152945_win/Exchange
fi

#if [ ! -d "./run_log/log_20250607152945_win/Solar" ]; then
#    mkdir ./run_log/log_20250607152945_win/Solar
#fi

if [ ! -d "./run_log/log_20250607152945_win/weather" ]; then
    mkdir ./run_log/log_20250607152945_win/weather
fi

if [ ! -d "./run_log/log_20250607152945_win/Traffic" ]; then
    mkdir ./run_log/log_20250607152945_win/Traffic
fi
#
#if [ ! -d "./run_log/log_20250607152945_win/PEMS03" ]; then
#    mkdir ./run_log/log_20250607152945_win/PEMS03
#fi
#
#if [ ! -d "./run_log/log_20250607152945_win/PEMS04" ]; then
#    mkdir ./run_log/log_20250607152945_win/PEMS04
#fi
#
#if [ ! -d "./run_log/log_20250607152945_win/PEMS07" ]; then
#    mkdir ./run_log/log_20250607152945_win/PEMS07
#fi
#if [ ! -d "./run_log/log_20250607152945_win/PEMS08" ]; then
#    mkdir ./run_log/log_20250607152945_win/PEMS08
#fi




#上个个版本  2213
#这个版本 3322










use_init_loss=0
#------------------------

del_orth_loss=0
del_season_freq_loss=0
del_smoothness_loss=0
del_freq_loss=0
del_recon_loss=0

for denoise_layers_num in 1;do

log_var_freq=0.5
log_var_orth=6.0
log_var_smooth=1.0
log_var_season_freq=4.0
log_var_recon=0.1

#echo "Exchange $denoise_layers_num"
#
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
#    --del_recon_loss $del_recon_loss \
#    --log_var_recon $log_var_recon \
#    --log_var_freq $log_var_freq \
#    --log_var_orth $log_var_orth \
#    --log_var_smooth $log_var_smooth \
#    --log_var_season_freq $log_var_season_freq \
#    --use_init_loss $use_init_loss \
#    --down_sampling_window 2 \
#> ./run_log/log_20250607152945_win/Exchange/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_'0.01.log 2>&1
#
#
#for pred_len in 96 192 336 720; do
#echo "Exchange $denoise_layers_num _ $pred_len"
#
#    python -u run.py \
#        --task_name finetune \
#        --is_training 1 \
#        --root_path ./datasets/exchange_rate/ \
#        --data_path exchange_rate.csv \
#        --model_id Exchange \
#        --model TimeDART \
#        --data Exchange \
#        --features M \
#        --input_len 336 \
#        --label_len 48 \
#        --pred_len $pred_len \
#        --e_layers 2 \
#        --denoise_layers_num $denoise_layers_num \
#        --enc_in 8 \
#        --dec_in 8 \
#        --c_out 8 \
#        --n_heads 8 \
#        --d_model 256 \
#        --d_ff 64 \
#        --patch_len 2 \
#        --stride 2 \
#        --dropout 0.2 \
#        --head_dropout 0.1 \
#        --batch_size 16 \
#        --lr_decay 0.8 \
#        --lradj step \
#        --time_steps 1000 \
#        --scheduler cosine \
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
#        --patience 3 \
#        --learning_rate 0.0001 \
#        --pct_start 0.3 \
#       --use_init_loss $use_init_loss \
#       --down_sampling_window 2 \
#> ./run_log/log_20250607152945_win/Exchange/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_'0.01.log 2>&1
#
#done






log_var_freq=0.2
log_var_orth=6.0
log_var_smooth=1.0
log_var_season_freq=4.0
log_var_recon=0.1
##ecl
#
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
    --n_heads 16 \
    --d_model 64 \
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
    --batch_size 16 \
    --use_init_loss $use_init_loss \
    --down_sampling_window 2 \
    --train_epochs 50 \
> ./run_log/log_20250607152945_win/electricity/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_'0.01.log 2>&1

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
        --n_heads 16 \
        --d_model 64 \
        --d_ff 256 \
        --patch_len 8 \
        --stride 8 \
        --dropout 0.2 \
        --head_dropout 0.0 \
        --batch_size 16 \
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
> ./run_log/log_20250607152945_win/electricity/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_'0.01.log 2>&1

done




log_var_freq=0.2
log_var_orth=6.0
log_var_smooth=1.0
log_var_season_freq=4.0
log_var_recon=0.1
#Traffic
echo "Traffic $denoise_layers_num "

python -u run.py \
    --task_name pretrain \
    --root_path ./datasets/traffic/ \
    --data_path traffic.csv \
    --model_id Traffic \
    --model TimeDART \
    --data Traffic \
    --features M \
    --input_len 336 \
    --e_layers 3 \
    --d_layers 1 \
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862 \
    --n_heads 16 \
    --d_model 32 \
    --d_ff 128 \
    --patch_len 8 \
    --stride 8 \
    --head_dropout 0.1 \
    --denoise_layers_num $denoise_layers_num \
    --dropout 0.2 \
    --time_steps 1000 \
    --scheduler cosine \
    --lr_decay 0.95 \
    --learning_rate 0.0001 \
    --batch_size 8 \
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
    --train_epochs 50 \
    --use_init_loss $use_init_loss \
    --down_sampling_window 2 \
> ./run_log/log_20250607152945_win/Traffic/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_'0.01.log 2>&1



for pred_len in 96 192 336 720; do
echo "Traffic $denoise_layers_num _ $pred_len "

    python -u run.py \
        --task_name finetune \
        --is_training 1 \
        --root_path ./datasets/traffic/ \
        --data_path traffic.csv \
        --model_id Traffic \
        --model TimeDART \
        --data Traffic \
        --features M \
        --input_len 336 \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 3 \
        --enc_in 862 \
        --dec_in 862 \
        --c_out 862 \
        --n_heads 16 \
        --d_model 32 \
        --d_ff 128 \
        --patch_len 8 \
        --stride 8 \
        --dropout 0.2 \
        --head_dropout 0.1 \
        --batch_size 8 \
        --denoise_layers_num $denoise_layers_num \
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
        --learning_rate 0.003 \
        --pct_start 0.2 \
      --use_init_loss $use_init_loss \
      --down_sampling_window 2 \
> ./run_log/log_20250607152945_win/Traffic/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_'0.01.log 2>&1

done


done


#------------------------
del_orth_loss=1
del_season_freq_loss=0
del_smoothness_loss=0
del_freq_loss=0
del_recon_loss=0



for denoise_layers_num in 1;do

log_var_freq=0.5
log_var_orth=6.0
log_var_smooth=1.0
log_var_season_freq=4.0
log_var_recon=0.1

#echo "Exchange $denoise_layers_num"
#
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
#    --del_recon_loss $del_recon_loss \
#    --log_var_recon $log_var_recon \
#    --log_var_freq $log_var_freq \
#    --log_var_orth $log_var_orth \
#    --log_var_smooth $log_var_smooth \
#    --log_var_season_freq $log_var_season_freq \
#    --use_init_loss $use_init_loss \
#    --down_sampling_window 2 \
#> ./run_log/log_20250607152945_win/Exchange/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_'0.01.log 2>&1
#
#
#for pred_len in 96 192 336 720; do
#echo "Exchange $denoise_layers_num _ $pred_len"
#
#    python -u run.py \
#        --task_name finetune \
#        --is_training 1 \
#        --root_path ./datasets/exchange_rate/ \
#        --data_path exchange_rate.csv \
#        --model_id Exchange \
#        --model TimeDART \
#        --data Exchange \
#        --features M \
#        --input_len 336 \
#        --label_len 48 \
#        --pred_len $pred_len \
#        --e_layers 2 \
#        --denoise_layers_num $denoise_layers_num \
#        --enc_in 8 \
#        --dec_in 8 \
#        --c_out 8 \
#        --n_heads 8 \
#        --d_model 256 \
#        --d_ff 64 \
#        --patch_len 2 \
#        --stride 2 \
#        --dropout 0.2 \
#        --head_dropout 0.1 \
#        --batch_size 16 \
#        --lr_decay 0.8 \
#        --lradj step \
#        --time_steps 1000 \
#        --scheduler cosine \
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
#        --patience 3 \
#        --learning_rate 0.0001 \
#        --pct_start 0.3 \
#       --use_init_loss $use_init_loss \
#       --down_sampling_window 2 \
#> ./run_log/log_20250607152945_win/Exchange/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_'0.01.log 2>&1
#
#done






log_var_freq=0.2
log_var_orth=6.0
log_var_smooth=1.0
log_var_season_freq=4.0
log_var_recon=0.1
##ecl
#
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
    --n_heads 16 \
    --d_model 64 \
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
    --batch_size 16 \
    --use_init_loss $use_init_loss \
    --down_sampling_window 2 \
    --train_epochs 50 \
> ./run_log/log_20250607152945_win/electricity/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_'0.01.log 2>&1

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
        --n_heads 16 \
        --d_model 64 \
        --d_ff 256 \
        --patch_len 8 \
        --stride 8 \
        --dropout 0.2 \
        --head_dropout 0.0 \
        --batch_size 16 \
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
> ./run_log/log_20250607152945_win/electricity/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_'0.01.log 2>&1

done




log_var_freq=0.2
log_var_orth=6.0
log_var_smooth=1.0
log_var_season_freq=4.0
log_var_recon=0.1
#Traffic
echo "Traffic $denoise_layers_num "

python -u run.py \
    --task_name pretrain \
    --root_path ./datasets/traffic/ \
    --data_path traffic.csv \
    --model_id Traffic \
    --model TimeDART \
    --data Traffic \
    --features M \
    --input_len 336 \
    --e_layers 3 \
    --d_layers 1 \
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862 \
    --n_heads 16 \
    --d_model 32 \
    --d_ff 128 \
    --patch_len 8 \
    --stride 8 \
    --head_dropout 0.1 \
    --denoise_layers_num $denoise_layers_num \
    --dropout 0.2 \
    --time_steps 1000 \
    --scheduler cosine \
    --lr_decay 0.95 \
    --learning_rate 0.0001 \
    --batch_size 8 \
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
    --train_epochs 50 \
    --use_init_loss $use_init_loss \
    --down_sampling_window 2 \
> ./run_log/log_20250607152945_win/Traffic/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_'0.01.log 2>&1



for pred_len in 96 192 336 720; do
echo "Traffic $denoise_layers_num _ $pred_len "

    python -u run.py \
        --task_name finetune \
        --is_training 1 \
        --root_path ./datasets/traffic/ \
        --data_path traffic.csv \
        --model_id Traffic \
        --model TimeDART \
        --data Traffic \
        --features M \
        --input_len 336 \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 3 \
        --enc_in 862 \
        --dec_in 862 \
        --c_out 862 \
        --n_heads 16 \
        --d_model 32 \
        --d_ff 128 \
        --patch_len 8 \
        --stride 8 \
        --dropout 0.2 \
        --head_dropout 0.1 \
        --batch_size 8 \
        --denoise_layers_num $denoise_layers_num \
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
        --learning_rate 0.003 \
        --pct_start 0.2 \
      --use_init_loss $use_init_loss \
      --down_sampling_window 2 \
> ./run_log/log_20250607152945_win/Traffic/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_'0.01.log 2>&1

done


done




#------------------------
del_orth_loss=0
del_season_freq_loss=1
del_smoothness_loss=0
del_freq_loss=0
del_recon_loss=0



for denoise_layers_num in 1;do

log_var_freq=0.5
log_var_orth=6.0
log_var_smooth=1.0
log_var_season_freq=4.0
log_var_recon=0.1

#echo "Exchange $denoise_layers_num"
#
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
#    --del_recon_loss $del_recon_loss \
#    --log_var_recon $log_var_recon \
#    --log_var_freq $log_var_freq \
#    --log_var_orth $log_var_orth \
#    --log_var_smooth $log_var_smooth \
#    --log_var_season_freq $log_var_season_freq \
#    --use_init_loss $use_init_loss \
#    --down_sampling_window 2 \
#> ./run_log/log_20250607152945_win/Exchange/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_'0.01.log 2>&1
#
#
#for pred_len in 96 192 336 720; do
#echo "Exchange $denoise_layers_num _ $pred_len"
#
#    python -u run.py \
#        --task_name finetune \
#        --is_training 1 \
#        --root_path ./datasets/exchange_rate/ \
#        --data_path exchange_rate.csv \
#        --model_id Exchange \
#        --model TimeDART \
#        --data Exchange \
#        --features M \
#        --input_len 336 \
#        --label_len 48 \
#        --pred_len $pred_len \
#        --e_layers 2 \
#        --denoise_layers_num $denoise_layers_num \
#        --enc_in 8 \
#        --dec_in 8 \
#        --c_out 8 \
#        --n_heads 8 \
#        --d_model 256 \
#        --d_ff 64 \
#        --patch_len 2 \
#        --stride 2 \
#        --dropout 0.2 \
#        --head_dropout 0.1 \
#        --batch_size 16 \
#        --lr_decay 0.8 \
#        --lradj step \
#        --time_steps 1000 \
#        --scheduler cosine \
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
#        --patience 3 \
#        --learning_rate 0.0001 \
#        --pct_start 0.3 \
#       --use_init_loss $use_init_loss \
#       --down_sampling_window 2 \
#> ./run_log/log_20250607152945_win/Exchange/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_'0.01.log 2>&1
#
#done






log_var_freq=0.2
log_var_orth=6.0
log_var_smooth=1.0
log_var_season_freq=4.0
log_var_recon=0.1
##ecl
#
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
    --n_heads 16 \
    --d_model 64 \
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
    --batch_size 16 \
    --use_init_loss $use_init_loss \
    --down_sampling_window 2 \
    --train_epochs 50 \
> ./run_log/log_20250607152945_win/electricity/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_'0.01.log 2>&1

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
        --n_heads 16 \
        --d_model 64 \
        --d_ff 256 \
        --patch_len 8 \
        --stride 8 \
        --dropout 0.2 \
        --head_dropout 0.0 \
        --batch_size 16 \
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
> ./run_log/log_20250607152945_win/electricity/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_'0.01.log 2>&1

done




log_var_freq=0.2
log_var_orth=6.0
log_var_smooth=1.0
log_var_season_freq=4.0
log_var_recon=0.1
#Traffic
echo "Traffic $denoise_layers_num "

python -u run.py \
    --task_name pretrain \
    --root_path ./datasets/traffic/ \
    --data_path traffic.csv \
    --model_id Traffic \
    --model TimeDART \
    --data Traffic \
    --features M \
    --input_len 336 \
    --e_layers 3 \
    --d_layers 1 \
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862 \
    --n_heads 16 \
    --d_model 32 \
    --d_ff 128 \
    --patch_len 8 \
    --stride 8 \
    --head_dropout 0.1 \
    --denoise_layers_num $denoise_layers_num \
    --dropout 0.2 \
    --time_steps 1000 \
    --scheduler cosine \
    --lr_decay 0.95 \
    --learning_rate 0.0001 \
    --batch_size 8 \
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
    --train_epochs 50 \
    --use_init_loss $use_init_loss \
    --down_sampling_window 2 \
> ./run_log/log_20250607152945_win/Traffic/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_'0.01.log 2>&1



for pred_len in 96 192 336 720; do
echo "Traffic $denoise_layers_num _ $pred_len "

    python -u run.py \
        --task_name finetune \
        --is_training 1 \
        --root_path ./datasets/traffic/ \
        --data_path traffic.csv \
        --model_id Traffic \
        --model TimeDART \
        --data Traffic \
        --features M \
        --input_len 336 \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 3 \
        --enc_in 862 \
        --dec_in 862 \
        --c_out 862 \
        --n_heads 16 \
        --d_model 32 \
        --d_ff 128 \
        --patch_len 8 \
        --stride 8 \
        --dropout 0.2 \
        --head_dropout 0.1 \
        --batch_size 8 \
        --denoise_layers_num $denoise_layers_num \
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
        --learning_rate 0.003 \
        --pct_start 0.2 \
      --use_init_loss $use_init_loss \
      --down_sampling_window 2 \
> ./run_log/log_20250607152945_win/Traffic/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_'0.01.log 2>&1

done


done




#------------------------
del_orth_loss=0
del_season_freq_loss=0
del_smoothness_loss=1
del_freq_loss=0
del_recon_loss=0



for denoise_layers_num in 1;do

log_var_freq=0.5
log_var_orth=6.0
log_var_smooth=1.0
log_var_season_freq=4.0
log_var_recon=0.1

#echo "Exchange $denoise_layers_num"
#
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
#    --del_recon_loss $del_recon_loss \
#    --log_var_recon $log_var_recon \
#    --log_var_freq $log_var_freq \
#    --log_var_orth $log_var_orth \
#    --log_var_smooth $log_var_smooth \
#    --log_var_season_freq $log_var_season_freq \
#    --use_init_loss $use_init_loss \
#    --down_sampling_window 2 \
#> ./run_log/log_20250607152945_win/Exchange/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_'0.01.log 2>&1
#
#
#for pred_len in 96 192 336 720; do
#echo "Exchange $denoise_layers_num _ $pred_len"
#
#    python -u run.py \
#        --task_name finetune \
#        --is_training 1 \
#        --root_path ./datasets/exchange_rate/ \
#        --data_path exchange_rate.csv \
#        --model_id Exchange \
#        --model TimeDART \
#        --data Exchange \
#        --features M \
#        --input_len 336 \
#        --label_len 48 \
#        --pred_len $pred_len \
#        --e_layers 2 \
#        --denoise_layers_num $denoise_layers_num \
#        --enc_in 8 \
#        --dec_in 8 \
#        --c_out 8 \
#        --n_heads 8 \
#        --d_model 256 \
#        --d_ff 64 \
#        --patch_len 2 \
#        --stride 2 \
#        --dropout 0.2 \
#        --head_dropout 0.1 \
#        --batch_size 16 \
#        --lr_decay 0.8 \
#        --lradj step \
#        --time_steps 1000 \
#        --scheduler cosine \
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
#        --patience 3 \
#        --learning_rate 0.0001 \
#        --pct_start 0.3 \
#       --use_init_loss $use_init_loss \
#       --down_sampling_window 2 \
#> ./run_log/log_20250607152945_win/Exchange/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_'0.01.log 2>&1
#
#done






log_var_freq=0.2
log_var_orth=6.0
log_var_smooth=1.0
log_var_season_freq=4.0
log_var_recon=0.1
##ecl
#
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
    --n_heads 16 \
    --d_model 64 \
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
    --batch_size 16 \
    --use_init_loss $use_init_loss \
    --down_sampling_window 2 \
    --train_epochs 50 \
> ./run_log/log_20250607152945_win/electricity/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_'0.01.log 2>&1

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
        --n_heads 16 \
        --d_model 64 \
        --d_ff 256 \
        --patch_len 8 \
        --stride 8 \
        --dropout 0.2 \
        --head_dropout 0.0 \
        --batch_size 16 \
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
> ./run_log/log_20250607152945_win/electricity/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_'0.01.log 2>&1

done




log_var_freq=0.2
log_var_orth=6.0
log_var_smooth=1.0
log_var_season_freq=4.0
log_var_recon=0.1
#Traffic
echo "Traffic $denoise_layers_num "

python -u run.py \
    --task_name pretrain \
    --root_path ./datasets/traffic/ \
    --data_path traffic.csv \
    --model_id Traffic \
    --model TimeDART \
    --data Traffic \
    --features M \
    --input_len 336 \
    --e_layers 3 \
    --d_layers 1 \
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862 \
    --n_heads 16 \
    --d_model 32 \
    --d_ff 128 \
    --patch_len 8 \
    --stride 8 \
    --head_dropout 0.1 \
    --denoise_layers_num $denoise_layers_num \
    --dropout 0.2 \
    --time_steps 1000 \
    --scheduler cosine \
    --lr_decay 0.95 \
    --learning_rate 0.0001 \
    --batch_size 8 \
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
    --train_epochs 50 \
    --use_init_loss $use_init_loss \
    --down_sampling_window 2 \
> ./run_log/log_20250607152945_win/Traffic/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_'0.01.log 2>&1



for pred_len in 96 192 336 720; do
echo "Traffic $denoise_layers_num _ $pred_len "

    python -u run.py \
        --task_name finetune \
        --is_training 1 \
        --root_path ./datasets/traffic/ \
        --data_path traffic.csv \
        --model_id Traffic \
        --model TimeDART \
        --data Traffic \
        --features M \
        --input_len 336 \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 3 \
        --enc_in 862 \
        --dec_in 862 \
        --c_out 862 \
        --n_heads 16 \
        --d_model 32 \
        --d_ff 128 \
        --patch_len 8 \
        --stride 8 \
        --dropout 0.2 \
        --head_dropout 0.1 \
        --batch_size 8 \
        --denoise_layers_num $denoise_layers_num \
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
        --learning_rate 0.003 \
        --pct_start 0.2 \
      --use_init_loss $use_init_loss \
      --down_sampling_window 2 \
> ./run_log/log_20250607152945_win/Traffic/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_'0.01.log 2>&1

done


done




#------------------------
del_orth_loss=0
del_season_freq_loss=0
del_smoothness_loss=0
del_freq_loss=1
del_recon_loss=0



for denoise_layers_num in 1;do

log_var_freq=0.5
log_var_orth=6.0
log_var_smooth=1.0
log_var_season_freq=4.0
log_var_recon=0.1

#echo "Exchange $denoise_layers_num"
#
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
#    --del_recon_loss $del_recon_loss \
#    --log_var_recon $log_var_recon \
#    --log_var_freq $log_var_freq \
#    --log_var_orth $log_var_orth \
#    --log_var_smooth $log_var_smooth \
#    --log_var_season_freq $log_var_season_freq \
#    --use_init_loss $use_init_loss \
#    --down_sampling_window 2 \
#> ./run_log/log_20250607152945_win/Exchange/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_'0.01.log 2>&1
#
#
#for pred_len in 96 192 336 720; do
#echo "Exchange $denoise_layers_num _ $pred_len"
#
#    python -u run.py \
#        --task_name finetune \
#        --is_training 1 \
#        --root_path ./datasets/exchange_rate/ \
#        --data_path exchange_rate.csv \
#        --model_id Exchange \
#        --model TimeDART \
#        --data Exchange \
#        --features M \
#        --input_len 336 \
#        --label_len 48 \
#        --pred_len $pred_len \
#        --e_layers 2 \
#        --denoise_layers_num $denoise_layers_num \
#        --enc_in 8 \
#        --dec_in 8 \
#        --c_out 8 \
#        --n_heads 8 \
#        --d_model 256 \
#        --d_ff 64 \
#        --patch_len 2 \
#        --stride 2 \
#        --dropout 0.2 \
#        --head_dropout 0.1 \
#        --batch_size 16 \
#        --lr_decay 0.8 \
#        --lradj step \
#        --time_steps 1000 \
#        --scheduler cosine \
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
#        --patience 3 \
#        --learning_rate 0.0001 \
#        --pct_start 0.3 \
#       --use_init_loss $use_init_loss \
#       --down_sampling_window 2 \
#> ./run_log/log_20250607152945_win/Exchange/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_'0.01.log 2>&1
#
#done






log_var_freq=0.2
log_var_orth=6.0
log_var_smooth=1.0
log_var_season_freq=4.0
log_var_recon=0.1
##ecl
#
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
    --n_heads 16 \
    --d_model 64 \
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
    --batch_size 16 \
    --use_init_loss $use_init_loss \
    --down_sampling_window 2 \
    --train_epochs 50 \
> ./run_log/log_20250607152945_win/electricity/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_'0.01.log 2>&1

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
        --n_heads 16 \
        --d_model 64 \
        --d_ff 256 \
        --patch_len 8 \
        --stride 8 \
        --dropout 0.2 \
        --head_dropout 0.0 \
        --batch_size 16 \
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
> ./run_log/log_20250607152945_win/electricity/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_'0.01.log 2>&1

done




log_var_freq=0.2
log_var_orth=6.0
log_var_smooth=1.0
log_var_season_freq=4.0
log_var_recon=0.1
#Traffic
echo "Traffic $denoise_layers_num "

python -u run.py \
    --task_name pretrain \
    --root_path ./datasets/traffic/ \
    --data_path traffic.csv \
    --model_id Traffic \
    --model TimeDART \
    --data Traffic \
    --features M \
    --input_len 336 \
    --e_layers 3 \
    --d_layers 1 \
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862 \
    --n_heads 16 \
    --d_model 32 \
    --d_ff 128 \
    --patch_len 8 \
    --stride 8 \
    --head_dropout 0.1 \
    --denoise_layers_num $denoise_layers_num \
    --dropout 0.2 \
    --time_steps 1000 \
    --scheduler cosine \
    --lr_decay 0.95 \
    --learning_rate 0.0001 \
    --batch_size 8 \
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
    --train_epochs 50 \
    --use_init_loss $use_init_loss \
    --down_sampling_window 2 \
> ./run_log/log_20250607152945_win/Traffic/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_'0.01.log 2>&1



for pred_len in 96 192 336 720; do
echo "Traffic $denoise_layers_num _ $pred_len "

    python -u run.py \
        --task_name finetune \
        --is_training 1 \
        --root_path ./datasets/traffic/ \
        --data_path traffic.csv \
        --model_id Traffic \
        --model TimeDART \
        --data Traffic \
        --features M \
        --input_len 336 \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 3 \
        --enc_in 862 \
        --dec_in 862 \
        --c_out 862 \
        --n_heads 16 \
        --d_model 32 \
        --d_ff 128 \
        --patch_len 8 \
        --stride 8 \
        --dropout 0.2 \
        --head_dropout 0.1 \
        --batch_size 8 \
        --denoise_layers_num $denoise_layers_num \
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
        --learning_rate 0.003 \
        --pct_start 0.2 \
      --use_init_loss $use_init_loss \
      --down_sampling_window 2 \
> ./run_log/log_20250607152945_win/Traffic/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_'0.01.log 2>&1

done


done




#------------------------
del_orth_loss=0
del_season_freq_loss=0
del_smoothness_loss=0
del_freq_loss=0
del_recon_loss=1



for denoise_layers_num in 1;do

log_var_freq=0.5
log_var_orth=6.0
log_var_smooth=1.0
log_var_season_freq=4.0
log_var_recon=0.1

#echo "Exchange $denoise_layers_num"
#
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
#    --del_recon_loss $del_recon_loss \
#    --log_var_recon $log_var_recon \
#    --log_var_freq $log_var_freq \
#    --log_var_orth $log_var_orth \
#    --log_var_smooth $log_var_smooth \
#    --log_var_season_freq $log_var_season_freq \
#    --use_init_loss $use_init_loss \
#    --down_sampling_window 2 \
#> ./run_log/log_20250607152945_win/Exchange/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_'0.01.log 2>&1
#
#
#for pred_len in 96 192 336 720; do
#echo "Exchange $denoise_layers_num _ $pred_len"
#
#    python -u run.py \
#        --task_name finetune \
#        --is_training 1 \
#        --root_path ./datasets/exchange_rate/ \
#        --data_path exchange_rate.csv \
#        --model_id Exchange \
#        --model TimeDART \
#        --data Exchange \
#        --features M \
#        --input_len 336 \
#        --label_len 48 \
#        --pred_len $pred_len \
#        --e_layers 2 \
#        --denoise_layers_num $denoise_layers_num \
#        --enc_in 8 \
#        --dec_in 8 \
#        --c_out 8 \
#        --n_heads 8 \
#        --d_model 256 \
#        --d_ff 64 \
#        --patch_len 2 \
#        --stride 2 \
#        --dropout 0.2 \
#        --head_dropout 0.1 \
#        --batch_size 16 \
#        --lr_decay 0.8 \
#        --lradj step \
#        --time_steps 1000 \
#        --scheduler cosine \
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
#        --patience 3 \
#        --learning_rate 0.0001 \
#        --pct_start 0.3 \
#       --use_init_loss $use_init_loss \
#       --down_sampling_window 2 \
#> ./run_log/log_20250607152945_win/Exchange/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_'0.01.log 2>&1
#
#done






log_var_freq=0.2
log_var_orth=6.0
log_var_smooth=1.0
log_var_season_freq=4.0
log_var_recon=0.1
##ecl
#
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
    --n_heads 16 \
    --d_model 64 \
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
    --batch_size 16 \
    --use_init_loss $use_init_loss \
    --down_sampling_window 2 \
    --train_epochs 50 \
> ./run_log/log_20250607152945_win/electricity/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_'0.01.log 2>&1

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
        --n_heads 16 \
        --d_model 64 \
        --d_ff 256 \
        --patch_len 8 \
        --stride 8 \
        --dropout 0.2 \
        --head_dropout 0.0 \
        --batch_size 16 \
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
> ./run_log/log_20250607152945_win/electricity/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_'0.01.log 2>&1

done




log_var_freq=0.2
log_var_orth=6.0
log_var_smooth=1.0
log_var_season_freq=4.0
log_var_recon=0.1
#Traffic
echo "Traffic $denoise_layers_num "

python -u run.py \
    --task_name pretrain \
    --root_path ./datasets/traffic/ \
    --data_path traffic.csv \
    --model_id Traffic \
    --model TimeDART \
    --data Traffic \
    --features M \
    --input_len 336 \
    --e_layers 3 \
    --d_layers 1 \
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862 \
    --n_heads 16 \
    --d_model 32 \
    --d_ff 128 \
    --patch_len 8 \
    --stride 8 \
    --head_dropout 0.1 \
    --denoise_layers_num $denoise_layers_num \
    --dropout 0.2 \
    --time_steps 1000 \
    --scheduler cosine \
    --lr_decay 0.95 \
    --learning_rate 0.0001 \
    --batch_size 8 \
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
    --train_epochs 50 \
    --use_init_loss $use_init_loss \
    --down_sampling_window 2 \
> ./run_log/log_20250607152945_win/Traffic/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_'0.01.log 2>&1



for pred_len in 96 192 336 720; do
echo "Traffic $denoise_layers_num _ $pred_len "

    python -u run.py \
        --task_name finetune \
        --is_training 1 \
        --root_path ./datasets/traffic/ \
        --data_path traffic.csv \
        --model_id Traffic \
        --model TimeDART \
        --data Traffic \
        --features M \
        --input_len 336 \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 3 \
        --enc_in 862 \
        --dec_in 862 \
        --c_out 862 \
        --n_heads 16 \
        --d_model 32 \
        --d_ff 128 \
        --patch_len 8 \
        --stride 8 \
        --dropout 0.2 \
        --head_dropout 0.1 \
        --batch_size 8 \
        --denoise_layers_num $denoise_layers_num \
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
        --learning_rate 0.003 \
        --pct_start 0.2 \
      --use_init_loss $use_init_loss \
      --down_sampling_window 2 \
> ./run_log/log_20250607152945_win/Traffic/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_'0.01.log 2>&1

done


done




#------------------------
del_orth_loss=1
del_season_freq_loss=1
del_smoothness_loss=1
del_freq_loss=1
del_recon_loss=1



for denoise_layers_num in 1;do

log_var_freq=0.5
log_var_orth=6.0
log_var_smooth=1.0
log_var_season_freq=4.0
log_var_recon=0.1

echo "Exchange $denoise_layers_num"

python -u run.py \
    --task_name pretrain \
    --root_path ./datasets/exchange_rate/ \
    --data_path exchange_rate.csv \
    --model_id Exchange \
    --model TimeDART \
    --data Exchange \
    --features M \
    --input_len 336 \
    --e_layers 2 \
    --d_layers 1 \
    --enc_in 8 \
    --dec_in 8 \
    --denoise_layers_num $denoise_layers_num \
    --c_out 8 \
    --n_heads 8 \
    --d_model 32 \
    --d_ff 64 \
    --patch_len 2 \
    --stride 2 \
    --head_dropout 0.1 \
    --dropout 0.2 \
    --time_steps 1000 \
    --scheduler cosine \
    --lr_decay 0.95 \
    --learning_rate 0.001 \
    --batch_size 16 \
    --train_epochs 50 \
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
    --use_init_loss $use_init_loss \
    --down_sampling_window 2 \
> ./run_log/log_20250607152945_win/Exchange/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_'0.01.log 2>&1


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
        --n_heads 8 \
        --d_model 256 \
        --d_ff 64 \
        --patch_len 2 \
        --stride 2 \
        --dropout 0.2 \
        --head_dropout 0.1 \
        --batch_size 16 \
        --lr_decay 0.8 \
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
       --use_init_loss $use_init_loss \
       --down_sampling_window 2 \
> ./run_log/log_20250607152945_win/Exchange/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_'0.01.log 2>&1

done






log_var_freq=0.2
log_var_orth=6.0
log_var_smooth=1.0
log_var_season_freq=4.0
log_var_recon=0.1
##ecl
#
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
    --n_heads 16 \
    --d_model 64 \
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
    --batch_size 16 \
    --use_init_loss $use_init_loss \
    --down_sampling_window 2 \
    --train_epochs 50 \
> ./run_log/log_20250607152945_win/electricity/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_'0.01.log 2>&1

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
        --n_heads 16 \
        --d_model 64 \
        --d_ff 256 \
        --patch_len 8 \
        --stride 8 \
        --dropout 0.2 \
        --head_dropout 0.0 \
        --batch_size 16 \
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
> ./run_log/log_20250607152945_win/electricity/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_'0.01.log 2>&1

done




log_var_freq=0.2
log_var_orth=6.0
log_var_smooth=1.0
log_var_season_freq=4.0
log_var_recon=0.1
#Traffic
echo "Traffic $denoise_layers_num "

python -u run.py \
    --task_name pretrain \
    --root_path ./datasets/traffic/ \
    --data_path traffic.csv \
    --model_id Traffic \
    --model TimeDART \
    --data Traffic \
    --features M \
    --input_len 336 \
    --e_layers 3 \
    --d_layers 1 \
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862 \
    --n_heads 16 \
    --d_model 32 \
    --d_ff 128 \
    --patch_len 8 \
    --stride 8 \
    --head_dropout 0.1 \
    --denoise_layers_num $denoise_layers_num \
    --dropout 0.2 \
    --time_steps 1000 \
    --scheduler cosine \
    --lr_decay 0.95 \
    --learning_rate 0.0001 \
    --batch_size 8 \
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
    --train_epochs 50 \
    --use_init_loss $use_init_loss \
    --down_sampling_window 2 \
> ./run_log/log_20250607152945_win/Traffic/'TimeDART_pretrain_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_'0.01.log 2>&1



for pred_len in 96 192 336 720; do
echo "Traffic $denoise_layers_num _ $pred_len "

    python -u run.py \
        --task_name finetune \
        --is_training 1 \
        --root_path ./datasets/traffic/ \
        --data_path traffic.csv \
        --model_id Traffic \
        --model TimeDART \
        --data Traffic \
        --features M \
        --input_len 336 \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 3 \
        --enc_in 862 \
        --dec_in 862 \
        --c_out 862 \
        --n_heads 16 \
        --d_model 32 \
        --d_ff 128 \
        --patch_len 8 \
        --stride 8 \
        --dropout 0.2 \
        --head_dropout 0.1 \
        --batch_size 8 \
        --denoise_layers_num $denoise_layers_num \
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
        --learning_rate 0.003 \
        --pct_start 0.2 \
      --use_init_loss $use_init_loss \
      --down_sampling_window 2 \
> ./run_log/log_20250607152945_win/Traffic/'TimeDART_finetune'$pred_len'_'$denoise_layers_num'_f'$del_freq_loss'_'$log_var_freq'_o'$del_orth_loss'_'$log_var_orth'_s'$del_smoothness_loss'_'$log_var_smooth'_sea'$del_season_freq_loss'_'$log_var_season_freq'_r'$del_recon_loss'_'$log_var_recon'_'0.01.log 2>&1

done


done




