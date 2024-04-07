if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=288
model_name=patch_towerTST
# model_name=PatchTST
# root_path_name=/home/jmz/Workstation/niuxl27/patchtst_two/PatchTST_supervised/dataset
root_path_name=/data/working_projects/niuxl27/patchtst_two/PatchTST_supervised/dataset
# root_path_name=/home/jmz/Workstation/niuxl27/patchtst_two/PatchTST_supervised/dataset/ETT-small/
data_path_name=traffic.csv
model_id_name=traffic
data_name=custom

random_seed=2021
for pred_len in 48 96 196 288
# for pred_len in 144
do
    python -u run_longExp_two.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 862 \
      --e_layers 3 \
      --n_heads 4 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.3\
      --fc_dropout 0.2\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 100\
      --two_tower_col 862 \
      --con_dropout  \
      --patience 20\
      --itr 1 --batch_size 8 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done
