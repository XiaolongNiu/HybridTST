if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=288
model_name=patch_towerTST
# model_name=PatchTST
root_path_name=/home/jmz/Workstation/niuxl27/patchtst_two/PatchTST_supervised/dataset
# root_path_name=/home/jmz/Workstation/niuxl27/patchtst_two/PatchTST_supervised/dataset/ETT-small/
data_path_name=huangshan_data_fill_count_in_feature_eng_all.csv
model_id_name=huangshan
data_name=custom

random_seed=2021
for pred_len in 144 288 576 864
do
    python -u patchtst_supervised.py \
      --is_train 1 \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --dset huangshan \
      --context_points $seq_len \
      --target_points $pred_len \
      --n_layers 3 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --n_epochs 10\
      --batch_size 24 --lr 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done
