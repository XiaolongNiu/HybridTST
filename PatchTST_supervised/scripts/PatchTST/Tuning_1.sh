if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
#!/bin/bash

seq_len=288
model_name=patch_towerTST
root_path_name=/data/niuxl27/patchtst_two/PatchTST_supervised/dataset
data_path_name=weather.csv
model_id_name=weather
data_name=custom

random_seed=2021
dropout_range="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9"
fc_dropout_range="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9"
head_dropout_range="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9"
con_dropout_range="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9"
best_val_loss="99"
best_iter="0"
# Create a file to record parameter combinations and validation losses
result_file="dropout_tuning_results.txt"
echo "Iteration Dropout FC_Dropout Head_Dropout Con_Dropout Validation_Loss" > $result_file

for pred_len in 48 
do
    for i in {1..10}  # Number of random search iterations
    do
        echo "Iteration $i"

        # Randomly sample dropout parameters from the defined ranges
        dropout=$(echo $dropout_range | tr ' ' '\n' | shuf -n 1)
        fc_dropout=$(echo $fc_dropout_range | tr ' ' '\n' | shuf -n 1)
        head_dropout=$(echo $head_dropout_range | tr ' ' '\n' | shuf -n 1)
        con_dropout=$(echo $con_dropout_range | tr ' ' '\n' | shuf -n 1)

        # Run your Python script with the sampled dropout parameters
        python -u run_longExp_two.py \
            --random_seed $random_seed \
            --is_training 1 \
            --root_path $root_path_name \
            --data_path $data_path_name \
            --model_id "${model_id_name}_${seq_len}_${pred_len}_${i}" \
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
            --dropout $dropout \
            --fc_dropout $fc_dropout \
            --head_dropout $head_dropout \
            --con_dropout $con_dropout   \
            --patch_len 16 \
            --stride 8 \
            --des 'Exp' \
            --train_epochs 10 \
            --two_tower_col 21 \
            --patience 20 \
            --itr 1 --batch_size 8 --learning_rate 0.0001 >logs/LongForecasting/${model_name}_${model_id_name}_${seq_len}_${pred_len}_${i}.log 

        # Extract validation loss from log file
        val_loss=$(grep "Vali Loss:" logs/LongForecasting/${model_name}_${model_id_name}_${seq_len}_${pred_len}_${i}.log | awk -F 'Vali Loss:' '{print $2}' | awk '{print $1}' | tail -n 1)
        echo $val_loss
        # Record the parameters and validation loss to the result file
        echo "$i $dropout $fc_dropout $head_dropout $con_dropout $val_loss" >> $result_file
        echo $best_val_loss
        # Update best validation loss and dropout parameters if current iteration has lower validation loss
        if (( $(echo "$val_loss < $best_val_loss" | bc -l) == 1 )); then
            best_val_loss=$val_loss
            best_dropout="$dropout $fc_dropout $head_dropout $con_dropout"
            best_iter=$i
        fi
    done

    # Print best dropout parameters for this prediction length
    echo "Best dropout parameters for pred_len $pred_len: $best_dropout"
done

# Print the overall best dropout parameters
echo "Overall best dropout parameters: $best_dropout at $best_iter"
