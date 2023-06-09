cd "$(dirname $0)"

python ../finetune.py \
    --base_model '/path/to/model/chinese-alpaca-lora-13b' \
    --data_path '/path/to/data/firefly/firefly-train-0.json' \
    --output_dir '/path/to/output/run-2-alpaca-13b-1' \
    --batch_size 256 \
    --micro_batch_size 8 \
    --num_epochs 3 \
    --learning_rate 1e-4 \
    --cutoff_len 512 \
    --val_set_size 2000 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,v_proj]' \
    --train_on_inputs False \
    --group_by_length 