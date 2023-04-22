# CUDA_VISIBLE_DEVICES=0,1,2,3
cd "$(dirname $0)"
export PATH="/usr/local/conda/envs/lora/bin:$PATH"
python ../finetune.py \
    --base_model '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/chihuixuan/llm/data/chinese-alpaca-lora-13b' \
    --data_path '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/chihuixuan/DATA4HOPE/alpaca-lora/firefly/firefly-train-0.json' \
    --output_dir '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/chihuixuan/llm/output/run-2-alpaca-13b-1' \
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