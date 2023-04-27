CUDA_VISIBLE_DEVICES=0
cd "$(dirname $0)"
export PATH="/usr/local/conda/envs/lora/bin:$PATH"

python ../generate.py \
    --gpu_id 2 \
    --load_8bit \
    --base_model '/path/to/model/chinese-alpaca-lora-13b' \
    --lora_weights '/path/to/output/run-2-alpaca-13b-1' 