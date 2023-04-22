CUDA_VISIBLE_DEVICES=0
cd "$(dirname $0)"
export PATH="/usr/local/conda/envs/lora/bin:$PATH"

python ../generate.py \
    --gpu_id 1 \
    --load_8bit \
    --base_model '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/chihuixuan/llm/data/chinese-alpaca-lora-13b' \
    --lora_weights '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/chihuixuan/llm/output/rerun-2-alpaca-13b-2-1' 