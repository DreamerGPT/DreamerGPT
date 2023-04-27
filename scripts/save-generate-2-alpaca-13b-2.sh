cd "$(dirname $0)"

python ../generate.py \
    --gpu_id 2 \
    --load_8bit \
    --base_model '/path/to/model/chinese-alpaca-lora-13b' \
    --lora_weights '/path/to/output/rerun-2-alpaca-13b-2' \
    --is_ui False \
    --test_input_path '/path/to/test_input/1.json'