cd ../..

model="deepseek-r1"
domains=("hotel")
# Experiment 1: Full and only Test Env Tool List on Five Domains with FC
method="react"
for domain in "${domains[@]}"; do
    CUDA_VISIBLE_DEVICES=$devices python run_simulation.py \
            --domain $domain \
            --assistant_model $model \
            --env_mode prompt \
            --tool_list full \
            --assistant_max_tokens 4096 \
            --tool_call_mode $method \
            --print_conv
done
