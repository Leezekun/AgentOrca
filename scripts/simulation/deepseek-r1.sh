cd ../..

model="deepseek-r1"
domains=("dmv" "healthcare" "library" "online_market" "bank")
# Experiment 1: Full and only Test Env Tool List on Five Domains with FC
method="react"
for domain in "${domains[@]}"; do
    CUDA_VISIBLE_DEVICES=$devices python run_simulation.py \
            --domain $domain \
            --assistant_model $model \
            --env_mode prompt \
            --tool_list full \
            --assistant_max_tokens 2048 \
            --tool_call_mode $method
done
