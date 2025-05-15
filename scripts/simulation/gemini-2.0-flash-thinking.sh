cd ../..

model="gemini-2.0-flash-thinking-exp"
domains=("healthcare" "university")

# Experiment 1: Full and only Test Env Tool List on Five Domains with FC
method="react"
for domain in "${domains[@]}"; do
    CUDA_VISIBLE_DEVICES=$devices python run_simulation.py \
            --domain $domain \
            --user_model adv \
            --assistant_model $model \
            --env_mode prompt \
            --tool_list full \
            --tool_call_mode $method \
            --assistant_max_tokens 4096
done
