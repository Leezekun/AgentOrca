cd ../..

devices="4,5,6,7"

model="qwen2.5-3b-instruct"
domains=("dmv" "healthcare" "library" "online_market" "bank")
tool_lists=("full" "oracle")
method="react"

for domain in "${domains[@]}"; do
    for tool_list in "${tool_lists[@]}"; do
        CUDA_VISIBLE_DEVICES=$devices python run_simulation.py \
                --domain $domain \
                --assistant_model $model \
                --env_mode prompt \
                --tool_list $tool_list \
                --tool_call_mode $method \
                --num_gpus 4 \
                --gpu_memory_utilization 0.9
    done
done

# Experiment 2: Full and Env Tool List on Five Domains with React
methods=("react" "act-only")
tool_list="full"
for domain in "${domains[@]}"; do
    for method in "${methods[@]}"; do
        CUDA_VISIBLE_DEVICES=$devices python run_simulation.py \
                --domain $domain \
                --assistant_model $model \
                --env_mode prompt \
                --tool_list $tool_list \
                --tool_call_mode $method \
                --num_gpus 4 \
                --gpu_memory_utilization 0.9
    done
done