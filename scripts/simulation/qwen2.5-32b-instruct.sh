cd ../..

devices="4,5,6,7"

model="qwen2.5-32b-instruct"
domains=("healthcare" "university" "dmv" "online_market" "bank" "hotel" "library")
tool_lists=("full")
method="react"

for i in {1..5}; do
    for domain in "${domains[@]}"; do
        for tool_list in "${tool_lists[@]}"; do
            CUDA_VISIBLE_DEVICES=$devices python run_simulation.py \
                --domain $domain \
                --user_model adv \
                --assistant_model $model \
                --env_mode prompt \
                --tool_list $tool_list \
                --tool_call_mode $method \
                --num_gpus 4 \
                --gpu_memory_utilization 0.9
        done
    done
done

# Experiment 2: Full and Env Tool List on Five Domains with React
# methods=("react" "act-only")
# tool_list="full"
# for domain in "${domains[@]}"; do
#     for method in "${methods[@]}"; do
#         CUDA_VISIBLE_DEVICES=$devices python run_simulation.py \
#                 --domain $domain \
#                 --assistant_model $model \
#                 --env_mode prompt \
#                 --tool_list $tool_list \
#                 --tool_call_mode $method \
#                 --num_gpus 4 \
#                 --gpu_memory_utilization 0.9
#     done
# done