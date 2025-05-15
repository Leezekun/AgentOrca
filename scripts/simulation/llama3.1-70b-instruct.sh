cd ../..

devices="0,1,2,3,4,5,6,7"

model="llama3.1-70b-instruct"
domains=("university" "healthcare" "dmv" "online_market" "bank" "hotel" "library")
tool_lists=("full")
method="react"

for tool_list in "${tool_lists[@]}"; do
    for domain in "${domains[@]}"; do
        CUDA_VISIBLE_DEVICES=$devices python run_simulation.py \
                --domain $domain \
                --user_model adv \
                --assistant_model $model \
                --env_mode prompt \
                --tool_list $tool_list \
                --tool_call_mode $method \
                --num_gpus 8 \
                --gpu_memory_utilization 0.9
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
#                 --num_gpus 8 \
#                 --gpu_memory_utilization 0.9
#     done
# done