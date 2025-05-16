cd ..

# models=(
#         "gpt-4o" 
#         "gpt-4o-mini" 
#         "claude-3-5-sonnet-20241022" 
#         "gemini-2.0-flash-001" 
#         "gemini-1.5-pro"
#         "gemini-2.0-flash-thinking-exp" 
#         "o1"
#         "deepseek-r1"
#         "llama3.1-70b-instruct" 
#         "qwen2.5-72b-instruct" 
#         "llama3.1-8b-instruct"
#         "llama3.2-3b-instruct"
#         "qwen2.5-32b-instruct"
#         "qwen2.5-14b-instruct"
#         "qwen2.5-7b-instruct"
#         "qwen2.5-3b-instruct"
#         )
# domains=(
#     "healthcare" 
#     "dmv" 
#     "library" 
#     "online_market" 
#     "bank"
#     )
# tool_lists=(
#     "full" 
#     "oracle"
#     )
# tool_call_modes=(
#     "fc" 
#     "react" 
#     "act-only"
#     )

# Default settings
# for model in "${models[@]}"; do
#     for domain in "${domains[@]}"; do
#         for tool_list in "${tool_lists[@]}"; do
#             for tool_call_mode in "${tool_call_modes[@]}"; do
#                 CUDA_VISIBLE_DEVICES=$devices python run_evaluation.py \
#                     --domain $domain \
#                     --assistant_model $model \
#                     --tool_list $tool_list \
#                     --tool_call_mode $tool_call_mode
#             done
#         done
#     done
# done

# model="gemini-1.5-pro"
# model="llama3.1-70b-instruct"
# model="o4-mini-high"
# model="qwen2.5-72b-instruct"
# model="gemini-2.0-flash-thinking-exp"
# model="claude-3-7-sonnet-20250219"
# model="claude-3-5-sonnet-20241022"
# model="deepseek-r1"
# model="gemini-2.0-flash-thinking-exp"
# model="gemini-1.5-pro"
model="o4-mini-high"
# model="gpt-4o-mini"
# model="gemini-2.0-flash-001"
# model="gemini-2.0-flash-thinking-exp"
# model="gemini-1.5-pro"
# model="gpt-4o"
# model="deepseek-r1"
# model="llama3.1-70b-instruct"
# model="qwen2.5-32b-instruct"
# model="claude-3-7-sonnet-20250219"
# model="gpt-4.1"
domain="all"
tool_list="full"
tool_call_mode="fc"

python run_evaluation1.py \
    --domain $domain \
    --assistant_model $model \
    --tool_list $tool_list \
    --tool_call_mode $tool_call_mode