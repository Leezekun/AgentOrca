cd ..

models=("gpt-4o" "claude-3-5-sonnet-20241022" "gemini-2.0-flash-001" "gpt-4o-mini" 
        "qwen2.5-72b-instruct" "gemini-2.0-flash-thinking-exp" "llama3.1-70b-instruct" 
        "deepseek-r1" "o1")
domains=("healthcare" "dmv" "library" "online_market" "bank")
tool_lists=("full" "test")
tool_call_modes=("fc" "react" "act-only")

# Default settings
for domain in "${domains[@]}"; do
    for tool_list in "${tool_lists[@]}"; do
        for tool_call_mode in "${tool_call_modes[@]}"; do
            for model in "${models[@]}"; do
            CUDA_VISIBLE_DEVICES=$devices python run_evaluation.py \
                        --domain $domain \
                        --assistant_model $model \
                        --tool_list $tool_list \
                        --tool_call_mode $tool_call_mode
            done
        done
    done
done