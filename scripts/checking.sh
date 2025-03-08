#!/bin/bash
cd ..

# Default values
output_dir="./output"
domain="online_market"
assistant_model="gpt-4o"
tool_call_mode="fc"
default_constraint_option="full"
constraint_descr_format="structured"
tool_list="oracle"

python run_checking.py \
  --output_dir $output_dir \
  --domain $domain \
  --assistant_model $assistant_model \
  --tool_call_mode $tool_call_mode \
  --default_constraint_option $default_constraint_option \
  --constraint_descr_format $constraint_descr_format \
  --tool_list $tool_list