"""
running the task generation
"""

from env.generation import task_generation

import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run task generation")
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=2000)
    parser.add_argument("--gpt_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--domain_str", type=str, default="bank")
    parser.add_argument("--default_constraint_option", type=str, default="full", choices=["required", "full"])
    parser.add_argument("--generation_limit", type=int, default=4)
    parser.add_argument("--autogen_manfix", action="store_true")
    parser.add_argument("--debug_mode", action="store_true")
    parser.add_argument("--testing_mode", action="store_true")
    parser.add_argument("--testing_mode_last_task", action="store_true")
    parser.add_argument("--testing_mode_user_goal", type=str, default="transfer_funds")
    parser.add_argument("--print_pipeline_disable", action="store_false")
    parser.add_argument("--domains_dir", type=str, default="env\\domains")
    parser.add_argument("--write_output_disable", action="store_false")
    parser.add_argument("--indent_amount", type=int, default=2)
    args = parser.parse_args()
    all_run_usage = task_generation(args)