import os
import re
import copy
import argparse
import json
from tqdm import tqdm
from env.task import evaluator_function_directed_graph

def try_eval(x):
    try:
        return eval(x)
    except:
        return x
    
def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    # Model settings
    parser.add_argument("--user_model", type=str, default=None,
                       help="Model to use for the user agent")
    parser.add_argument("--assistant_model", type=str, default="gpt-4o-mini",
                       help="Model to use for the assistant agent")
    
    # Evaluation settings
    parser.add_argument("--tool_call_mode", type=str, default="fc",
                       help="Tool call mode for the assistant model", choices=["fc", "act-only", "react"])
    parser.add_argument("--tool_list", type=str, default="test",
                        choices=["full", "test"], help="Tool list to use for the simulation, only use the tools that have been evaluated or full tool list")
    parser.add_argument("--shuffle_func", action="store_true",
                       help="Whether to shuffle assistant functions")
    parser.add_argument("--default_constraint_option", type=str, default="full",
                        choices=["full", "required"], help="Default dependency to use for the other unevaluated actions")
    parser.add_argument("--constraint_descr_format", type=str, default="structured",
                        choices=["old", "structured"], help="Constraint dependency description format")
    parser.add_argument("--num_run_per_interaction", type=int, default=1,
                       help="Number of interactions per task")
    
    # Data settings
    parser.add_argument("--output_dir", type=str, default="./output",
                       help="Output directory")
    parser.add_argument("--domain", type=str, default="bank",
                       choices=["bank", "online_market", "dmv", "healthcare", "library", "all"], help="Domain name")
    
    args = parser.parse_args()
    
    return args

def save_results(output_file, results, verbose=False):
    """
    Save results to a JSON file.
    
    Args:
        output_file (str): Path to the output file
        results (list): List of task simulation results to save
        verbose (bool): Whether to print saving status
    """
    try:
        # Add debug print before saving
        print(f"Saving {len(results)} results. First result has evaluations: {'evaluations' in results[0] if results else False}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)

        if verbose:
            print(f"Results saved successfully to {output_file}")
    except Exception as e:
        print(f"Error saving results to {output_file}: {str(e)}")

def load_existing_results(output_file):
    """
    Load existing results from a JSON file or return empty list if file doesn't exist.
    
    Args:
        output_file (str): Path to the output file
        
    Returns:
        list: List of task simulation results
    """
    try:
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
                print(f"Loaded {len(results)} results from {output_file}")  # Add debug print
                return results
        else:
            print(f"File {output_file} does not exist!")
    except Exception as e:
        print(f"Error loading results from {output_file}: {str(e)}")
    return []

def main():
    """Main function to run the simulation."""
    args = parse_args()
    
    # Define sort key function for constraint groups
    def constraint_group_sort_key(item):
        key = item[0]
        if key == "6+":
            return 6  # Make "6+" sort after 5
        return int(key)  # Convert other keys to integers
    
    # Define domains to process
    domains_to_process = ["bank", "online_market", "dmv", "healthcare", "library"] if args.domain == "all" else [args.domain]
    
    # Initialize combined results for all domains
    combined_results = {}
    # Initialize global counters for call_database when domain is "all"
    total_call_database_count = 0
    total_cases_count = 0
    
    for current_domain in domains_to_process:
        # Setup output path
        output_dir = f"{args.output_dir}/{current_domain}"
        new_output_dir = f"{args.output_dir}/eval_results/{current_domain}"
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(new_output_dir, exist_ok=True)
        output_file = os.path.join(
            output_dir,
            (f"usr_{args.user_model.replace('/', '_')}-" if args.user_model else "") + \
            f"ast_{args.assistant_model.replace('/', '_')}-" + \
            f"mode_{args.tool_call_mode}-" + \
            f"dep_{args.default_constraint_option}-" + \
            f"fmt_{args.constraint_descr_format}-" + \
            f"tool_{args.tool_list}-" + \
            f"shuffle_{args.shuffle_func}.json"
        )
        new_output_file = os.path.join(
            new_output_dir,
            (f"usr_{args.user_model.replace('/', '_')}-" if args.user_model else "") + \
            f"ast_{args.assistant_model.replace('/', '_')}-" + \
            f"mode_{args.tool_call_mode}-" + \
            f"dep_{args.default_constraint_option}-" + \
            f"fmt_{args.constraint_descr_format}-" + \
            f"tool_{args.tool_list}-" + \
            f"shuffle_{args.shuffle_func}.json"
        )

        print_results = {
            "user_model": args.user_model,
            "assistant_model": args.assistant_model,
            "domain": current_domain,
            "total_tasks": 0,
            "total_interactions": 0,
            "total_success": 0,
            "avg_num_messages": 0,
            "avg_num_function_calls": 0,
            "avg_num_constraints": 0,
            "avg_num_constraints_expanded": 0,
            "call_database_stats": {          # Add this new field
                "total_call_database": 0,
                "total_cases": 0,
                "percentage": 0
            },
            "goal_statistics": {},
            "constraint_group_statistics": {},  # Group by constraint count
            "error_statistics": {              # Modified error tracking
                "total_evaluations": 0,
                "total_failures": 0,           # Add counter for failed cases
                "error_causes": {
                    "no_tool_call_error": 0,
                    "constraint_not_violated": 0,
                    "database_match": 0,
                    "dirgraph_satisfied": 0,
                    "action_called_correctly": 0
                }
            },
        }
        # Accuracy Metrics: Pass@N
        for i in range(args.num_run_per_interaction):
            print_results[f"pass@{i+1}"] = 0
            print_results[f"total_test_cases@{i+1}"] = 0
        
        # Load existing results
        task_simulations = load_existing_results(output_file)
        
        # Create a new list to store updated simulations
        updated_task_simulations = []
        
        # Add a counter for call_database in the main function where evaluations are processed
        call_database_count = 0
        total_cases = 0
        
        for idx, task_simulation in tqdm(enumerate(task_simulations), desc="Evaluating task simulations"):
            evaluations = []
            user_goal = task_simulation["task"]["user_goal"]
            dependency = task_simulation["task"]["dependency_original"]
            dependency_expanded = task_simulation["task"]["dependency"]
            
            # Count constraints for this task
            num_constraints = count_constraint_units(dependency)
            num_constraints_expanded = count_constraint_units(dependency_expanded)

            # Initialize goal statistics if not exists
            if user_goal not in print_results["goal_statistics"]:
                print_results["goal_statistics"][user_goal] = {
                    "total_interactions": 0,
                    "total_success": 0,
                    "avg_num_messages": 0,
                    "avg_num_function_calls": 0,
                    "total_tasks": 0,
                    "avg_num_constraints": 0,
                    "avg_num_constraints_expanded": 0,  # New field for expanded constraints per goal
                    "total_constraints": 0,
                    "total_constraints_expanded": 0     # New field to help calculate average
                }
                for i in range(args.num_run_per_interaction):
                    print_results["goal_statistics"][user_goal][f"pass@{i+1}"] = 0
                    print_results["goal_statistics"][user_goal][f"total_test_cases@{i+1}"] = 0
            
            call_database = False
            for interaction_log in task_simulation["interactions"]:
                
                results = {"final_database": interaction_log["database"]}
                interaction = interaction_log["interaction"]
                for message in interaction:
                    if "tool_calls" in message.keys():
                        tool_calls = message["tool_calls"]
                        if tool_calls:
                            for tool_call in tool_calls:
                                if tool_call["function"]["name"] == "internal_get_database":
                                    call_database = True
                                    break
                
                # collect the function call and response
                func_calls = []
                for i in range(len(interaction)-1):
                    if interaction[i].get("tool_calls", None):
                        func_calls.append({
                            "tool_name": interaction[i+1]["tool_name"],
                            "arguments": try_eval(interaction[i]["tool_calls"][0]["function"]["arguments"]),
                            "content": try_eval(interaction[i+1]["content"])
                        })

                # Use directed graph evaluator instead of gorilla
                evaluation_result = evaluator_function_directed_graph(
                    domain_str=task_simulation["domain"],
                    task=task_simulation["task"],
                    log_msg_fcall=interaction,
                    func_calls=func_calls,
                    results=results,
                    default_constraint_option=args.default_constraint_option)
                print(json.dumps(evaluation_result, indent=4))
                # _ = input("Press ENTER to continue...")
                # save the evaluation result for this interaction
                evaluations.append(evaluation_result)

                # Update error statistics
                print_results["error_statistics"]["total_evaluations"] += 1
                
                # Only track errors for failed cases
                if not evaluation_result.get("success", False):
                    print_results["error_statistics"]["total_failures"] += 1
                    # Check all status indicators for each failed evaluation
                    if not evaluation_result["no_tool_call_error"]:
                        print_results["error_statistics"]["error_causes"]["no_tool_call_error"] += 1

                    if not evaluation_result["constraint_not_violated"]:
                        print_results["error_statistics"]["error_causes"]["constraint_not_violated"] += 1

                    if not evaluation_result["database_match"]:
                        print_results["error_statistics"]["error_causes"]["database_match"] += 1

                    if not evaluation_result["dirgraph_satisfied"]:
                        print_results["error_statistics"]["error_causes"]["dirgraph_satisfied"] += 1

                    if not evaluation_result["action_called_correctly"]:
                        print_results["error_statistics"]["error_causes"]["action_called_correctly"] += 1

                if call_database:
                    break
            
            # Increment counters after checking all interactions for this task
            if call_database:
                call_database_count += 1
                if args.domain == "all":
                    total_call_database_count += 1
            total_cases += 1
            if args.domain == "all":
                total_cases_count += 1
            
            # collect the function call and response
            func_calls = []
            for i in range(len(evaluations)-1):
                func_calls.append({
                    "tool_name": evaluations[i+1]["tool_name"],
                    "arguments": evaluations[i]["arguments"],
                    "content": evaluations[i]["content"]
                })

            # Create a new task simulation dict with the evaluations
            updated_simulation = copy.deepcopy(task_simulation)
            updated_simulation["evaluations"] = evaluations
            updated_simulation["statistics"] = task_statistics(evaluations)
            
            # Add to our new list
            updated_task_simulations.append(updated_simulation)
            
            # update the overall statistics and goal-specific statistics
            if updated_simulation["statistics"]["total_interactions"] > 0:
                # record the print results
                print_results["total_tasks"] += 1
                print_results["total_interactions"] += updated_simulation["statistics"]["total_interactions"]
                print_results["total_success"] += updated_simulation["statistics"]["total_success"]
                print_results["avg_num_messages"] += updated_simulation["statistics"]["avg_num_messages"]
                print_results["avg_num_function_calls"] += updated_simulation["statistics"]["avg_num_function_calls"]
                print_results["avg_num_constraints"] += num_constraints
                print_results["avg_num_constraints_expanded"] += num_constraints_expanded  # Add expanded constraints

                # Update goal-specific statistics
                goal_stats = print_results["goal_statistics"][user_goal]
                goal_stats["total_tasks"] += 1
                goal_stats["total_interactions"] += updated_simulation["statistics"]["total_interactions"]
                goal_stats["total_success"] += updated_simulation["statistics"]["total_success"]
                goal_stats["avg_num_messages"] += updated_simulation["statistics"]["avg_num_messages"]
                goal_stats["avg_num_function_calls"] += updated_simulation["statistics"]["avg_num_function_calls"]
                goal_stats["total_constraints"] += num_constraints
                goal_stats["total_constraints_expanded"] += num_constraints_expanded  # Add expanded constraints

                # Pass@N for overall and goal-specific
                for i in range(min(args.num_run_per_interaction, updated_simulation["statistics"]["total_interactions"])):
                    print_results[f"pass@{i+1}"] += int(updated_simulation["statistics"][f"pass@{i+1}"])
                    print_results[f"total_test_cases@{i+1}"] += 1
                    goal_stats[f"pass@{i+1}"] += int(updated_simulation["statistics"][f"pass@{i+1}"])
                    goal_stats[f"total_test_cases@{i+1}"] += 1

            # Group by number of constraints
            constraint_count = num_constraints
            if constraint_count >= 6:  # Group all counts >= 6 into "6+"
                constraint_count = "6+"
            elif constraint_count <= 1:  # Group 0 and 1 together as "1"
                constraint_count = 1
                
            if constraint_count not in print_results["constraint_group_statistics"]:
                print_results["constraint_group_statistics"][constraint_count] = {
                    "total_tasks": 0,
                    "total_interactions": 0,
                    "total_success": 0,
                    "avg_num_messages": 0,
                    "avg_num_function_calls": 0,
                    "pass_rate": 0
                }
            
            group_stats = print_results["constraint_group_statistics"][constraint_count]
            if updated_simulation["statistics"]["total_interactions"] > 0:
                group_stats["total_tasks"] += 1
                group_stats["total_interactions"] += updated_simulation["statistics"]["total_interactions"]
                group_stats["total_success"] += updated_simulation["statistics"]["total_success"]
                group_stats["avg_num_messages"] += updated_simulation["statistics"]["avg_num_messages"]
                group_stats["avg_num_function_calls"] += updated_simulation["statistics"]["avg_num_function_calls"]

        # After processing all tasks, update call_database statistics
        print_results["call_database_stats"]["total_call_database"] = call_database_count
        print_results["call_database_stats"]["total_cases"] = total_cases
        print_results["call_database_stats"]["percentage"] = (call_database_count / total_cases) * 100 if total_cases > 0 else 0

        # Save the updated task simulations
        save_results(new_output_file, updated_task_simulations, verbose=True)
        
        if args.domain == "all":
            combined_results[current_domain] = print_results
        
        # Calculate percentage for individual domain
        if args.domain != "all":
            call_database_percentage = (call_database_count / total_cases) * 100 if total_cases > 0 else 0
            print(f"Call database count for {current_domain}: {call_database_count}/{total_cases} ({call_database_percentage:.2f}%)")
    
    # Calculate and print overall percentage for "all" domains
    if args.domain == "all":
        overall_percentage = (total_call_database_count / total_cases_count) * 100 if total_cases_count > 0 else 0
        print(f"Total call database count across all domains: {total_call_database_count}/{total_cases_count} ({overall_percentage:.2f}%)")

    if args.domain == "all":
        # Create "all" directory
        all_output_dir = f"{args.output_dir}/all"
        os.makedirs(all_output_dir, exist_ok=True)
        
        # Calculate and save aggregate statistics across all domains
        aggregate_results = {
            "user_model": args.user_model,
            "assistant_model": args.assistant_model,
            "domain": "all",
            "domain_specific_results": combined_results,
            "aggregate_statistics": {
                "total_tasks": sum(r["total_tasks"] for r in combined_results.values()),
                "total_interactions": sum(r["total_interactions"] for r in combined_results.values()),
                "total_success": sum(r["total_success"] for r in combined_results.values()),
                "avg_num_messages": sum(r["avg_num_messages"] * r["total_tasks"] for r in combined_results.values()) / 
                                  sum(r["total_tasks"] for r in combined_results.values()),
                "avg_num_function_calls": sum(r["avg_num_function_calls"] * r["total_tasks"] for r in combined_results.values()) / 
                                        sum(r["total_tasks"] for r in combined_results.values()),
                "avg_num_constraints": sum(r["avg_num_constraints"] * r["total_tasks"] for r in combined_results.values()) / 
                                     sum(r["total_tasks"] for r in combined_results.values()),
                "avg_num_constraints_expanded": sum(r["avg_num_constraints_expanded"] * r["total_tasks"] for r in combined_results.values()) / 
                                              sum(r["total_tasks"] for r in combined_results.values()),
                "call_database_stats": {          # Add this new field
                    "total_call_database": total_call_database_count,
                    "total_cases": total_cases_count,
                    "percentage": (total_call_database_count / total_cases_count) * 100 if total_cases_count > 0 else 0
                }
            },
            "error_statistics": {              # Add error statistics for all domains
                "total_evaluations": sum(r["error_statistics"]["total_evaluations"] for r in combined_results.values()),
                "total_failures": sum(r["error_statistics"]["total_failures"] for r in combined_results.values()),
                "error_causes": {
                    "no_tool_call_error": sum(r["error_statistics"]["error_causes"]["no_tool_call_error"] for r in combined_results.values()),
                    "constraint_not_violated": sum(r["error_statistics"]["error_causes"]["constraint_not_violated"] for r in combined_results.values()),
                    "database_match": sum(r["error_statistics"]["error_causes"]["database_match"] for r in combined_results.values()),
                    "dirgraph_satisfied": sum(r["error_statistics"]["error_causes"]["dirgraph_satisfied"] for r in combined_results.values()),
                    "action_called_correctly": sum(r["error_statistics"]["error_causes"]["action_called_correctly"] for r in combined_results.values())
                }
            }
        }

        # Aggregate constraint group statistics across all domains
        all_constraint_groups = {}
        
        for domain_results in combined_results.values():
            # Existing constraint count aggregation
            for constraint_count, stats in domain_results["constraint_group_statistics"].items():
                if constraint_count not in all_constraint_groups:
                    all_constraint_groups[constraint_count] = {
                        "total_tasks": 0,
                        "total_interactions": 0,
                        "total_success": 0,
                        "avg_num_messages": 0,
                        "avg_num_function_calls": 0,
                    }
                group = all_constraint_groups[constraint_count]
                group["total_tasks"] += stats["total_tasks"]
                group["total_interactions"] += stats["total_interactions"]
                group["total_success"] += stats["total_success"]
                group["avg_num_messages"] += stats["avg_num_messages"] * stats["total_tasks"]
                group["avg_num_function_calls"] += stats["avg_num_function_calls"] * stats["total_tasks"]
            
            # Pass@N for overall and goal-specific
            for i in range(args.num_run_per_interaction):
                if f"pass@{i+1}" in domain_results:  # Check if the pass@N metric exists
                    for goal, goal_stats in domain_results["goal_statistics"].items():
                        if f"pass@{i+1}" in goal_stats:  # Check if the goal has this metric
                            goal_stats[f"pass@{i+1}"] += domain_results[f"pass@{i+1}"]
                            goal_stats[f"total_test_cases@{i+1}"] += 1

        # Calculate averages for aggregated constraint groups
        for constraint_count, stats in all_constraint_groups.items():
            if stats["total_tasks"] > 0:
                stats["avg_num_messages"] /= stats["total_tasks"]
                stats["avg_num_function_calls"] /= stats["total_tasks"]
                stats["pass_rate"] = stats["total_success"] / stats["total_interactions"] if stats["total_interactions"] > 0 else 0

        # Sort constraint groups using the already defined sort key function
        aggregate_results["aggregate_constraint_groups"] = dict(sorted(
            all_constraint_groups.items(),
            key=constraint_group_sort_key
        ))

        # Calculate aggregate pass rates
        total_interactions = sum(r["total_interactions"] for r in combined_results.values())
        aggregate_results["aggregate_statistics"]["pass_rate"] = (
            sum(r["total_success"] for r in combined_results.values()) / total_interactions if total_interactions > 0 else 0
        )
        
        # Calculate error percentages based on total evaluations for all domains
        total_evaluations = aggregate_results["error_statistics"]["total_evaluations"]
        if total_evaluations > 0:
            error_percentages = {}
            for error, count in aggregate_results["error_statistics"]["error_causes"].items():
                error_percentages[f"{error}_percentage"] = (count / total_evaluations) * 100
            aggregate_results["error_statistics"]["percentages"] = error_percentages
        
        # Save aggregate results in the "all" directory
        aggregate_output_file = os.path.join(
            all_output_dir,
            (f"usr_{args.user_model.replace('/', '_')}-" if args.user_model else "") + \
            f"ast_{args.assistant_model.replace('/', '_')}-" + \
            f"mode_{args.tool_call_mode}-" + \
            f"dep_{args.default_constraint_option}-" + \
            f"fmt_{args.constraint_descr_format}-" + \
            f"tool_{args.tool_list}-" + \
            f"shuffle_{args.shuffle_func}.json"
        )
        
        with open(aggregate_output_file, 'w') as f:
            json.dump(aggregate_results, f, indent=4)
        
        # Print the aggregate results
        print(json.dumps(aggregate_results, indent=4))
    else:
        # Calculate overall statistics         
        print_results["pass_rate"] = print_results["total_success"] / print_results["total_interactions"]
        print_results["avg_num_messages"] /= print_results["total_tasks"]
        print_results["avg_num_function_calls"] /= print_results["total_tasks"]
        print_results["avg_num_constraints"] /= print_results["total_tasks"]
        print_results["avg_num_constraints_expanded"] /= print_results["total_tasks"]
        
        # Update call_database statistics in the final output
        print_results["call_database_stats"]["total_call_database"] = call_database_count
        print_results["call_database_stats"]["total_cases"] = total_cases
        print_results["call_database_stats"]["percentage"] = (call_database_count / total_cases) * 100 if total_cases > 0 else 0
        
        # calculate pass rate
        for i in range(args.num_run_per_interaction):
            if print_results[f"total_test_cases@{i+1}"] > 0:
                print_results[f"pass@{i+1}"] /= print_results[f"total_test_cases@{i+1}"]
            else:
                del print_results[f"pass@{i+1}"]
            del print_results[f"total_test_cases@{i+1}"]

        # Calculate goal-specific statistics
        for goal, stats in print_results["goal_statistics"].items():
            if stats["total_interactions"] > 0:
                stats["pass_rate"] = stats["total_success"] / stats["total_interactions"]
                stats["avg_num_messages"] /= stats["total_tasks"]
                stats["avg_num_function_calls"] /= stats["total_tasks"]
                stats["avg_num_constraints"] = stats["total_constraints"] / stats["total_tasks"]
                stats["avg_num_constraints_expanded"] = stats["total_constraints_expanded"] / stats["total_tasks"]
                del stats["total_constraints"]
                del stats["total_constraints_expanded"]
                
                # Calculate pass@N for each goal
                for i in range(args.num_run_per_interaction):
                    if stats[f"total_test_cases@{i+1}"] > 0:
                        stats[f"pass@{i+1}"] /= stats[f"total_test_cases@{i+1}"]
                    else:
                        del stats[f"pass@{i+1}"]
                    del stats[f"total_test_cases@{i+1}"]

        # Sort goal_statistics by avg_num_constraints
        sorted_goals = dict(sorted(
            print_results["goal_statistics"].items(),
            key=lambda x: x[1]["avg_num_constraints"]
        ))
        print_results["goal_statistics"] = sorted_goals

        # Calculate averages for constraint groups
        for constraint_count, stats in print_results["constraint_group_statistics"].items():
            if stats["total_tasks"] > 0:
                stats["avg_num_messages"] /= stats["total_tasks"]
                stats["avg_num_function_calls"] /= stats["total_tasks"]
                stats["pass_rate"] = stats["total_success"] / stats["total_interactions"]

        # Sort constraint_group_statistics using the already defined sort key function
        sorted_constraint_groups = dict(sorted(
            print_results["constraint_group_statistics"].items(),
            key=constraint_group_sort_key
        ))
        print_results["constraint_group_statistics"] = sorted_constraint_groups

        # Calculate error percentages based on total evaluations
        total_evaluations = print_results["error_statistics"]["total_evaluations"]
        if total_evaluations > 0:
            error_percentages = {}
            for error, count in print_results["error_statistics"]["error_causes"].items():
                error_percentages[f"{error}_percentage"] = (count / total_evaluations) * 100
            print_results["error_statistics"]["percentages"] = error_percentages

        # Print the results
        print(json.dumps(print_results, indent=4))

def count_constraint_units(dependency):
    """
    Count the number of constraint units in a dependency structure.
    A constraint unit is a 'single' condition that represents a basic constraint.
    
    Args:
        dependency: A nested list representing the dependency structure
        
    Returns:
        int: Number of constraint units found
    """
    if not dependency:
        return 0
    
    # If it's a single constraint
    if isinstance(dependency, list) and len(dependency) >= 1 and dependency[0] == "single":
        return 1
    
    # If it's a logical operator (and/or)
    if isinstance(dependency, list) and len(dependency) >= 2 and dependency[0] in ["and", "or"]:
        count = 0
        # Recursively count constraints in each branch
        for branch in dependency[1]:
            count += count_constraint_units(branch)
        return count
    return 0

def task_statistics(evaluations):
    """Calculate statistics for a task's evaluations."""
    if not evaluations:
        return {
            "total_interactions": 0,
            "total_success": 0,
            "avg_num_messages": 0,
            "avg_num_function_calls": 0
        }
    
    # Count constraint units for the first evaluation
    # (assuming all evaluations in the same task have the same dependency structure)
    num_constraints = 0
    if evaluations and "dependency" in evaluations[0]:
        num_constraints = count_constraint_units(evaluations[0]["dependency"])
        
    stats = {
        "total_interactions": len(evaluations),
        "total_success": sum(1 for e in evaluations if e["success"]),
        "avg_num_messages": sum(e["num_messages"] for e in evaluations) / len(evaluations),
        "avg_num_function_calls": sum(e["num_function_calls"] for e in evaluations) / len(evaluations),
        "num_constraints": num_constraints  # Add the constraint count to statistics
    }
    
    # Calculate Pass@N statistics
    for i in range(len(evaluations)):
        # Check if any evaluation up to index i was successful
        stats[f"pass@{i+1}"] = any(e["success"] for e in evaluations[:i+1])
    
    return stats

if __name__ == "__main__":
    main()