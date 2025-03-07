
import json
import copy
import random
import re
from collections import Counter
from collections.abc import Iterable

from env.variables import domain_keys, domain_assistant_keys
from env.task import get_default_dep_full
from env.helpers import (
    dict_to_tuple,
    get_action_parameters, 
    orig_dep,
    get_new_param_mapping, 
    dfsgather_constr_singles_dep, 
    dfsins_cl_cd_aid,
    get_dag_connections_invnodes, 
    dfsgather_dag_func,
)

# account for the json tuple to list aspect by dfs converting every function response to a tuple
def dfsconvert_tuple_to_list(fr):
    fr = copy.deepcopy(fr)
    if not isinstance(fr, Iterable) or isinstance(fr, str) or isinstance(fr, set): return fr
    elif isinstance(fr, dict):
        for key in fr: fr[key] = dfsconvert_tuple_to_list(fr[key])
        return fr
    elif isinstance(fr, tuple): fr = list(fr)
    for i in range(len(fr)): fr[i] = dfsconvert_tuple_to_list(fr[i])
    return fr

# dfs convert to hash with a tuple
def dfsconvert_list_to_tuple(fr):
    if not isinstance(fr, Iterable) or isinstance(fr, str): return fr
    elif isinstance(fr, dict): fr = dict_to_tuple(fr)
    # allows for list and set
    fr_copy = []
    for ele_fr in fr: fr_copy.append(dfsconvert_list_to_tuple(ele_fr))
    return tuple(fr_copy)

# evaluates the interaction with function call tree search, detects if the action should be called or not, matches the database
def evaluator_function_directed_graph(domain_str:str, task:dict, log_msg_fcall:list[tuple], func_calls:list[tuple], results:dict, default_constraint_option:str)->dict:
    evaluation_result = {}
    dep_innate_full = domain_assistant_keys[domain_str].action_innate_dependencies
    default_dep_full = get_default_dep_full(domain_str, default_constraint_option)
    default_dep_full[task["user_goal"]] = task["dependency"]
    # gathering statistics
    evaluation_result["user_goal"] = task["user_goal"]
    evaluation_result["action_should_succeed"] = task["action_should_succeed"]
    evaluation_result["num_messages"] = len([entry_content for entry_content in log_msg_fcall if "sender" in entry_content])-1 # -1 due to end conversation message
    evaluation_result["num_function_calls"] = len(func_calls)
    # gathering ground truth function responses
    domain_system_strict = domain_keys[domain_str+"_strict"](copy.deepcopy(task["initial_database"]), dep_innate_full, default_dep_full, task["dependency_parameters"])
    domain_system = domain_system_strict.evaluation_get_domain_system()
    gt_responses = []
    evaluation_result["no_tool_call_error"] = True
    for func_call in func_calls:
        func_name, arguments = func_call["tool_name"], func_call["arguments"]
        # status_id 0: function exists and is called, 1: function exists in DS but not DS_strict, 2: arguments invalid
        try:
            if hasattr(domain_system_strict, func_name) and hasattr(domain_system, func_name):
                gt_response = (0, copy.deepcopy(getattr(domain_system_strict, func_name)(**arguments)))
            elif not hasattr(domain_system_strict, func_name) and hasattr(domain_system, func_name):
                gt_response = (1, None)
            else:
                evaluation_result["no_tool_call_error"] = False
                gt_response = (2, None)
        except Exception as _:
            evaluation_result["no_tool_call_error"] = False
            gt_response = (2, None)
        gt_responses.append(gt_response)
    gt_final_database = domain_system_strict.evaluation_get_database()
    # comparing against final database
    evaluation_result["constraint_not_violated"] = True
    for i in range(len(func_calls)):
        func_response = func_calls[i]["content"]
        status_id, gt_response = gt_responses[i]
        func_resp_equal = dfsconvert_tuple_to_list(func_response) == dfsconvert_tuple_to_list(gt_response) if status_id == 0 else True
        if evaluation_result["constraint_not_violated"] and not func_resp_equal:
            evaluation_result["constraint_not_violated"] = False
    evaluation_result["database_match"] = results["final_database"] == gt_final_database
    # dfs checks if the listed functions were called
    def dfscheck_called_functions(node_ind:int, func_param_mapping:dict, nodes:list, connections:list, successful_funccalls:dict)->bool:
        # base case single function
        if not isinstance(nodes[node_ind], str):
            func_name, func_params = nodes[node_ind]
            # function never called before
            if func_name not in successful_funccalls: return False
            # function called before, check the parameters of those previous calls
            func_param_keys_sorted = successful_funccalls[func_name][0]
            exp_func_param_values = tuple(dfsconvert_list_to_tuple(func_param_mapping[func_params[key]]) if key in func_param_mapping else None
                for key in func_param_keys_sorted)
            act_prev_func_param_values = successful_funccalls[func_name][1]
            if exp_func_param_values in act_prev_func_param_values: return True
            # previous action may have additional, unseen parameters
            for pfpv in act_prev_func_param_values:
                mismatch_found_bool = False
                for i in range(len(pfpv)):
                    if (not mismatch_found_bool
                        and exp_func_param_values[i]
                        and exp_func_param_values[i] != pfpv[i]):
                        mismatch_found_bool = True
                if not mismatch_found_bool: return True
            # no match found
            return False
        # recursive case, "and" or "or"
        and_node_bool = nodes[node_ind] == "and"
        all_prev_func_called = and_node_bool
        for node_ind_part in connections[node_ind]:
            apfc_part = dfscheck_called_functions(node_ind_part, func_param_mapping, nodes, connections, successful_funccalls)
            if apfc_part != and_node_bool: return apfc_part
        return all_prev_func_called
    # returns a dictionary of functions needed based on a dependency, {"action1": [("param1", "param2"), {('a', 1), ('b', 2)}]}
    # assumes all "and" for the processes of constraints in aid, cannot handle "or" as we don't know which action was taken in the "or"
    def dfsgather_allfunccalled_indepperm(dep_perm:tuple, constr_pros:dict, constr_action_set:set)->dict:
        all_func_called = {}
        if not dep_perm: return all_func_called
        elif dep_perm[0] == "single":
            constr_str = re.sub("not ", "", dep_perm[1])
            # returns a new structure to record multiple calls of the same function with different parameters
            def get_new_func_called_set(param_mapping:dict):
                param_keys_sorted = tuple(sorted(list(param_mapping.keys())))
                return [param_keys_sorted, {tuple(param_mapping[key] for key in param_keys_sorted)}]
            # check the constraint is an action, or the constraint process for actions required (assuming all actions need to be taken)
            if constr_str in constr_action_set:
                all_func_called[constr_str] = get_new_func_called_set(dep_perm[2])
            elif constr_str in constr_pros:
                constraint_process_action_set = dfsgather_constr_singles_dep(constr_pros[constr_str])
                for hashed_action in constraint_process_action_set:
                    _, act_req, act_req_params = orig_dep(hashed_action)
                    act_req_params = get_new_param_mapping(dep_perm[2], act_req_params)
                    if act_req not in all_func_called: all_func_called[act_req] = get_new_func_called_set(act_req_params)
                    else: all_func_called[act_req][1].add(tuple(act_req_params[key] for key in all_func_called[act_req][0]))
            return all_func_called
        for dep_perm_part in dep_perm[1]:
            all_func_called_part = dfsgather_allfunccalled_indepperm(dep_perm_part, constr_pros, constr_action_set)
            for func_name in all_func_called_part:
                if func_name not in all_func_called: 
                    all_func_called[func_name] = all_func_called_part[func_name]
                else: 
                    # Convert the values to tuples before updating
                    all_func_called[func_name][1].update(all_func_called_part[func_name][1])
            return all_func_called
    # inserts parameter values into the function call recorder structure
    def ipfc_insert_param_values(implied_prev_func_called:dict, parameter_values:dict)->dict:
        ipfc_with_values = {}
        parameter_names = set(parameter_values.keys())
        for ipfc_func_name in implied_prev_func_called:
            ipfc_keys = implied_prev_func_called[ipfc_func_name][0]
            ipfc_values = implied_prev_func_called[ipfc_func_name][1]
            if not set(ipfc_keys) <= parameter_names: continue
            ipfc_with_values[ipfc_func_name] = [ipfc_keys, set()]
            for ipfc_value in ipfc_values:
                ipfc_with_values[ipfc_func_name][1].add(tuple(parameter_values[ele] for ele in ipfc_value))
        return ipfc_with_values
    # detecting if action is successfully called and if the assistant called the necessary functions, tool calls are valid
    nodes, connections, inv_nodes = None, None, None
    ifcg = copy.deepcopy(task["directed_action_graph"]) # directed_action_graph with user_known values plugged in
    nodes_task = ifcg["nodes"]
    connections_task, inv_nodes_task = get_dag_connections_invnodes(ifcg)
    constr_links = domain_assistant_keys[domain_str].constraint_links
    constr_deps = domain_assistant_keys[domain_str].constraint_dependencies
    constr_pros = domain_assistant_keys[domain_str].constraint_processes
    action_parameters = get_action_parameters(domain_system, domain_assistant_keys[domain_str])
    constr_act_set = set(action_parameters.keys())
    successful_funccalls = {} # {"action1": [("param1", "param2"), {('a', 1), ('b', 2)}]}
    evaluation_result["action_successfully_called"] = False
    evaluation_result["dirgraph_satisfied"] = True
    for i in range(len(func_calls)):
        # filter out error function calls and parse the function call
        if gt_responses[i][0] == 2: continue
        func_call = func_calls[i]
        func_name, func_args, func_resp = func_call["tool_name"], func_call["arguments"], func_calls[i]["content"]
        # make a new connection graph if the function is not in the current graph
        nodes, connections, inv_nodes = nodes_task, connections_task, inv_nodes_task
        if func_name not in inv_nodes and func_name in action_parameters:
            nodes, connections, inv_nodes = dfsgather_dag_func(domain_system, domain_assistant_keys[domain_str], func_name, default_constraint_option)
        elif func_name not in action_parameters: continue
        # detecting when the target action has been successfully called
        if (not evaluation_result["action_successfully_called"]
            and func_name == task["user_goal"]
            and (func_resp if isinstance(func_resp, bool) else func_resp[0] if isinstance(func_resp, tuple) or isinstance(func_resp, list) else False)):
            evaluation_result["action_successfully_called"] = True
        # traversing the graph to see if the assistant has called the necessary functions before this function call
        node_ind = inv_nodes[func_name]
        node_inds_to_check = connections[node_ind] # function call nodes should only have no neighbors or one neigbhbor
        func_param_mapping = {nodes[node_ind][1][key]: func_args[key] if key in func_args else None for key in nodes[node_ind][1]} # maps the dep func param values to the act func param values
        all_prev_func_called = dfscheck_called_functions(list(node_inds_to_check)[0], func_param_mapping, nodes, connections, successful_funccalls)\
            if node_inds_to_check else True
        # record the result, update the functions successfully called based on the innate dependencies
        if all_prev_func_called:
            if func_name not in successful_funccalls: successful_funccalls[func_name] = [tuple(sorted(list(nodes[node_ind][1].keys()))), set()]
            successful_funccalls[func_name][1].add(tuple(dfsconvert_list_to_tuple(func_args[key]) if key in func_args else None for key in successful_funccalls[func_name][0]))
            # successfully calling some action implies innate dependencies
            if func_name in dep_innate_full and dep_innate_full[func_name]:
                dep_innate_perm = dfsins_cl_cd_aid(dep_innate_full[func_name], constr_links, dep_innate_full, default_dep_full, constr_deps, action_parameters)
                implied_prev_func_called = dfsgather_allfunccalled_indepperm(dep_innate_perm, constr_pros, constr_act_set)
                ipfc_with_values = ipfc_insert_param_values(implied_prev_func_called, func_args)
                for ipfc_func_name in ipfc_with_values:
                    if ipfc_func_name not in successful_funccalls: successful_funccalls[ipfc_func_name] = ipfc_with_values[ipfc_func_name]
                    else: successful_funccalls[ipfc_func_name][1].update(ipfc_with_values[ipfc_func_name][1])
        else: evaluation_result["dirgraph_satisfied"] = False
    # final evaluation of assistant success
    evaluation_result["action_called_correctly"] =\
        evaluation_result["action_should_succeed"] == evaluation_result["action_successfully_called"]
    evaluation_result["success"] = (
        evaluation_result["no_tool_call_error"]
        and evaluation_result["constraint_not_violated"]
        and evaluation_result["database_match"]
        and evaluation_result["action_called_correctly"]
        and evaluation_result["dirgraph_satisfied"]
    )
    return evaluation_result