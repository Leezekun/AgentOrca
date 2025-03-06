"""
file used to task initialization, exchange internal state change, and evaluation
"""

import json
import copy
import random
import re
import inspect
from collections import Counter
from collections.abc import Iterable

from env.variables import domain_keys, domain_assistant_keys
from env.helpers import (
    get_action_parameters, 
    dict_to_tuple, 
    orig_dep,
    get_new_param_mapping, 
    dfsgather_constr_singles_dep, 
    dfsins_cl_cd_aid,
    get_ifcg_connections_invnodes, 
    dfsgather_ifcdg_func,
    gather_action_default_dependencies,
    InvalidConstraintOption
)

# retrieves the verbalization for a single constraint
def get_single_dep_verb(domain_str:dict, dep:tuple, dep_params:dict)->str:
    domain_assistant = domain_assistant_keys[domain_str]
    dep_not = re.sub("not ", "", dep[1])
    dep_str_params = dep[2] | dep_params if dep[2] else dep_params
    pos_dep_str = domain_assistant.positive_constraint_descriptions[dep_not].format(**dep_str_params)
    neg_dep_str = domain_assistant.negative_constraint_descriptions[dep_not].format(**dep_str_params)
    return pos_dep_str if "not " not in dep[1] else neg_dep_str

# dfs finds all constraints, then verbalizes them with format inputs
def dfsget_depverb_old(domain_str:str, dep:tuple, dep_params:dict)->set[str]:
    if not dep: return None
    elif dep[0] == "single": return {get_single_dep_verb(domain_str, dep, dep_params)}
    set_dep_str = set()
    for d in dep[1]:
        set_dep_str_temp = dfsget_depverb_old(domain_str, d, dep_params)
        if set_dep_str_temp: set_dep_str = set_dep_str | set_dep_str_temp
    return set_dep_str

# dfs finds all constraints, then verbalizes them with format inputs - structured format
def dfsget_depverb_structured(domain_str:str, dep:tuple, dep_params:dict, indent_level:int=0)->str:
    if not dep: return "None"
    elif dep[0] == "single": 
        return get_single_dep_verb(domain_str, dep, dep_params)
    
    parts = []
    
    # Add header based on type
    if dep[0] == "and":
        parts.append("ALL of these conditions must be met:")
    elif dep[0] == "or":
        parts.append("ANY ONE of these conditions must be met:")
    elif dep[0] == "chain":
        parts.append("These steps must be completed in order:")
    
    # Process child constraints with increased indentation
    for i, dep_part in enumerate(dep[1], 1):
        part_str = dfsget_depverb_structured(domain_str, dep_part, dep_params, indent_level + 1)
        part_lines = part_str.split('\n')
        
        # Calculate indentation for content
        indent = "  " * indent_level
        
        # Add bullet or number
        if dep[0] == "chain":
            first_line = f"{indent}{i}. {part_lines[0]}"
        else:
            first_line = f"{indent}• {part_lines[0]}"
        
        # For multiline content
        if len(part_lines) > 1:
            rest_lines = []
            for line in part_lines[1:]:
                rest_lines.append(f"{indent}  {line.strip()}")
            parts.append('\n'.join([first_line] + rest_lines))
        else:
            parts.append(first_line)
    
    return '\n'.join(parts)

# receives the dfs results, formats it into a string
def get_dep_verb(domain_str:str, dep:tuple, dep_params:dict, constraint_descr_format:str)->str:
    dep_verb = None
    match constraint_descr_format:
        case "old":
            set_dep_str = dfsget_depverb_old(domain_str, dep, dep_params)
            if not set_dep_str: return "None"
            dep_verb = list(set_dep_str)
            dep_verb = [f"{i+1}. {dep_verb[i]}" for i in range(len(dep_verb))]
            dep_verb = '\n'.join(dep_verb)
        case "structured": dep_verb = dfsget_depverb_structured(domain_str, dep, dep_params)
        case _ : dep_verb = "None"
    return dep_verb

# returns the default dependency based on the domain and the option enumerated
def get_default_dep_full(domain_str:str, default_constraint_option:str, add_constr_dep_bool:bool=True)->dict:
    ard = domain_assistant_keys[domain_str].action_required_dependencies
    acd = domain_assistant_keys[domain_str].action_customizable_dependencies
    cd = domain_assistant_keys[domain_str].constraint_dependencies
    return gather_action_default_dependencies(ard, acd, cd if add_constr_dep_bool else None, default_constraint_option)

# returns the default full dependency of the tasks and the default descriptions
def task_default_dep_full(domain_str:str, default_constraint_option:str, constraint_descr_format:str, dependency_verb_dep_orig:bool=False)->tuple[dict,dict,dict]:
    # collecting the default dependencies for non-tested actions
    dep_innate_full = domain_assistant_keys[domain_str].action_innate_dependencies
    default_dep_full = get_default_dep_full(domain_str, default_constraint_option)
    ddf_to_be_verbalized = default_dep_full if not dependency_verb_dep_orig else get_default_dep_full(domain_str, default_constraint_option, False)
    default_dep_full_descr = {}
    default_domain_system_strict = domain_keys[domain_str+"_strict"]()
    dep_params = default_domain_system_strict.evaluation_get_dependency_parameters()
    for action in ddf_to_be_verbalized:
        dep_verb = get_dep_verb(domain_str, ddf_to_be_verbalized[action], dep_params, constraint_descr_format)
        default_dep_full_descr[action] = dep_verb
    return dep_innate_full, default_dep_full, default_dep_full_descr

dep_descr_format_instr = {
    "old": "You must follow the routines and constraints in the order that they are listed."\
        + " Routines describe the chain and set of conditions that must be met in order to execute an action.",
    "structured": "The constraints are organized hierarchically:\n"\
        + "- 'ALL of these conditions must be met' indicates that every listed condition is required (AND logic)\n"\
        + "- 'ANY ONE of these conditions must be met' indicates that at least one condition is required (OR logic)\n"\
        + "- 'These steps must be completed in order' indicates a sequence that must be followed (CHAIN logic)\n"\
        + "Numbered items (1., 2., etc.) represent ordered steps, while bulleted items (•) represent unordered conditions.\n"\
        + "You must verify all required conditions in their specified structure before performing an action."
}

# gathering the dependency instructions, dependencies guaranteed to be with the assistant
def gather_dependency_instructions(domain_str:str, dep_full_descr:dict, user_goal:str, dep:dict,
    dep_params:dict, included_functions:list[str] | None, shuffle_func:bool, constraint_descr_format:str, provide_database_getter:bool=False)->str:
    # fill in the user goal dependency
    dep_verb = get_dep_verb(domain_str, dep, dep_params, constraint_descr_format)
    dep_full_descr[user_goal] = dep_verb
    # construct the full verbalization
    # service actions and internal functions
    service_funcs, internal_funcs = [], []
    for action in dep_full_descr:
        if dep_full_descr[action] == "None" and action.startswith("internal_"): internal_funcs.append(action)
        else: service_funcs.append(action)
    # shuffle the service actions
    if shuffle_func: 
        random.shuffle(service_funcs)
        random.shuffle(internal_funcs)
    list_dep_instr = []
    # add the service actions to the list
    list_dep_instr.append("### Actions with Constraints:")
    for service_func in service_funcs:
        if not included_functions or service_func in included_functions:
            list_dep_instr.append(f"* {service_func}:\n{dep_full_descr[service_func]}")
    # add the internal functions to the end of the list
    list_dep_instr.append("### Internal Verification Functions:")
    for internal_func in internal_funcs:
        if not included_functions or internal_func in included_functions:
            if not provide_database_getter and internal_func == "internal_get_database": continue
            list_dep_instr.append(f"* {internal_func}")
    # adding instructions on how to interpret the descriptions based on format type
    global dep_descr_format_instr
    ddfi = f"{dep_descr_format_instr[constraint_descr_format]}\n\n" if constraint_descr_format in dep_descr_format_instr else ""
    dependency_instructions = ddfi + '\n\n'.join(list_dep_instr)
    return dependency_instructions

# initializes the task environment, need to consider if there is no task
def task_initializer(domain_str:str, task:dict, dep_innate_full:dict, default_dep_full:dict, default_dep_full_descr:dict, 
                     included_functions:list[str] | None, mode:str, shuffle_func:bool, constraint_descr_format:str, dependency_verb_dep_orig:bool=True)->tuple:
    # initializing the domain system
    dep_full = copy.deepcopy(default_dep_full)
    dep_full_descr = copy.deepcopy(default_dep_full_descr)
    user_goal = task["user_goal"] if task else None
    dep = task["dependency"] if task else {}
    dep_orig = task["dependency_original"] if task else tuple()
    dep_params = None
    domain_system = None
    # if task is not specified, use defaults constraints
    if task:
        data = copy.deepcopy(task["initial_database"])
        dep_full[user_goal] = dep
        dep_params = task["dependency_parameters"]
        if mode != "program": domain_system = domain_keys[domain_str](data, dep_innate_full, dep_params)
        else: domain_system = domain_keys[domain_str+"_strict"](data, dep_innate_full, dep_full, dep_params)
    else:
        domain_system = domain_keys[domain_str+"_strict"](dep_innate_full=dep_innate_full, dep_full=dep_full)
        dep_params = domain_system.evaluation_get_dependency_parameters()
        if mode != "program": domain_system = domain_keys[domain_str](dep_innate_full=dep_innate_full, dep_params=dep_params)
    
    # compiling the user instructions
    user_instructions = f"You should roleplay as a user has requests within the {domain_str} domain. Your goal is: " + task["verb_user_goal"] if task else "None"
    user_known = task["user_known"] if task else {}
    for parameter in user_known: user_instructions += f" \"{parameter}\" is \"{user_known[parameter]}\"."
    user_info = {"instructions":user_instructions, "known":user_known}
 
    # compiling the assistant dependency instructions
    # assistant_dependency_instructions = gather_dependency_instructions(domain_str, dep_full_descr, user_goal, dep,
        # dep_params, included_functions, shuffle_func, constraint_descr_format) if mode != "program" else None
    dep_to_be_verbalized = dep if not dependency_verb_dep_orig else dep_orig
    assistant_dependency_instructions = gather_dependency_instructions(domain_str, dep_full_descr, user_goal, dep_to_be_verbalized,
        dep_params, included_functions, shuffle_func, constraint_descr_format) if mode != "program" else None
    assistant_info = create_assistant(domain_str, shuffle_func, mode, included_functions, assistant_dependency_instructions)

    # task_information for internal state during the interaction
    task_information = {"domain_str": domain_str, "initial_database": copy.deepcopy(domain_system.evaluation_get_database())}
    return domain_system, user_info, assistant_info, task_information

def dfsgather_constr_singles_dep(dep:tuple)->set:
    params_set = set()
    if not dep: return params_set
    match dep[0]:
        case "single":
            params_set = {(dep[0], re.sub("not ", "", dep[1]), dict_to_tuple(dep[2]))}
        case "and" | "or" | "chain" | "gate":
            for ele in dep[1]: params_set = params_set | dfsgather_constr_singles_dep(ele)
        case _: raise InvalidConstraintOption(f"invalid dependency option selected: {dep[0]}")
    return params_set

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
    connections_task, inv_nodes_task = get_ifcg_connections_invnodes(ifcg)
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
            nodes, connections, inv_nodes = dfsgather_ifcdg_func(domain_system, domain_assistant_keys[domain_str], func_name, default_constraint_option)
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


def create_assistant(domain_str:str, shuffle_func:bool, mode:str, included_functions:list[str] | None, assistant_dependency_instructions:str=None, provide_database_getter:bool=False):
    # assistant description for all assistants, length limit 512
    assistant_description = """Roleplay as an assistant that helps the user with his request.
        Access Control: You and your functions are the only way the user can receive services and assistance.
        There are no alternatives to accessing the database, system, or accounts."""
    # assistant_description = re.sub(r"\n+\t*\s\s+", " ", assistant_description)
    assistant_core_instructions = """
    1. Action Selection:
     - Choose the most appropriate, direct, and best-fit action for the user's task or checking constraints.
     - Avoid unnecessary function calls or actions that provide excessive information
    2. Action Validation:
     - Validate all required conditions in the specified order before proceeding with the target action.
     - Use the most relevant tools to verify each prerequisite condition.
     - Proceed with the target action only when all conditions are met.
     - If any condition fails, explain why and decline the action. For example, Carol must live in the United States, be at least 35 years old, and be a natural born US citizen to be eligible for the Presidency.
    3. Exit Conversation:
     - Exit the conversation if the request is completed or you cannot assist me with this request."""
    # assistant_core_instructions = re.sub(r"\n+\t*\s\s+", " ", assistant_core_instructions)
    
    # parse the data
    domain_assistant = domain_assistant_keys[domain_str]
    name = domain_assistant.name
    instructions = f"{assistant_description}\n\n\n### Role Description:\n{domain_assistant.instructions}"\
        + f"\n\n\n### Core Operating Principles:\n{assistant_core_instructions}"
    actions = copy.deepcopy(domain_assistant.actions)
    # remove the internal functions if in the strict mode (oracle mode)
    if mode == "program":
        # remove the internal_ function entries from actions
        i = 0
        while i < len(actions):
            if "internal_" not in actions[i]["name"]: i += 1
            else: actions.pop(i)
        # remove the internal_ function entries from descriptions
        action_complete_descriptions = [domain_assistant.action_descriptions, domain_assistant.action_returns]
        action_complete_description_keys_copy = list(domain_assistant.action_descriptions.keys())
        for action_complete_description_key in action_complete_description_keys_copy:
            if "internal_" not in action_complete_description_key: continue
            for action_description_part in action_complete_descriptions:
                del action_description_part[action_complete_description_key]
    
    # keep only the included functions
    if included_functions:
        actions = [action for action in actions if action["name"] in included_functions]
    if not provide_database_getter:
        actions = [action for action in actions if action["name"] != "internal_get_database"]
    
    # constructing the action descriptions
    for action in actions:
        # each action is guaranteed to have a description and return
        action["description"] = domain_assistant.action_descriptions[action["name"]]\
            + ' ' + domain_assistant.action_returns[action["name"]]
    if assistant_dependency_instructions: 
        # instructions += f"\n\n\n### Action Constraints:\n\n{assistant_dependency_instructions}"
        instructions += f"\n\n\n{assistant_dependency_instructions}"
    actions_shuffled = actions
    if shuffle_func: random.shuffle(actions_shuffled)
    tools = [{"function":action, "type":"function"} for action in actions_shuffled]
    assistant = {"name":name, "instructions":instructions, "tools":tools}
    return assistant

# calculates the statistics of the entire run
def task_domain_statistics(all_statistics_results:list[dict], ex_task_eval_res:dict, ex_task_stat_res:dict,
    allowed_statistic_types:list[str]=["total_", "distr_"])->dict:
    # determine the evaluation attributes that are booleans and the statistic attributes that are averaged
    boolean_attributes = set()
    for key in ex_task_eval_res:
        if isinstance(ex_task_eval_res[key], bool): boolean_attributes.add(key)
    averaged_attributes = set()
    for key in ex_task_stat_res:
        if key.find("avg_") == 0: averaged_attributes.add(key[len("avg_"):])
    # gather the attributes needed
    ds = domain_statistics = {key:0 if key.find("total_")==0 else {}
        for key in ex_task_stat_res
        if any(key.find(word)==0 for word in allowed_statistic_types)}
    for task_stat_res in all_statistics_results:
        for key in task_stat_res:
            if key.find("total_")==0: ds[key] += task_stat_res[key]
            elif key.find("distr_")==0: ds[key] = Counter(ds[key]) + Counter(task_stat_res[key])
    # calculating proportion and averages
    def get_underlying_attribute(attribute:str, statistic_types:list[str])->str:
        for stat_type in statistic_types:
            if stat_type in attribute: return attribute[len(stat_type):]
        return None
    ds_keys_copy = list(ds.keys())
    for key in ds_keys_copy:
        und_attr = get_underlying_attribute(key, allowed_statistic_types)
        if not und_attr: continue
        elif und_attr in boolean_attributes: ds[f"prop_{und_attr}"] = round(ds[f"total_{und_attr}"] / ds[f"total_interactions"], 5)
        elif und_attr in averaged_attributes: ds[f"avg_{und_attr}"] = round(ds[f"total_{und_attr}"] / ds[f"total_interactions"], 5)
    return domain_statistics