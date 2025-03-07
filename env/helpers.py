"""
File to hold helper functions for task_generation that could be used elsewhere
"""

import re
import copy
import itertools
import inspect
from collections import deque
from difflib import SequenceMatcher
from textwrap import dedent


"""basic helper functions"""

class InvalidConstraintOption(Exception): pass

# inverses the string of the single constraint
def inv_constr(constr_str:str)->str:
    return f"not {constr_str}" if "not " not in constr_str else constr_str[constr_str.find("not ")+len("not "):]

# recursively inverses the dependency, basically DeMorgan's Law; special case "chain", where we only flip the last element
def inv_dep(dep:tuple, cl_handle:bool=False)->tuple:
    if not dep: return None
    dep_new = None
    match dep[0]:
        case "single": dep_new = ("single", inv_constr(dep[1]), dep[2])
        case "and" | "or": dep_new = ("or" if dep[0] == "and" else "and", [inv_dep(ele, cl_handle) for ele in dep[1]])
        case "chain" | "gate":
            if not cl_handle: dep_new = ("gate" if dep[0] == "chain" else "chain", [inv_dep(dep[1][i], cl_handle) for i in range(len(dep[1]))])
            else: dep_new = (dep[0], [dep[1][i] if i < len(dep[1])-1 else inv_dep(dep[1][i], cl_handle) for i in range(len(dep[1]))])
        case _: raise InvalidConstraintOption(f"invalid dependency option selected: {dep[0]}")
    return dep_new

# modifies a formatted prompt to strip and remove the new lines in between each line
def modify_prompt(prompt:str)->str:
    prompt = re.sub(r"\n\s\s\s\s", ' ', dedent(prompt.strip()))
    remove_nl_pos = [i for i in range(1, len(prompt)) if prompt[i] == '\n' and prompt[i-1] != '\n'] # new line positions we want to remove'
    prompt_modified = prompt[:remove_nl_pos[0]]
    for i in range(1, len(remove_nl_pos)): prompt_modified += prompt[remove_nl_pos[i-1]+1:remove_nl_pos[i]]
    if remove_nl_pos[-1] < len(prompt)-1: prompt_modified += prompt[remove_nl_pos[-1]+1:]
    return prompt_modified

# combines the descriptions for the action and the return
def get_action_full_description(action_descriptions:dict, action_returns:dict, action_str:str)->str:
    return f"{action_descriptions[action_str]} {action_returns[action_str]}"

# puts none for all dependencies if no dependencies are specified
def get_domain_dependency_none(class_name:str)->dict:
    return {func:None for func in dir(class_name) if callable(getattr(class_name, func))}

# gets the action parameters for the domain
def get_action_parameters(domain_system, domain_assistant)->dict:
    list_action_name = [func for func in dir(domain_system)
        if callable(getattr(domain_system, func)) and not func.startswith("_") and not func.startswith("evaluation_")]
    action_parameters = {}
    for action_name in list_action_name:
        signature = inspect.signature(getattr(domain_system, action_name))
        action_params = {name for name, param in signature.parameters.items()
            if param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)}
        if hasattr(domain_assistant, "action_params_user_not_needed")\
            and action_name in domain_assistant.action_params_user_not_needed:
            action_params -= set(domain_assistant.action_params_user_not_needed[action_name])
        action_parameters[action_name] = action_params
    return action_parameters


"""converts a constraint to and from a hashable tuple"""

# turns a dict into a hashable tuple
def dict_to_tuple(dic:dict)->tuple:
    if not dic: return ()
    sorted_items = sorted(dic.items(), key=lambda item: item[0])
    dep_params_list = []
    for dict_item in sorted_items: dep_params_list.extend(dict_item)
    return tuple(dep_params_list)

# turns a tuple of values into a dictionary
def tuple_to_dict(tup:tuple)->dict:
    if not tup: return {}
    return {tup[i]: tup[i+1] for i in range(0, len(tup)-1, 2)}

# turns a constraint into a hashable tuple
def hashable_dep(dep:tuple)->tuple:
    if not dep: return None
    if dep[0] == "single": return ("single", dep[1], dict_to_tuple(dep[2]))
    if dep[0] not in ["and", "or", "chain", "gate"]: raise InvalidConstraintOption(f"invalid dependency option selected: {dep[0]}")
    dep_list = [hashable_dep(ele_dep) for ele_dep in dep[1]]
    if dep[0] in ["and", "or"]: dep_list = sorted(dep_list)
    return (dep[0], tuple(dep_list))

# turns a hashable constraint into its original state
def orig_dep(dep:tuple)->tuple:
    if not dep: return None
    if dep[0] == "single": return ("single", dep[1], tuple_to_dict(dep[2]))
    if dep[0] not in ["and", "or", "chain", "gate"]: raise InvalidConstraintOption(f"invalid dependency option selected: {dep[0]}")
    return (dep[0], tuple([orig_dep(ele_dep) for ele_dep in dep[1]]))


"""pruning the dependency or process tree"""

# used for pruning the dependency
asc_order_restr = ["chain", "and", "gate", "or"]

# check if the dep is directly seen or is encapsulated by another dependency, no exact matched found at this point, orig_dep_rel is not "single"
# "and" seen dep is True, "or" seen dep is False
def check_dep_seen_or_encapsulated(dep:tuple, seen_hashed_dep_set:set, orig_dep_rel:str)->bool:
    # dep directly seen before
    hashed_dep = hashable_dep(dep)
    if hashed_dep in seen_hashed_dep_set: return True
    # if not, then encapsulation detection method is needed
    global asc_order_restr
    pos_orig_dep_rel:int = asc_order_restr.index(orig_dep_rel)
    pos_dep_rel:int = asc_order_restr.index(dep[0]) if dep[0] != "single" else pos_orig_dep_rel # chain single == chain chain single
    dep_set:set = set(hashed_dep[1]) if dep[0] != "single" else {hashed_dep} # set of deps that dep represents
    encap_dep_found:bool = False
    for seen_hashed_dep in seen_hashed_dep_set:
        pos_seen_dep_rel:int = asc_order_restr.index(seen_hashed_dep[0]) if seen_hashed_dep[0] != "single" else pos_orig_dep_rel
        # check if the relations are equivalent in type and if the seen dep encapsulates the current dep
        rel_a, rel_b = (pos_seen_dep_rel, pos_dep_rel) if pos_orig_dep_rel < 1.5 else (pos_dep_rel, pos_seen_dep_rel)
        rel_equiv:bool = ((rel_a <= 1.5) == (rel_b <= 1.5)) # same type of relation
        rel_encap:bool = rel_a <= rel_b # one relation encapsulates the other
        if not rel_encap: continue
        # check if seen dep has a subset of current dep if relations are the same type, otherwise check or intersection
        seen_dep_set:set = set(seen_hashed_dep[1]) if seen_hashed_dep[0] != "single" else {seen_hashed_dep}
        depset_subset_inter:bool = not bool(dep_set - seen_dep_set) if rel_equiv else bool(seen_dep_set & dep_set)
        if not depset_subset_inter: continue
        # encapsulating dependency found
        encap_dep_found = True
        break
    return encap_dep_found

# removes dependencies if certain dependencies from the higher tiers have been seen
def dfsremove_if_unnecessary(dep:tuple, seen_hashed_dep_set:set, orig_dep_rel:str, inco_dep_rel:str,
    force_remove:bool=False)->tuple|bool:
    # None case
    if not dep: return dep
    global asc_order_restr
    # check if dependency is seen at the highest level
    pos_orig_dep_rel:int = asc_order_restr.index(orig_dep_rel)
    pos_inco_dep_rel:int = asc_order_restr.index(inco_dep_rel)
    if check_dep_seen_or_encapsulated(dep, seen_hashed_dep_set, orig_dep_rel)\
        and not (pos_orig_dep_rel // 2 == pos_inco_dep_rel // 2 and pos_inco_dep_rel < pos_orig_dep_rel):
        return pos_orig_dep_rel // 2 == 0
    # if not seen before
    if dep[0] == "single": return dep
    pos_dep_rel:int = asc_order_restr.index(dep[0])
    and_rel:bool = pos_dep_rel // 2 == 0 # True for "and", False for "or"
    dep_list_new = []
    for dep_part in dep[1]:
        dep_part = dfsremove_if_unnecessary(dep_part, seen_hashed_dep_set, orig_dep_rel, dep[0])
        # if dep_part is: bool (return a bool or remove), tuple (add into original list)
        if isinstance(dep_part, bool) and not force_remove and and_rel != dep_part: return dep_part
        elif isinstance(dep_part, tuple): dep_list_new.append(dep_part)
    return (dep[0], dep_list_new)

# collapses dependencies if there is only one dependency left in the "and" or "or"
def dfscollapse_dep(dep:tuple)->tuple:
    if dep[0] == "single": return dep
    dep_list_new = []
    for dep_part in dep[1]:
        dep_part = dfscollapse_dep(dep_part)
        if not dep_part: continue
        if dep[0] != dep_part[0]: dep_list_new.append(dep_part)
        else: dep_list_new.extend(dep_part[1])
    return (dep[0], dep_list_new) if len(dep_list_new) > 1 else\
        dep_list_new[0] if len(dep_list_new) == 1 else None

# prunes the dependency, nested "and" and "or", consider both True and False cases for simple constraints
def dfsprune_dep_pros(dep:tuple)->tuple:
    # base cases
    if not dep: return None
    elif dep[0] == "single": return dep
    # returns a set of dependencies that may be removed from elsewhere in the wider dependency, mainly handling "chain" and "gate"
    def dep_set_add_chaingate_ele(dep_part:tuple, dep_rel:str)->set:
        # if the dependency is "single" no additional elements added
        if dep_part[0] == "single": return set()
        global asc_order_restr
        # if the relation is not "chain" or "gate", no further actions required, else, add the "and" and "or" relation, respetively
        pos_dep_part_rel = asc_order_restr.index(dep_part[0])
        if not (pos_dep_part_rel % 2 == 0): return set()
        dep_set = {hashable_dep((asc_order_restr[pos_dep_part_rel+1], dep_part[1]))}
        # wider relation "chain" or "and" and "gate" or "or" matches "chain" and "gate" respectively
        pos_dep_rel = asc_order_restr.index(dep_rel)
        if pos_dep_part_rel // 2 == pos_dep_rel // 2:
            for dep_part_part in dep_part[1]: dep_set.add(hashable_dep(dep_part_part))
        return dep_set
    # prune the parts of the dependency, remove redundant constraints, once for each direction
    dep_list = []
    seen_dep_set = set()
    global asc_order_restr
    for dep_part in dep[1]:
        # prune and hash to check for identical
        dep_part = dfsprune_dep_pros(dep_part)
        hashed_dep_part = hashable_dep(dep_part)
        if not dep_part or hashed_dep_part in seen_dep_set: continue
        dep_list.append(dep_part)
        # update the set
        dep_part_set = {hashed_dep_part} | dep_set_add_chaingate_ele(dep_part, dep[0])
        seen_dep_set |= dep_part_set
    # remove identical dependencies in reverse
    rev_dep_list = dep_list[::-1]
    dep_list = []
    seen_dep_set = set()
    list_seen_dep_part_set = []
    for dep_part in rev_dep_list:
        # no None and already pruned
        hashed_dep_part = hashable_dep(dep_part)
        if hashed_dep_part in seen_dep_set: continue
        dep_list.append(dep_part)
        # update the set
        dep_part_set = {hashed_dep_part} | dep_set_add_chaingate_ele(dep_part, dep[0])
        seen_dep_set |= dep_part_set
        list_seen_dep_part_set.append(dep_part_set)
    dep_list = dep_list[::-1]
    list_seen_dep_part_set = list_seen_dep_part_set[::-1]
    # remove each part from the other parts depending on restrictiveness, 0 for forward and 1 for backward
    for ind_trav_type in range(2):
        ind_trav = range(len(dep_list)) if ind_trav_type == 0 else range(len(dep_list)-1, -1, -1)
        for i in ind_trav:
            # check for redundencies
            seen_dep_set_oneless = seen_dep_set.copy()
            seen_dep_set_oneless -= list_seen_dep_part_set[i]
            dep_part_old = dep_list[i]
            dep_list[i] = dfsremove_if_unnecessary(dep_list[i], seen_dep_set_oneless, dep[0], dep[0])
            # if the dependency changed, update the overarching set of seen dependencies
            if dep_part_old != dep_list[i]:
                if isinstance(dep_list[i], bool): list_seen_dep_part_set[i] = dep_list[i]
                else: list_seen_dep_part_set[i] = {hashable_dep(dep_list[i])} | dep_set_add_chaingate_ele(dep_list[i], dep[0])
                seen_dep_set = set.union(*[list_seen_dep_part_set[i] for i in range(len(dep_list)) if not isinstance(dep_list[i], bool)])
        dep_list = [ele for ele in dep_list if not isinstance(ele, bool)]
        list_seen_dep_part_set = [ele for ele in list_seen_dep_part_set if not isinstance(ele, bool)]
    # collapse if there is only one dependency part left
    dep_new = dfscollapse_dep((dep[0], dep_list))
    # return the new dependency, prune again for unseen corner cases
    if hashable_dep(dep) != hashable_dep(dep_new): dep_new = dfsprune_dep_pros(dep_new)
    return dep_new


"""basic dependency actions"""

# replace the keys of d1 in the values of d2 with the values of d1: d2[key] = d1[d2[key]]
def get_new_param_mapping(d1:dict, d2:dict)->dict:
    d = copy.deepcopy(d2)
    for key in d:
        if d[key] in d1: d[key] = d1[d[key]]
    return d

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

# dfs gather the parameter values in a dependency
def dfsgather_param_names_dep(dep:tuple)->set:
    params_set = set()
    if not dep: return params_set
    match dep[0]:
        case "single":
            if dep[2]: params_set = set(ele for ele in dep[2].values() if "value " not in ele)
        case "and" | "or" | "chain" | "gate":
            for ele in dep[1]: params_set = params_set | dfsgather_param_names_dep(ele)
        case _: raise InvalidConstraintOption(f"invalid dependency option selected: {dep[0]}")
    return params_set


"""retrieves the default dependencies, inserts the constraint dependencies"""

# dfs replaces param keys found in dep parameter values with the param values
def dfsplace_param_names(dep:tuple, params:dict)->tuple:
    if not dep: return None
    elif dep[0] == "single":
        params_new = copy.deepcopy(dep[2]) if dep[2] else {}
        for k in params_new:
            if params_new[k] in params: params_new[k] = params[params_new[k]]
        return dep[0], dep[1], params_new if params_new else None
    list_dep = []
    for dep_ele in dep[1]: list_dep.append(dfsplace_param_names(dep_ele, params))
    return dep[0], list_dep

# incorporates the constraint dependencies into the action dependency
def dfsins_constr_deps(dep:tuple, act_deps:dict, constr_deps:dict)->tuple:
    if not dep: return None
    dep_new = None
    match dep[0]:
        case "single":
            constr_str = re.sub("not ", "", dep[1])
            if not (constr_str in constr_deps and constr_deps[constr_str]): return dep
            constr_dep = dfsins_constr_deps(constr_deps[constr_str], act_deps, constr_deps)
            constr_dep = dfsplace_param_names(constr_dep, dep[2])
            dep_new = ("chain", [constr_dep, dep]) if "not " not in dep[1] else ("gate", [inv_dep(constr_dep), dep])
        case "and" | "or" | "chain" | "gate" :
            dep_new = (dep[0], [dfsins_constr_deps(dep_part, act_deps, constr_deps) for dep_part in dep[1]])
        case _: raise InvalidConstraintOption(f"invalid dependency option selected: {dep[0]}")
    return dfsprune_dep_pros(dep_new)

# gathers the default dependencies for each action
def gather_action_default_dependencies(action_required_dependencies:dict, action_customizable_dependencies:dict,
    constraint_dependencies:dict=None, default_dependency_option:str="required")->dict[str:tuple]:
    default_dep_full = copy.deepcopy(action_required_dependencies)
    if default_dependency_option == "full":
        for action in default_dep_full:
            action_cust_dep = copy.deepcopy(action_customizable_dependencies[action])
            if action_cust_dep and isinstance(action_cust_dep, list): action_cust_dep = ("and", action_cust_dep)
            if default_dep_full[action] and action_cust_dep:
                default_dep_full[action] = ("and", [default_dep_full[action], action_cust_dep])
            elif action_cust_dep: default_dep_full[action] = action_cust_dep
    if constraint_dependencies:
        default_dep_full = {action: dfsins_constr_deps(default_dep_full[action], default_dep_full, constraint_dependencies) for action in default_dep_full}
    return {action: dfsprune_dep_pros(default_dep_full[action]) for action in default_dep_full}

# dfs insert the innate dependencies
def dfsins_innate_deps(dep:tuple, aid:dict)->tuple:
    if not dep: return None
    dep_new = None
    match dep[0]:
        case "single":
            constr_str = re.sub("not ", "", dep[1])
            if not (constr_str in aid and aid[constr_str]): return dep
            dep_part = dfsins_innate_deps(aid[constr_str], aid)
            dep_part = dfsplace_param_names(dep_part, dep[2])
            dep_new = ("chain", [dep_part, dep]) if "not " not in dep[1] else ("gate", [inv_dep(dep_part), dep])
        case "and" | "or" | "chain" | "gate":
            dep_new = (dep[0], [dfsins_innate_deps(dep_part, aid) for dep_part in dep[1]])
        case _: raise InvalidConstraintOption(f"invalid dependency option selected: {dep[0]}")
    return dfsprune_dep_pros(dep_new)


"""gathers the action dependency with information needed (constraints) in mind"""

# given the constraint and the constr_str_seen, the function gives the proper function call (parameters previously found or brand new)
def get_cl_param_mapping(constr:tuple, constr_links:dict, action_parameters:dict[str:set], constr_str_seen:dict[str:dict[tuple:dict]]={}):
    constr_str, constr_param_mapping = constr
    # parse the contraint link action and parameter mapping
    cl_action, cl_param_mapping = copy.deepcopy(constr_links[constr_str]) if isinstance(constr_links[constr_str], tuple) else\
        (copy.deepcopy(constr_links[constr_str]), {})
    cl_param_mapping = cl_param_mapping if cl_param_mapping else {}
    for key in cl_param_mapping:
        if constr_param_mapping and cl_param_mapping[key] in constr_param_mapping: cl_param_mapping[key] = constr_param_mapping[cl_param_mapping[key]]
    # find the parameters of the current dependency and the linked action depedency
    constr_param_values = set(constr_param_mapping.values())
    depnew_param_values_new = copy.deepcopy(action_parameters[cl_action])
    for key in cl_param_mapping: depnew_param_values_new.remove(key)
    # load and/or record the correct parameter mapping
    key_param = tuple(sorted(list(constr_param_values)))
    if constr_str not in constr_str_seen:
        value_param_mapping = {key:key for key in depnew_param_values_new}
        constr_str_seen[constr_str] = {key_param: value_param_mapping}
    elif key_param not in constr_str_seen[constr_str]:
        # new mapping determined by count, could also do it by the constr params
        dpvn_param_mapping = {}
        for param_value in depnew_param_values_new:
            param_value_variations = [constr_str_seen[constr_str][kp_other][param_value] for kp_other in constr_str_seen[constr_str]]
            pvv_counts = [int(re.sub(param_value, "", pvv)) for pvv in param_value_variations if pvv != param_value]
            new_count = (max(pvv_counts) + 1) if pvv_counts else 0
            dpvn_param_mapping[param_value] = param_value + str(new_count)
        constr_str_seen[constr_str][key_param] = dpvn_param_mapping
    cl_param_mapping |= constr_str_seen[constr_str][key_param]
    # return the action
    return cl_action, cl_param_mapping

# dfs insert the constraint links: replaces the original constraint with the constraints of the linked action
# constr_str_seen key constr_str and value dict, dict key params tuple and value dict of param mappings to new params from the dep
def dfsins_constr_links(dep:tuple, constraint_links:dict, default_deps:dict, action_parameters:dict,
    constr_str_seen:dict[str:dict[tuple:dict]]={})->tuple:
    if not dep: return None
    dep_new = dep
    match dep[0]:
        case "single":
            constr_str = re.sub("not ", "", dep[1])
            cl = constraint_links
            if not (constr_str in cl and cl[constr_str]): return dep
            # parse the contraint link action and parameter mapping
            cl_action, cl_param_mapping = get_cl_param_mapping((constr_str, dep[2]), cl, action_parameters, constr_str_seen)
            # process the dependency
            cl_action_dep = copy.deepcopy(default_deps[cl_action])
            dep_part = dfsplace_param_names(cl_action_dep, cl_param_mapping)
            dep_part = dfsins_constr_links(dep_part, constraint_links, default_deps, action_parameters, constr_str_seen) # only recurse on the part that was inserted
            action_constr = ("single", cl_action, cl_param_mapping)
            dep_new = ("chain", [dep_part, action_constr]) if dep_part else action_constr
            if "not " in dep[1]: dep_new = inv_dep(dep_new) 
        case "and" | "or" | "chain" | "gate":
            dep_new = (dep[0], [dfsins_constr_links(dep_part, constraint_links, default_deps, action_parameters, constr_str_seen) for dep_part in dep[1]])
        case _: raise InvalidConstraintOption(f"invalid dependency option selected: {dep[0]}")
    return dfsprune_dep_pros(dep_new)

# recursively inserts constraint links, constraint dependencies, and action innate dependencies
# default deps already has constraint dependencies, inversing a chain is only inverting the last element
def dfsins_cl_cd_aid(dep:tuple, constr_links:dict, act_innate_deps:dict, act_def_deps:dict, constr_deps:dict, action_parameters:dict,
    constr_str_seen:dict[str:dict[tuple:dict]]={})->tuple:
    if not dep: return None
    dep_new = None
    cl, aid, ad, cd = constr_links, act_innate_deps, act_def_deps, constr_deps
    match dep[0]:
        case "single":
            constr_str = re.sub("not ", "", dep[1])
            # constraint is in constraint links
            if constr_str in cl and cl[constr_str]:
                cl_action, cl_action_params = get_cl_param_mapping((constr_str, dep[2]), constr_links, action_parameters, constr_str_seen)
                constr_single = ("single", cl_action if "not " not in dep[1] else ("not " + cl_action), cl_action_params)
                dep_new = dfsins_cl_cd_aid(constr_single, cl, aid, ad, cd, action_parameters, constr_str_seen)
            # constraint (is an action) seen in action innate dependencies or seen in constraint processes
            else:
                constr_locs = [aid, ad, cd]
                dep_new_chain = []
                for constr_loc in constr_locs:
                    if not(constr_str in constr_loc and constr_loc[constr_str]): continue
                    dep_new_prev = dfsplace_param_names(constr_loc[constr_str], dep[2])
                    dep_new_prev = dfsins_cl_cd_aid(dep_new_prev, cl, aid, ad, cd, action_parameters, constr_str_seen)
                    if dep_new_prev: dep_new_chain.append(dep_new_prev if "not " not in dep[1] else inv_dep(dep_new_prev))
                dep_new_chain.append(dep)
                dep_new = ("chain" if "not " not in dep[1] else "gate", dep_new_chain) if len(dep_new_chain) > 1 else dep_new_chain[0]
        case "and" | "or" | "chain" | "gate":
            dep_new = (dep[0], [dfsins_cl_cd_aid(dep_part, cl, aid, ad, cd, action_parameters, constr_str_seen) for dep_part in dep[1]])
        case _: raise InvalidConstraintOption(f"invalid dependency option selected: {dep[0]}")
    return dfsprune_dep_pros(dep_new)


"""retrieves the only the actions required to fullfill a dependency"""

# merges two lists, keeping the relative order
def merge_sequences(seq1, seq2)->list:
    sm = SequenceMatcher(a=seq1, b=seq2)
    res = []
    for (op, start1, end1, start2, end2) in sm.get_opcodes():
        # This range appears in both sequences, or only in the first one.
        if op == 'equal' or op == 'delete': res += seq1[start1:end1]
        # This range appears in only the second sequence.
        elif op == 'insert': res += seq2[start2:end2]
        # There are different ranges in each sequence - add both.
        elif op == 'replace':
            res += seq1[start1:end1]
            res += seq2[start2:end2]
    return res

# dfs gathers actions required to be called
def dfsgather_actions_required(dep_perm:tuple, hashed_cl_funcs:set)->list:
    if not dep_perm: return []
    deps_to_loop = []
    actions_required = []
    if dep_perm[0] == "single":
        func_str = re.sub("not ", "", dep_perm[1])
        func_params_to_find = {key: key for key in dep_perm[2]} if dep_perm[2] else None
        if (func_str, dict_to_tuple(func_params_to_find)) in hashed_cl_funcs:
            actions_required = [(func_str, dict_to_tuple(dep_perm[2]))]
    else: deps_to_loop = dep_perm[1]
    for dep_perm_part in deps_to_loop:
        actions_required = merge_sequences(actions_required, dfsgather_actions_required(dep_perm_part, hashed_cl_funcs))
    return actions_required


"""gather the functional call graph"""

# gathers all functions called later down the graph
def dfsgather_setfunccall_dag(dir_act_graph:dict, ind:int, set_func_call:set=set())->set:
    if not isinstance(dir_act_graph["nodes"][ind], str): return {dir_act_graph["nodes"][ind][0]}
    dag_conns = set((ind1, ind2) for ind1 in range(len(dir_act_graph["connections"])) for ind2 in dir_act_graph["connections"][ind1])
    for ind_from, ind_to in dag_conns:
        if ind_from != ind or ind_to in set_func_call: continue
        set_func_call |= dfsgather_setfunccall_dag(dir_act_graph, ind_to, set_func_call)
    return set_func_call

# updates the graph for a singular function call
def update_dir_act_graph_single(dir_act_graph:dict, func_str:str, func_params:dict, link_to_prev_root:bool)->dict:
    hashable_func = (func_str, dict_to_tuple(func_params))
    if hashable_func in dir_act_graph["inv_nodes"]: return dir_act_graph
    dir_act_graph["nodes"].append((func_str, func_params if func_params else {}))
    dir_act_graph["connections"].append(set())
    node_index = len(dir_act_graph["nodes"]) - 1
    dir_act_graph["inv_nodes"][hashable_func] = node_index
    if link_to_prev_root: dir_act_graph["connections"][node_index].add(dir_act_graph["root_ind"])
    dir_act_graph["root_ind"] = node_index
    return dir_act_graph

# updating the dir_act_graph with a part, connecting A nodes to B nodes
def update_dir_act_graph(dir_act_graph:dict, dir_act_graph_part:dict)->dict:
    # hashes the node
    def hash_node(node:tuple|str): return (node[0], dict_to_tuple(node[1])) if not isinstance(node, str) else node
    # checks if two nodes are identical
    def dfscheck_same_andornode(dir_act_graph:dict, dir_act_graph_part:dict, dag_node_ind:int, dagp_node_ind:int, seen_inds:list=[set(), set()])->bool:
        # updating seen indicies to not loop
        seen_inds = copy.deepcopy(seen_inds)
        seen_inds[0].add(dag_node_ind)
        seen_inds[1].add(dagp_node_ind)
        # parsing the parameters
        dag_nodes = dir_act_graph["nodes"]
        dagp_nodes = dir_act_graph_part["nodes"]
        dag_node = dag_nodes[dag_node_ind]
        dagp_node = dagp_nodes[dagp_node_ind]
        # check for node type
        if isinstance(dag_node, str) != isinstance(dagp_node, str): return False
        # check function nodes
        if not isinstance(dag_node, str): return dag_node[0] == dagp_node[0] and dag_node[1] == dagp_node[1]
        # check both are "and" or "or", check their connections
        if dag_node != dagp_node: return False
        dag_node_conns = dir_act_graph["connections"][dag_node_ind]
        dagp_node_conns = dir_act_graph_part["connections"][dagp_node_ind]
        if len(dag_node_conns) != len(dagp_node_conns): return False
        conn_pairs = set(itertools.product(dag_node_conns, dagp_node_conns))
        dag_dagp_mapping = {}
        for dag_conn, dagp_conn in conn_pairs:
            if dag_conn in seen_inds[0] or dagp_conn in seen_inds[1]: continue
            same_andornode = dfscheck_same_andornode(dir_act_graph, dir_act_graph_part, dag_conn, dagp_conn, seen_inds)
            if dag_conn not in dag_dagp_mapping and same_andornode: dag_dagp_mapping[dag_conn] = dagp_conn
        return dag_node_conns == set(dag_dagp_mapping.keys()) and dagp_node_conns == set(dag_dagp_mapping.values())
    # returns the position of the dir_act_graph_part node in the dir_act_graph
    def dag_pos_of_node(dir_act_graph:dict, dir_act_graph_part:dict, dagp_node_ind:int)->int:
        dagp_node = dir_act_graph_part["nodes"][dagp_node_ind]
        if not isinstance(dagp_node, str):
            hashed_node = hash_node(dagp_node)
            return dir_act_graph["inv_nodes"][hashed_node] if hashed_node in dir_act_graph["inv_nodes"] else -1
        else:
            for i in range(len(dir_act_graph["nodes"])):
                dag_node = dir_act_graph["nodes"][i]
                if (isinstance(dag_node, str)
                    and dfscheck_same_andornode(dir_act_graph, dir_act_graph_part, i, dagp_node_ind)):
                    return i
            return -1
    # find the mapping of indicies from dir_act_graph_part to dir_act_graph, inserting nodes and inv_nodes
    dagp_to_dag_mapping = []
    for dagp_node_ind in range(len(dir_act_graph_part["nodes"])):
        node = dir_act_graph_part["nodes"][dagp_node_ind]
        dag_node_ind = dag_pos_of_node(dir_act_graph, dir_act_graph_part, dagp_node_ind)
        if dag_node_ind >= 0: dagp_to_dag_mapping.append(dag_node_ind)
        else:
            dir_act_graph["nodes"].append(node)
            dir_act_graph["connections"].append(set())
            if not isinstance(node, str): dir_act_graph["inv_nodes"][hash_node(node)] = len(dir_act_graph["nodes"]) - 1
            dagp_to_dag_mapping.append(len(dir_act_graph["nodes"]) - 1)
    # inserting the new connections
    for ind_from in range(len(dir_act_graph_part["connections"])):
        dir_act_graph["connections"][dagp_to_dag_mapping[ind_from]] |=\
            set(dagp_to_dag_mapping[ind_dest] for ind_dest in dir_act_graph_part["connections"][ind_from])   
    # if the entire tree part is not seen before, connect it to the overarching node
    # else, set a new root node with a new overarching node (old overarching node guaranteed to be "and" or "or")
    dag_to_dagp_mapping = [-1 for _ in range(len(dir_act_graph["nodes"]))] # will contain more or equal to the number of nodes dagp has
    for i in range(len(dagp_to_dag_mapping)): dag_to_dagp_mapping[dagp_to_dag_mapping[i]] = i
    if dag_to_dagp_mapping[dir_act_graph["root_ind"]] < 0:
        dir_act_graph["connections"][dir_act_graph["root_ind"]].add(dagp_to_dag_mapping[dir_act_graph_part["root_ind"]])
    else:
        prev_root_ind = dir_act_graph["root_ind"]
        dir_act_graph["root_ind"] = dagp_to_dag_mapping[dir_act_graph_part["root_ind"]]
        # new parent is not the same overarching node
        if dir_act_graph["nodes"][prev_root_ind] != dir_act_graph["nodes"][dir_act_graph["root_ind"]]:
            dir_act_graph["nodes"].append(dir_act_graph["nodes"][prev_root_ind]) # should be an immutable string
            dir_act_graph["connections"].append({dir_act_graph["root_ind"]})
            dir_act_graph["root_ind"] = len(dir_act_graph["nodes"]) - 1
        # add the connection if it wasn't there before, duplicates will not be added to the set
        if dir_act_graph["root_ind"] != prev_root_ind: dir_act_graph["connections"][dir_act_graph["root_ind"]].add(prev_root_ind)
    # returning the result
    return dir_act_graph

# gathers the inverse function call graph represented by the process, constraint processes, and default dependency
def dfsgather_dir_act_graph_process(pro:tuple, constr_links:dict, constr_pros:dict, act_def_deps:dict, action_parameters:dict,
    constr_str_seen:dict[str:dict[tuple:dict]]={}, prev_func_call:tuple=None):
    dir_act_graph = {"nodes": [], "connections": [], "inv_nodes":{}, "root_ind": -1}
    if not pro: return dir_act_graph
    # singular action
    if pro[0] == "single":
        if act_def_deps[pro[1]]:
            action_dep = dfsplace_param_names(act_def_deps[pro[1]], pro[2])
            # action is guaranteed to be in action dependencies
            dir_act_graph = dfsgather_dir_act_graph_dependency(action_dep, constr_links, constr_pros,
                act_def_deps, action_parameters, constr_str_seen)
        # chain the previous graph to this action if need be
        dir_act_graph = update_dir_act_graph_single(dir_act_graph, pro[1], pro[2], bool(act_def_deps[pro[1]]))
        return dir_act_graph
    # "and" or "or"
    dir_act_graph["nodes"].append(pro[0])
    dir_act_graph["connections"].append(set())
    dir_act_graph["root_ind"] = len(dir_act_graph["nodes"]) - 1 # guaranteed to be 0
    for pro_part in pro[1]:
        dir_act_graph_part = dfsgather_dir_act_graph_process(pro_part, constr_links, constr_pros,
            act_def_deps, action_parameters, constr_str_seen, prev_func_call)
        if dir_act_graph_part["nodes"]: dir_act_graph = update_dir_act_graph(dir_act_graph, dir_act_graph_part)
    return dir_act_graph

# helper function that returns the connections between functions, may have loops
# processing actions in a chain, need actions from both ends
# "nodes" with functions or "and" or "or", "connections" with index pairs, "inv_nodes" that link function calls with an index
# connecting nodes backwards and forwards
def dfsgather_dir_act_graph_dependency(dep_orig:tuple,
    constr_links:dict, constr_pros:dict, action_default_deps_orig:dict, action_parameters:dict,
    constr_str_seen:dict[str:dict[tuple:dict]]={})->dict:
    dir_act_graph = {"nodes": [], "connections": [], "inv_nodes":{}, "root_ind": -1}
    # single case, constraints are guaranteed to be in constraint links or constraint processes
    if not dep_orig: return dir_act_graph
    elif dep_orig[0] == "single":
        constr_str = re.sub("not ", "", dep_orig[1])
        if constr_str in constr_links:
            action_name, action_params = get_cl_param_mapping((constr_str, dep_orig[2]), constr_links, action_parameters, constr_str_seen)
            action = ("single", action_name, action_params)
            dir_act_graph = dfsgather_dir_act_graph_process(action, constr_links, constr_pros,
                action_default_deps_orig, action_parameters, constr_str_seen)
        else:
            constr_pro = dfsplace_param_names(constr_pros[constr_str], dep_orig[2])
            action = ("single", dep_orig[1], dep_orig[2])
            dir_act_graph = dfsgather_dir_act_graph_process(constr_pro, constr_links, constr_pros,
                action_default_deps_orig, action_parameters, constr_str_seen, action)
        return dir_act_graph
    # initialize the multiple function call
    inds = None
    node_type = None
    match dep_orig[0]:
        case "and" | "or": node_type = dep_orig[0]
        case "chain" | "gate": node_type = "and" if dep_orig[0] == "chain" else "or"
        case _: raise InvalidConstraintOption(f"invalid dependency option selected: {dep_orig[0]}")
    dir_act_graph["nodes"].append(node_type)
    dir_act_graph["connections"].append(set())
    dir_act_graph["root_ind"] = len(dir_act_graph["nodes"]) - 1 # should be 0
    inds = range(len(dep_orig[1]))
    # loop through all indicies
    for i in inds:
        dep_perm_part = dep_orig[1][i]
        # process the sub part
        dir_act_graph_part = dfsgather_dir_act_graph_dependency(dep_perm_part, constr_links, constr_pros,
            action_default_deps_orig, action_parameters, constr_str_seen)
        # update the graph accordingly, connecting the functions accordingly, guaranteed to be "and" or "or"
        if dir_act_graph_part["nodes"]: dir_act_graph = update_dir_act_graph(dir_act_graph, dir_act_graph_part)
    # removing a node if there is only one action in the "and" or "or", subtracting one from all indicies
    root_old_ind = dir_act_graph["root_ind"]
    if isinstance(dir_act_graph["nodes"][root_old_ind], str) and len(dir_act_graph["connections"][root_old_ind]) == 1:
        # update the new root
        dir_act_graph["root_ind"] = list(dir_act_graph["connections"][root_old_ind])[0]
        if dir_act_graph["root_ind"] > root_old_ind: dir_act_graph["root_ind"] -= 1
        # edit the node list
        dir_act_graph["nodes"].pop(root_old_ind)
        dag_conns = dir_act_graph["connections"]
        dag_conns.pop(root_old_ind)
        dir_act_graph["connections"] = [set(ind_dest-1 if ind_dest > root_old_ind else ind_dest for ind_dest in dag_conns[ind_sour]) for ind_sour in range(len(dag_conns))]
        dir_act_graph["inv_nodes"] = {key: (dir_act_graph["inv_nodes"][key]-1) if dir_act_graph["inv_nodes"][key] > root_old_ind else dir_act_graph["inv_nodes"][key]
            for key in dir_act_graph["inv_nodes"]
            if dir_act_graph["inv_nodes"][key] != root_old_ind}
    # return the graph
    return dir_act_graph

# dfs gathers the inverse function call directed graph, passed in dependency has the constraint links and dependencies inserted
class InvalidGraphSelfLoop(Exception): pass
def dfsgather_directedactiongraph(dep_orig:tuple,
    constr_links:dict, constr_pros:dict, action_default_deps_orig:dict, action_parameters:dict,
    action_user_goal:tuple)->dict[str:list]:
    # find the connections that make up the function call graph
    connections = dfsgather_dir_act_graph_dependency(dep_orig, constr_links, constr_pros, action_default_deps_orig, action_parameters)
    # adding the user_goal into the front
    dir_act_graph = copy.deepcopy(connections)
    if not dir_act_graph["nodes"] or action_user_goal[0] != dir_act_graph["nodes"][dir_act_graph["root_ind"]][0]:
        dir_act_graph["nodes"].append(action_user_goal)
        if dir_act_graph["root_ind"] >= 0: dir_act_graph["connections"].append({dir_act_graph["root_ind"]})
        else: dir_act_graph["connections"].append(set())
    del dir_act_graph["inv_nodes"]
    del dir_act_graph["root_ind"]
    # renumbering the indicies for better readability, prioritizing the longest distance from the start
    dir_graph_branches = dir_act_graph["connections"]
    renumber_mapping = [-1 for _ in range(len(dir_act_graph["nodes"]))]
    queue_ind = deque([(len(dir_act_graph["nodes"])-1, 0)]) # insert right because stack
    branches = [set()] # each path through the graph has a chain, tracks visited node indicies
    counter_dist = 0
    while queue_ind:
        ind, branch_num = queue_ind.popleft()
        # cycle prevention
        if ind in branches[branch_num]: continue
        branches[branch_num].add(ind)
        # pushes "and" and "or" nodes towards the end of the numbering
        neighbor_inds = []
        neighbors_andor_startpos = 0
        for neighbor_ind in dir_graph_branches[ind]:
            if not isinstance(dir_act_graph["nodes"][neighbor_ind], str):
                neighbor_inds.insert(neighbors_andor_startpos, neighbor_ind)
                neighbors_andor_startpos += 1
            else: neighbor_inds.append(neighbor_ind)
        # extend the queue with the neighboring indexes, keep track of visited nodes
        if neighbor_inds:
            neighbors_ind_branchnum = [(neighbor_inds[0], branch_num)]
            for i in range(1, len(neighbor_inds)):
                branches.append(branches[branch_num].copy())
                neighbors_ind_branchnum.append((neighbor_inds[i], len(branches)-1))
            queue_ind.extend(neighbors_ind_branchnum)
        # recording the distance in the chain from the start
        renumber_mapping[ind] = counter_dist
        counter_dist += 1
    renumber_mapping_sorted_dest = sorted([(renumber_mapping[ind], ind) for ind in range(len(renumber_mapping))])
    mapping_new_to_old = [old_ind for _, old_ind in renumber_mapping_sorted_dest]
    mapping_old_to_new = [-1 for _ in range(len(mapping_new_to_old))]
    for i in range(len(mapping_new_to_old)): mapping_old_to_new[mapping_new_to_old[i]] = i
    # mapping from old indicies to new indicies
    dag_old = dir_act_graph
    dir_act_graph = {"nodes": [-1 for _ in range(len(dag_old["nodes"]))], "connections": [set() for _ in range(len(dag_old["nodes"]))]}
    for old_ind in range(len(mapping_old_to_new)): dir_act_graph["nodes"][mapping_old_to_new[old_ind]] = dag_old["nodes"][old_ind]
    for old_ind1 in range(len(dag_old["connections"])):
        dir_act_graph["connections"][mapping_old_to_new[old_ind1]] =\
            sorted(list(mapping_old_to_new[old_ind2] for old_ind2 in dag_old["connections"][old_ind1]))
    # list of connections
    dir_act_graph["connections"] = [(ind_sour, ind_dest)
        for ind_sour in range(len(dir_act_graph["connections"]))
        for ind_dest in dir_act_graph["connections"][ind_sour]]
    # small sanity check: self loops
    for ind_sour, ind_dest in dir_act_graph["connections"]:
        if ind_sour == ind_dest: raise InvalidGraphSelfLoop("node self-loop detected in the inverse function call graph")
    # return the graph
    return dir_act_graph


"""functions that are not used for task_generation, but are highly relevant and are used elsewhere"""

# gets the connections (position to multiple positions) and inverse nodes (function name to position)
def get_dag_connections_invnodes(dir_act_graph:dict)->tuple[list[list],dict]:
    dag_n = dir_act_graph["nodes"]
    dag_c = dir_act_graph["connections"]
    # put the connections into a 2D list
    connections = [[] for _ in range(len(dag_n))]
    for conn_from, conn_to in dag_c: connections[conn_from].append(conn_to)
    # inverse nodes to quickly find certain functions
    inv_nodes = {}
    for i in range(len(dag_n)):
        node = dag_n[i]
        if isinstance(node, str): continue
        fname = node[0]
        if fname not in inv_nodes: inv_nodes[fname] = i
    # return the connectiosn and inverse nodes
    return connections, inv_nodes

# gathers the inverse function call directed graph for a function of a domain, given that the action is a part of the domain
def dfsgather_dag_func(domain_system, domain_assistant:dict, action:str, default_dependency_option:str)->tuple[list,list,dict]:
    if action not in domain_assistant.action_descriptions: return None
    # variable loading
    ard = domain_assistant.action_required_dependencies
    acd = domain_assistant.action_customizable_dependencies
    cl = domain_assistant.constraint_links
    cp = domain_assistant.constraint_processes
    action_default_dep_orig = gather_action_default_dependencies(ard, acd, default_dependency_option=default_dependency_option)
    action_parameters = get_action_parameters(domain_system, domain_assistant)
    # process the graph
    dep_orig = action_default_dep_orig[action]
    user_goal_node = (action, {key: key for key in action_parameters[action]})
    dir_act_graph = dfsgather_directedactiongraph(dep_orig, cl, cp, action_default_dep_orig, action_parameters, user_goal_node)
    nodes = dir_act_graph["nodes"]
    connections, inv_nodes = get_dag_connections_invnodes(dir_act_graph)
    return nodes, connections, inv_nodes


"""functions that are not originally in task_generation_helpers"""

# base cases for fail: one is a value and one is an instance, parameter within a parameter, either could be None
# recursive cases for fail: two instance formats differ, two of the same parameter formats differ
def recur_data_consistency(d1, d2)->bool:
    if not d1 or not d2 or not isinstance(d1, dict) and not isinstance(d2, dict):
        return True # neither are instances (always dict)
    res = True
    d1_keys = d1.keys()
    d2_keys = d2.keys()
    for d1_key in d1_keys:
        # keys are the different: d1 and d2 are parameters
        if d1_keys != d2_keys:
            for d2_key in d2_keys:
                if ((isinstance(d1[d1_key], dict) and isinstance(d2[d2_key], dict) and d1[d1_key].keys() != d2[d2_key].keys())
                    or (isinstance(d1[d1_key], dict) or isinstance(d2[d2_key], dict)) and type(d1[d1_key]) != type(d2[d2_key])):
                    return False # parameter within a parameter case
                res = res and recur_data_consistency(d1[d1_key], d2[d2_key])
        # keys are the same: d1 and d2 are instances
        else:
            if (isinstance(d1[d1_key], dict) or isinstance(d2[d1_key], dict)) and type(d1[d1_key]) != type(d2[d1_key]):
                return True # instances could have parameters with different types dict and str
            res = res and recur_data_consistency(d1[d1_key], d2[d1_key])
    return res

# writes to a file
import os
def write_data_file(data_dir:str, document_name:str, data:str, option:str='w'):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    with open(os.path.join(data_dir, document_name), option, encoding='utf-8', errors='ignore') as f:
        f.write(data)