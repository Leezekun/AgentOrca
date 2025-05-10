"""
Appendix B
This document handles the domain action and constraint descriptions and constraint and action compositions, respectively.
"""

from env.variables import domain_keys, domain_assistant_keys
from env.helpers import hashable_dep, dfsremove_if_unnecessary, dfsprune_dep_pro, gather_action_default_dependencies

import re

bank_rac = {
    "internal_check_credit_card_exist": "internal_credit_card_exist",
    "internal_check_foreign_currency_available": "internal_foreign_curr_avail",
    "internal_check_username_exist": "internal_user_exist",
    "internal_get_database": "REMOVE",
    "call_get_database": "REMOVE",
    "get_bank_maximum_loan_amount": "REMOVE",
}

dmv_rac = {
    'internal_check_test_slot_available': "internal_test_slot_avail",
    "internal_check_username_exist": "internal_user_exist",
    "internal_get_database": "REMOVE",
}

healthcare_rac = {
    "internal_get_database": "REMOVE",
}

library_rac = {
    "internal_get_database": "REMOVE",
    "internal_all_slots_available_for_the_room_on_the_date": "internal_room_slot_avail",
    "internal_check_date_available_for_the_room": "internal_room_date_avail",
    "sufficient_account_balance_for_late_fee": "suff_acc_bal_late_fee",
    "sufficient_account_balance_for_membership": "suff_acc_bal_mem",
}

online_market_rac = {
    "internal_get_database": "REMOVE",
    "credit_status_not_restricted_or_suspended": "credit_status_good",
    "not_already_added_shipping_address": "not_shipping_addr_exist"
}

rename_action_constraint = {
    "bank": bank_rac,
    "dmv": dmv_rac,
    "healthcare": healthcare_rac,
    "library": library_rac,
    "online_market": online_market_rac
}

def remove_and_prune(dep:tuple, rdep:tuple)->tuple:
    ndep = dfsremove_if_unnecessary(dep, {hashable_dep(rdep)}, dep[0], dep[0], force_remove=True)
    return dfsprune_dep_pro(ndep)

def dfsget_ad_cp_str(pro:tuple, domain_str:str)->str:
    if not pro: return "None"
    if pro[0] == "single":
        pro_part_str = pro[1]
        if pro_part_str in rename_action_constraint[domain_str]:
            pro_part_str = rename_action_constraint[domain_str][pro_part_str]
        return pro_part_str
    list_pro_str = []
    for pro_part in pro[1]:
        pro_part_str = dfsget_ad_cp_str(pro_part, domain_str)
        if pro_part[0] == "single":
            if pro_part_str in rename_action_constraint[domain_str]:
                list_pro_str.append(rename_action_constraint[domain_str][pro_part_str])
            else: list_pro_str.append(pro_part_str)
        else: list_pro_str.append(f"({pro_part_str})")
    join_str = ""
    match pro[0]:
        case "and": join_str = " AND "
        case "or": join_str = " OR "
        case "chain": join_str = " THEN "
        case "gate": join_str = " IF NOT, THEN "
    return join_str.join(list_pro_str)

def print_action_descriptions(domain_str:str):
    domain_system = domain_keys[domain_str]()
    domain_assistant = domain_assistant_keys[domain_str]
    list_action_name = sorted([func for func in dir(domain_system)
        if callable(getattr(domain_system, func)) and not func.startswith("_") and not func.startswith("evaluation_")])
    for action_name in list_action_name:
        action_description = domain_assistant.action_descriptions[action_name]
        if action_name in rename_action_constraint[domain_str]:
            action_name = rename_action_constraint[domain_str][action_name]
        action_name = re.sub("_", "\\_", action_name)
        print(f"{action_name} & {action_description} \\\\")
    print()
    
def print_action_dependencies(domain_str:str):
    domain_assistant = domain_assistant_keys[domain_str]
    ard = domain_assistant.action_required_dependencies
    acd = domain_assistant.action_customizable_dependencies
    ad = gather_action_default_dependencies(ard, acd, None, "full")
    actions = sorted([action for action in ad.keys()])
    for action in actions:
        ad_action = dfsget_ad_cp_str(ad[action], domain_str)
        if action in rename_action_constraint[domain_str]:
            action = rename_action_constraint[domain_str][action]
        display_str = f"{action} & {ad_action} \\\\"
        display_str = re.sub("_", "\\_", display_str)
        print(display_str)
    print()

def print_constraint_descriptions(domain_str:str):
    domain_assistant = domain_assistant_keys[domain_str]
    pcd = domain_assistant.positive_constraint_descriptions
    pcd_keys = sorted(list(pcd.keys()))
    for constraint in pcd_keys:
        constraint_description = pcd[constraint]
        if constraint in rename_action_constraint[domain_str]:
            constraint = rename_action_constraint[domain_str][constraint]
        display_str = f"{constraint} & {constraint_description}\\\\"
        display_str = re.sub("_", "\\_", display_str)
        display_str = re.sub("{", "", display_str)
        display_str = re.sub("}", "", display_str)
        print(display_str)
    print()
    
def print_constraint_processes(domain_str:str):
    domain_assistant = domain_assistant_keys[domain_str]
    pcd = domain_assistant.positive_constraint_descriptions
    cp = domain_assistant.constraint_processes
    cl = domain_assistant.constraint_links
    pcd_keys = sorted(list(pcd.keys()))
    for constraint in pcd_keys:
        constraint_process_str = None
        if constraint in cp:
            cp_pruned = remove_and_prune(cp[constraint], ("single", "internal_get_database", {}))
            constraint_process_str = dfsget_ad_cp_str(cp_pruned, domain_str)
        elif constraint in cl:
            constraint_process_str = cl[constraint][0]
        else:
            constraint_process_str = constraint
        if constraint in rename_action_constraint[domain_str]:
            constraint = rename_action_constraint[domain_str][constraint]
        display_str = f"{constraint} & {constraint_process_str}\\\\"
        display_str = re.sub("_", "\\_", display_str)
        print(display_str)
    print()

def paper_display_info(args):
    print_action_descriptions(args.domain_str)
    print_action_dependencies(args.domain_str)
    print_constraint_descriptions(args.domain_str)
    print_constraint_processes(args.domain_str)
