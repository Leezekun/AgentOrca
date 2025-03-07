"""file for redirecting variables"""

# string domain into domain assistant file
from env.domains.bank import bank_assistant
from env.domains.online_market import online_market_assistant
from env.domains.healthcare import healthcare_assistant
from env.domains.dmv import dmv_assistant
from env.domains.library import library_assistant
domain_assistant_keys = {
    "bank": bank_assistant,
    "online_market": online_market_assistant,
    "dmv": dmv_assistant,
    "healthcare": healthcare_assistant,
    "library": library_assistant,
}

from env.domains.bank.bank import Bank, Bank_w_Dependency_Verifier
from env.domains.dmv.dmv import DMV, DMV_w_Dependency_Verifier
from env.domains.healthcare.healthcare import Healthcare, Healthcare_w_Dependency_Verifier
from env.domains.online_market.online_market import OnlineMarket, OnlineMarket_w_Dependency_Verifier
from env.domains.library.library import Library, Library_w_Dependency_Verifier
domain_keys = {
    "bank": Bank,
    "bank_strict": Bank_w_Dependency_Verifier,
    "dmv": DMV,
    "dmv_strict": DMV_w_Dependency_Verifier,
    "healthcare": Healthcare,
    "healthcare_strict": Healthcare_w_Dependency_Verifier,
    "library": Library,
    "library_strict": Library_w_Dependency_Verifier,
    "online_market": OnlineMarket,
    "online_market_strict": OnlineMarket_w_Dependency_Verifier,
}