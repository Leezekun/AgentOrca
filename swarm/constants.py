# Project directory
PROJ_DIR = "/xxx/AgentOrca/"

# OpenAI model names
OPENAI_MODELS = [
    # GPT-4 variants
    "gpt-4o", # -> gpt-4o-2024-08-06
    "gpt-4o-2024-11-20",
    
    # GPT-4 Mini variants 
    "gpt-4o-mini", # -> gpt-4o-mini-2024-07-18
    
    # o1
    "o1", # -> o1-2024-12-17
    "o3-mini", # -> o3-mini-2025-01-31
]

# Gemini models
GEMINI_MODELS = [
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-2.0-flash-001",
    "gemini-2.0-flash",
    "gemini-2.0-flash-thinking-exp",
]

# Claude models
CLAUDE_MODELS = [
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
]

# Fireworks models
FIREWORKS_MODELS = {
    "llama3.1-405b-instruct": "accounts/fireworks/models/llama-v3p1-405b-instruct", # 3.0/M
    "llama3.1-70b-instruct": "accounts/fireworks/models/llama-v3p1-70b-instruct", # 0.9/M
    "llama3.3-70b-instruct": "accounts/fireworks/models/llama-v3p3-70b-instruct", # 0.9/M
    "qwen2.5-72b-instruct": "accounts/fireworks/models/qwen2p5-72b-instruct", # 0.9/M
    "qwen2.5-coder-32b-instruct": "accounts/fireworks/models/qwen2p5-coder-32b-instruct", # 0.9/M
    "deepseek-v3": "accounts/fireworks/models/deepseek-v3", # 0.9/M
    "deepseek-r1": "accounts/fireworks/models/deepseek-r1", # 3.0/M
    "mistral-8x22b-instruct": "accounts/fireworks/models/mixtral-8x22b-instruct", # 1.2/M
}

# 7B-32B Models
OSS_MODELS = {
    "qwen2.5-coder-7b-instruct": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-14b-instruct": "Qwen/Qwen2.5-14B-Instruct",
    "qwen2.5-32b-instruct": "Qwen/Qwen2.5-32B-Instruct",
    "llama3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "toolace-8b": "Team-ACE/ToolACE-8B",
}

# Tool chat parser mappings
# Refer to vLLM: https://docs.vllm.ai/en/stable/serving/openai_compatible_server.html#tool-calling-in-the-chat-completion-api
TOOL_CHAT_PARSERS = {
    # Hermes 2-Pro, 3, and Qwen2.5 series
    "hermes": [
        # Hermes models
        "NousResearch/Hermes-2-Pro-Mistral-7B",
        "NousResearch/Hermes-2-Pro-Llama-3-8B",
        "NousResearch/Hermes-2-Pro-Llama-3-70B",
        "NousResearch/Hermes-3-Llama-3.1-8B",
        "NousResearch/Hermes-3-Llama-3.1-70B",
        
        # Qwen2.5 models
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-14B-Instruct",
        "Qwen/Qwen2.5-32B-Instruct",
        "Qwen/Qwen2.5-72B-Instruct",
    ],
    
    # Mistral models
    "mistral": [
        "mistralai/Mistral-7B-Instruct-v0.3"
    ],
    
    # LLaMA 3.1 models
    "llama3_json": [
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "meta-llama/Meta-Llama-3.3-70B-Instruct",
    ],
    
    # Granite models
    "granite": [
        "ibm-granite/granite-3.0-8b-instruct"
    ],
    "granite-20b-fc": [
        "ibm-granite/granite-20b-functioncalling"
    ],
    
    # InternLM models
    "internlm": [
        "internlm/internlm2_5-7b-chat"
    ],
    
    # Tool-specialized models
    "pythonic": [
        "Team-ACE/ToolACE-8B",
        "meta-llama/Meta-Llama-3.2-1B-Instruct",
        "meta-llama/Meta-Llama-3.2-3B-Instruct"
    ],
}

# Template paths for different model families (based on the model name)
TOOL_CHAT_TEMPLATES = {
    # Granite templates
    "granite-20b-fc": f"{PROJ_DIR}/swarm/templates/tool_chat_template_granite_20b_fc.jinja",
    "granite": f"{PROJ_DIR}/swarm/templates/tool_chat_template_granite.jinja",
    
    # Other model family templates
    "internlm": f"{PROJ_DIR}/swarm/templates/tool_chat_template_internlm2.jinja",
    "mistral": f"{PROJ_DIR}/swarm/templates/tool_chat_template_mistral.jinja",
    "hermes": f"{PROJ_DIR}/swarm/templates/tool_chat_template_hermes.jinja",
    "llama3.1": f"{PROJ_DIR}/swarm/templates/tool_chat_template_llama3.1.jinja",
    "llama3.2": f"{PROJ_DIR}/swarm/templates/tool_chat_template_llama3.2_pythonic.jinja",
    "toolace": f"{PROJ_DIR}/swarm/templates/tool_chat_template_toolace.jinja",
}

# Available Models
AVAILABLE_MODELS = {
    "vllm": [
        "qwen2.5-7b-instruct",
        "qwen2.5-14b-instruct",
        "qwen2.5-32b-instruct",
        "llama3.1-8b-instruct",
        "toolace-8b",
        "qwen2.5-coder-7b-instruct",
        "qwen2.5-coder-32b-instruct",
    ],
    "fireworks": [
        "llama3.1-405b-instruct",
        "llama3.1-70b-instruct",
        "llama3.3-70b-instruct",
        "qwen2.5-72b-instruct",
        "mistral-8x22b-instruct",
        "qwen2.5-coder-32b-instruct",
        "deepseek-v3",
        "deepseek-r1",
    ],
    "gemini": [
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-2.0-flash-001",
        "gemini-2.0-flash",
        "gemini-2.0-flash-thinking-exp",
    ],
    "claude": [
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
    ],
    "openai": [
        "gpt-4o",
        "gpt-4o-2024-11-20",
        "gpt-4o-mini",
        "o1",
        "o3-mini",
    ]
}

# Model Class That Supports Function Calling
FUNCTION_CALLING_MODELS = {
    "vllm": [
        "qwen2.5-7b-instruct",
        "qwen2.5-14b-instruct",
        "qwen2.5-32b-instruct",
        "llama3.1-8b-instruct",
        "toolace-8b"
    ],
    "fireworks": [
        "llama3.1-405b-instruct",
        "llama3.1-70b-instruct",
        "qwen2.5-72b-instruct",
        "mistral-8x22b-instruct",
    ],
    "gemini": [
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-2.0-flash-001",
        "gemini-2.0-flash",
    ],
    "claude": [
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
    ],
    "openai": [
        "gpt-4o",
        "gpt-4o-2024-11-20",
        "gpt-4o-mini",
        "o1",
        "o3-mini",
    ]
}