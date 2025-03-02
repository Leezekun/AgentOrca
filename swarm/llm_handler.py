from typing import List, Dict, Any, Optional
from openai import OpenAI
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import random
import json
import requests
from tqdm import tqdm
import copy
from typing import Literal
from swarm.gemini import gemini_chat_completion_openai_format
from swarm.claude import claude_chat_completion_openai_format
from swarm.constants import (OPENAI_MODELS, 
                             GEMINI_MODELS, 
                             CLAUDE_MODELS, 
                             FIREWORKS_MODELS,
                             OSS_MODELS, 
                             TOOL_CHAT_PARSERS, 
                             TOOL_CHAT_TEMPLATES,
                             FUNCTION_CALLING_MODELS,
                             AVAILABLE_MODELS)
from swarm.util import display_messages
from dotenv import load_dotenv
load_dotenv()

class OpenAIHandler:
    """Handles interactions with OpenAI and VLLM-based models."""
    
    def __init__(
        self,
        model_name: str,
        num_gpus: int = 1,
        gpu_memory_utilization: float = 0.9,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int = 1024,
        lora_path: str = "",
        tool_calling: bool = False,
        dtype: str = "bfloat16",
    ) -> None:
        """
        Initialize the OpenAI handler.
        
        Args:
            model_name: Name of the model to use
            backend: Either "vllm" or "openai"
            num_gpus: Number of GPUs to use for VLLM
            gpu_memory_utilization: GPU memory utilization for VLLM
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum tokens to generate
            lora_path: Path to LoRA weights
            dtype: Data type for model weights
        """
        self.model_name = model_name # the short name of the model
        self.model_name_huggingface = model_name # the complete name of the model
        self.lora_path = lora_path
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.dtype = dtype
        self.tool_calling = tool_calling
        self.process = None
        
        # Check model support with case-insensitive comparison
        if model_name in OPENAI_MODELS:
            self.backend = "openai"
        elif model_name in GEMINI_MODELS:
            self.backend = "gemini"
        elif model_name in CLAUDE_MODELS:
            self.backend = "claude"
        elif model_name in FIREWORKS_MODELS:
            self.backend = "fireworks"
        elif model_name in OSS_MODELS:
            self.backend = "vllm"
        else:
            raise ValueError(f"Model {model_name} is not supported.")
        
        # Initialize the backend and the client
        if self.backend == "vllm":
            self.model_name_huggingface = OSS_MODELS[model_name]
            self._init_vllm(num_gpus, gpu_memory_utilization)
        elif self.backend == "fireworks":
            self.model_name_huggingface = FIREWORKS_MODELS[model_name]
            self._init_fireworks()
        elif self.backend == "openai":
            self._init_openai()
        elif self.backend == "claude":
            self._init_claude()
        elif self.backend == "gemini":
            self._init_gemini()

    def _init_openai(self) -> None:
        """Initialize OpenAI backend."""
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
    
    def _init_fireworks(self) -> None:
        self.client = OpenAI(
            base_url="https://api.fireworks.ai/inference/v1",
            api_key=os.getenv("FIREWORKS_API_KEY")
        )
    
    def _init_claude(self) -> None:
        self.client = claude_chat_completion_openai_format
    
    def _init_gemini(self) -> None:
        self.client = gemini_chat_completion_openai_format
    
    def _init_vllm(self, num_gpus: int, gpu_memory_utilization: float) -> None:
        """Initialize VLLM backend."""
        self.VLLM_PORT = random.randint(1000, 2000)
        self.client = OpenAI(
            base_url=f"http://localhost:{self.VLLM_PORT}/v1", 
            api_key="EMPTY"
        )

        # Prepare VLLM command
        vllm_cmd = [
            "vllm",
            "serve",
            str(self.model_name_huggingface),
            "--port", str(self.VLLM_PORT),
            "--dtype", str(self.dtype),
            "--tensor-parallel-size", str(num_gpus),
            "--gpu-memory-utilization", str(gpu_memory_utilization),
            "--trust-remote-code",
            "--max-model-len"
        ]
        
        # Add the max model length
        if "gemma" in self.model_name_huggingface.lower():
            vllm_cmd.append("4096")
        elif "mistral" in self.model_name_huggingface.lower():
            vllm_cmd.append("8192")
        elif "llama-3-" in self.model_name_huggingface.lower():
            vllm_cmd.append("8192")
        else:
            vllm_cmd.append("8192")
        
        # Check if tool calling is enabled
        if self.tool_calling:
            # Find the corresponding tool chat template
            tool_chat_parser, tool_chat_template = self._get_tool_chat_setup()
            
            if tool_chat_parser is not None:
                vllm_cmd.extend([
                    "--enable-auto-tool-choice",
                    "--tool-call-parser", tool_chat_parser,
                ])
            # The tool chat template is optional
            if tool_chat_template is not None:
                vllm_cmd.extend([
                    "--chat-template", tool_chat_template
                ])

        # Check if LoRA is enabled  
        if self.lora_path:
            vllm_cmd.extend([
                "--enable-lora",
                "--lora-modules",
                f"sql-lora={self.lora_path}"
            ])

        # Start the server
        self.process = subprocess.Popen(
            vllm_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Wait for the server to start
        self._wait_for_server()

    def _wait_for_server(self, max_retries: int = 15, retry_delay: int = 30) -> None:
        """Wait for VLLM server to start."""
        stop_event = threading.Event()

        # Start logging threads
        for pipe in [self.process.stdout, self.process.stderr]:
            thread = threading.Thread(
                target=self._log_subprocess_output,
                args=(pipe, stop_event)
            )
            thread.start()

        # Wait for server
        for retry in range(max_retries):
            try:
                response = requests.get(f"http://localhost:{self.VLLM_PORT}/v1/models")
                if response.status_code == 200:
                    print("Server is ready!")
                    stop_event.set()
                    return
            except requests.exceptions.ConnectionError:
                print(f"Server is not ready yet. Trying {retry+1} times...")
                time.sleep(retry_delay)

        raise ConnectionError(f"Server not ready after {max_retries} retries.")

    @staticmethod
    def _log_subprocess_output(pipe, stop_event):
        """Log subprocess output until stop event is set."""
        for line in iter(pipe.readline, ""):
            if stop_event.is_set():
                break
            print(line, end="")
        pipe.close()

    def kill_process(self):
        """
        Kill the server process.
        """
        if self.process:
            self.process.terminate()
            self.process.wait()
            print("Server process terminated.")

    def _get_tool_chat_setup(self) -> str:
        """Get the appropriate tool chat parser and template for the model.
        
        Returns:
            tuple: (tool_chat_parser, tool_chat_template)
        
        Raises:
            ValueError: If the model is not supported
        """
        # First, get the tool chat parser. Ensure that the model is supported.
        tool_chat_parser = None
        for parser, supported_models in TOOL_CHAT_PARSERS.items():
            if self.model_name_huggingface in supported_models:
                tool_chat_parser = parser
                break
        
        if tool_chat_parser is None:
            raise ValueError(f"Model {self.model_name_huggingface} is not supported.")
        
        # Then, get the tool chat template. The template can be None.
        tool_chat_template = None
        for model_family, template in TOOL_CHAT_TEMPLATES.items():
            if model_family in self.model_name_huggingface.lower():
                tool_chat_template = template
                break
        
        return (tool_chat_parser, tool_chat_template)
    
    def chat_completion(self, test_entry: dict, include_debugging_log: bool, tool_call_mode: Literal["fc", "prompt", "react", "act-only"] = "fc"):
        """
        OSS models have a different inference method.
        They need to spin up a server first and then send requests to it.
        It is more efficient to spin up the server once for the whole batch, instead of for each individual entry.
        So we implement batch_inference method instead.
        """        
        # remove the "tool_name" field for real openai format messages for chat completion
        norm_messages = copy.deepcopy(test_entry["messages"])
        if self.backend in ["openai", "vllm", "fireworks"]:
            for message in norm_messages:
                if message["role"] == "tool":
                    # openai format does not have tool_name
                    del message["tool_name"] 
                if message["role"] == "user":
                    if "sender" in message:
                        del message["sender"]
                    if "tool_calls" in message:
                        del message["tool_calls"]
                        
                # remove the sender field
                if "sender" in message:
                    del message["sender"]
                # remove the unneeded fields for non-openai format messages
                if self.backend != "openai" and message["role"] == "assistant":
                    if "function_call" in message:
                        del message["function_call"]
                    if "refusal" in message:
                        del message["refusal"]
                    if "audio" in message:
                        del message["audio"]

        # # For debugging
        # print("--------------------------------")
        # display_messages(norm_messages)
        # print("--------------------------------")
        
        # Create base parameters for completion request
        chat_completion_params = {
            "model": self.model_name_huggingface,
            "messages": norm_messages,
            "temperature": test_entry.get("temperature", self.temperature),
            "n": test_entry.get("n", 1),
            "top_p": test_entry.get("top_p", self.top_p),
            "max_tokens": test_entry.get("max_tokens", self.max_tokens),
            "logprobs": test_entry.get("logprobs", False),
            "stop": test_entry.get("stop", None)
        }
        
        # o1 and o3 do not support a few parameters
        if self.model_name_huggingface in ["o3-mini", "o1-mini", "o1", "o3"]:
            chat_completion_params["max_completion_tokens"] = chat_completion_params["max_tokens"]
            del chat_completion_params["max_tokens"]
            del chat_completion_params["temperature"]
            del chat_completion_params["top_p"]
        
        ######################################################################
        # TOOL CALLING: Different ways to call tools and take actions
        ######################################################################
        tools = test_entry.get("tools", None)
        if not tools:
            if self.backend in ["openai", "vllm", "fireworks"]:
                completion = self.client.chat.completions.create(**chat_completion_params)
            elif self.backend in ["gemini", "claude"]:
                completion = self.client(**chat_completion_params)
            else:
                raise ValueError(f"Model {self.model_name_huggingface} is not supported.")
            return {"idx": test_entry.get("idx", 0), "completion": completion}
        
        ######################################################################
        # FC or Prompt-based tool calling
        ######################################################################
        # FC-based tool calling
        if tool_call_mode == "fc":
            assert self.model_name in FUNCTION_CALLING_MODELS[self.backend], f"Model {self.model_name} is not supported for tool calling."
            chat_completion_params["tools"] = tools
            if self.backend in ["openai", "vllm"]:
                if self.model_name_huggingface not in ["o3-mini", "o1-mini", "o1", "o3"]: # o1 and o3 do not support parallel tool calls
                    chat_completion_params["parallel_tool_calls"] = test_entry.get("parallel_tool_calls", False)
                completion = self.client.chat.completions.create(**chat_completion_params)
            elif self.backend in ["fireworks"]:
                # Does not support "strict" field in tool description
                norm_tools = copy.deepcopy(tools)
                for tool in norm_tools:
                    if "strict" in tool["function"]:
                        del tool["function"]["strict"]
                chat_completion_params["tools"] = norm_tools
                completion = self.client.chat.completions.create(**chat_completion_params)
            elif self.backend in ["gemini", "claude"]:
                completion = self.client(**chat_completion_params)
            else:
                raise ValueError(f"Model {self.model_name_huggingface} is not supported.")
            return {"idx": test_entry.get("idx", 0), "completion": completion}
        
        
        # Prompt-based tool calling (react or act-only), 
        # Need input formatter and output parser, and change into FC format    
        elif tool_call_mode in ["react", "act-only"]:            
            if self.backend in ["openai", "vllm", "fireworks"]:
                chat_completion_func = self.client.chat.completions.create
            elif self.backend in ["gemini", "claude"]:
                chat_completion_func = self.client
            else:
                raise ValueError(f"Model {self.model_name_huggingface} is not supported.")
            
            # Call ReAct tool calling with planning mode if specified
            from swarm.ReAct import ReAct_tool_calling
            completion = ReAct_tool_calling(
                chat_completion_func=chat_completion_func,
                chat_completion_params=chat_completion_params,
                messages=norm_messages,
                tools=tools,
                reasoning=(tool_call_mode != "act-only")
            )
            return {"idx": test_entry.get("idx", 0), "completion": completion}
        
        else:
            raise ValueError(f"Tool call mode {tool_call_mode} is not supported.")
        

    def text_completion(self, test_entry: dict, include_debugging_log: bool):
        """
        OSS models have a different inference method.
        They need to spin up a server first and then send requests to it.
        It is more efficient to spin up the server once for the whole batch, instead of for each individual entry.
        So we implement batch_inference method instead.
        """
        # Fix: Use the correct method for creating a completion request
        completion = self.client.completions.create(
            model=self.model_name_huggingface,
            prompt=test_entry["prompt"],
            echo=False,
            temperature=test_entry.get("temperature", self.temperature),
            n=test_entry.get("n", 1),
            top_p=test_entry.get("top_p", self.top_p),
            max_tokens=test_entry.get("max_tokens", self.max_tokens),
            logprobs=test_entry.get("logprobs", False),
        )
        return {"idx": test_entry.get("idx", 0), "completion": completion}

    def inference(
        self, test_entry: dict, include_debugging_log: bool, mode: str = "chat", tool_call_mode: str = "fc"
    ):
        if mode == "chat":
            return self.chat_completion(test_entry, include_debugging_log, tool_call_mode)
        elif mode == "text":
            return self.text_completion(test_entry, include_debugging_log)
        else:
            raise ValueError(f"Mode {mode} is not supported.")

    def batch_inference(
        self,
        test_entries: List[dict],
        include_debugging_log: bool,
        mode: str = "chat",
        num_workers: int = 8,
    ) -> List[List[str]]:
        # Once the server is ready, make the completion requests
        results = []
        futures = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # with tqdm(total=len(test_entries)) as pbar:
            for idx, test_case in enumerate(test_entries):
                test_case["idx"] = idx
                future = executor.submit(
                    self.inference,
                    test_case,
                    include_debugging_log,
                    mode,
                )
                futures.append(future)

            for future in futures:
                # This will wait for the task to complete, so that we are always writing in order
                result = future.result()
                results.append(result)
                # pbar.update()
                
        # reorder the results based on the original order
        results = sorted(results, key=lambda x: x["idx"])
        results = [res["completion"] for res in results]
        return results



if __name__ == "__main__":
    
    tools = [
    {
        "type": "function",
        "function": {
            "name": "get_delivery_date",
            "description": "Get the delivery date for a customer's order. Call this whenever you need to know the delivery date.'",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "anyOf": [
                            {
                                "type": "string",
                                "description": "The customer's order ID.",
                            },
                            {
                                "type": "object",
                                "description": "The order person and name in the format of YYYY-MM-DD.",
                                "properties": {
                                    "order_person": {
                                        "type": "string",
                                        "description": "The order person.",
                                    },
                                    "order_name": {
                                        "type": "string",
                                        "description": "The name of the order.",
                                    }
                                },
                            }
                        ]
                    },
                },
                "required": ["order_id"],
            },
        }
        },
    {
        "type": "function",
        "function": {
            "name": "set_reminder",
            "description": "Get the delivery date for a customer's order. Call this whenever you need to know the delivery date.'",
            "parameters": {
                "type": "object",
                "properties": {
                    "reminder_time": {
                        "type": "string",
                        "description": "The time to set the reminder in the format of YYYY-MM-DD.",
                    },
                    "reminder_content": {
                        "type": "string",
                        "description": "The content of the reminder.",
                        },
                    },
                    "required": ["reminder_time", "reminder_content"],
                },
            }
        }
    ]

    messages = [
        {
            "role": "system",
            "content": "You are a helpful customer support assistant. Use the supplied tools to assist the user."
        },
        {
            "role": "user",
            "content": "Hi, can you tell me the delivery date for my order? The order person is John Doe and the order name is Dominos Pizza. Also, set a reminder for me to check the order at that time."
        },
        # {
        #     "role": "assistant",
        #     "content": None,
        #     "tool_calls": [
        #         {
        #             "id": "call_sNcq3LV89bWJCWgZvJpCac7I",
        #             "function": {
        #                 "arguments": "{\"order_person\":\"John Doe\",\"order_name\":\"Dominos Pizza\"}",
        #                 "name": "get_delivery_date"
        #             },
        #             "type": "function"
        #         }
        #     ]
        # },
        # {
        #     "role": "tool",
        #     "tool_call_id": "call_sNcq3LV89bWJCWgZvJpCac7I",
        #     "tool_name": "get_delivery_date",
        #     "content": "2024-12-12"
        # }
    ]
    
    model = OpenAIHandler(
        model_name="claude-3-5-sonnet-20241022",
        temperature=0.0,
        top_p=0.01,
        tool_calling=True,
        num_gpus=4,
        gpu_memory_utilization=0.5,
    )
    
    test_entry = {
        "messages": messages,
        "tools": tools,
        "n": 1,
    }

    print(model.inference(test_entry, include_debugging_log=True, 
                          mode="chat", tool_call_mode="fc"))

    model.kill_process()

    
