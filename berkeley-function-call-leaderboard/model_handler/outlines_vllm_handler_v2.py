import json
import os
import re
import time
import warnings
from functools import reduce
from typing import Union

import torch
from model_handler.constant import style_to_system_prompt, style_to_user_prompt
from model_handler.handler import BaseHandler
from model_handler.model_style import ModelStyle
from model_handler.utils import ast_parse #, _cast_to_openai_type
from outlines.fsm.json_schema import build_regex_from_schema, get_schema_from_signature
from tool_use.prompt import get_system_prompt, get_system_prompt_art #, get_tool_selector_system_prompt
from tool_use.tool import Tool
from tool_use.utils.regex_util import tools_to_schema


class OutlinesVllmHandler(BaseHandler):

    def __init__(
        self,
        model_name,
        temperature=0,
        top_p=1,
        max_tokens=4096,
        seed=42,
        gen_mode="conditional",
        n_tool_calls=1,
        user_prompt_style="json",
        system_prompt_style=None,
        ) -> None:

        self.model_style = ModelStyle.Outlines
        self._n_tool_calls = n_tool_calls
        self.n_tool_calls = None
        self.gen_mode = gen_mode
        self.user_prompt_style = user_prompt_style
        self.system_prompt_style = system_prompt_style

        self.gen_kwargs = dict(temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        super().__init__(model_name, temperature, top_p, max_tokens)

        # Initialize tool
        self.base_url = "http://localhost:8000/v1"
        self.api_key = "-"
        self.tool = Tool(self.base_url, self.api_key, self.model_name, gen_kwargs=self.gen_kwargs)

    def load_solution(self, user_query, test_category):
        test_category = test_category.replace("executable_", "")
        solutions_path = f"./data/possible_answer/gorilla_openfunctions_v1_test_{test_category}.json"
        questions_path = f"./data/gorilla_openfunctions_v1_test_{test_category}.json"

        # Find the index where the question matches the user query
        question_idx = 0
        with open(questions_path, "r") as f:
            for line in f:
                question = json.loads(line)["question"]
                if question == user_query:
                    break
                question_idx += 1

        # Get solution located at question_idx
        solution_idx = 0
        solution = {}
        with open(solutions_path, "r") as f:
            for line in f:
                if solution_idx == question_idx:
                    solution = json.loads(line)
                    break
                solution_idx += 1

        return solution

    def get_prompt(self, tools, user_query):

        system_prompt = ""
        if self.system_prompt_style == 'art':
            system_prompt = get_system_prompt_art(tools)
        elif self.system_prompt_style is not None:
            tools_schema = tools_to_schema(tools)
            system_prompt = style_to_system_prompt[self.system_prompt_style].format(tools_schema=tools_schema)

        user_prompt = user_query
        if self.user_prompt_style is not None:
            user_prompt = style_to_user_prompt[self.user_prompt_style].format(functions=str(tools), user_prompt=user_query)

        return system_prompt, user_prompt

    def inference(self, user_query, tools, test_category):
        # get n_tool_calls
        if self._n_tool_calls == "solution" and test_category == "relevance":
            raise ValueError("Solutions is not valid for relevance category.")
        elif self._n_tool_calls == "solution":
            self.solution = self.load_solution(user_query, test_category)
            self.n_tool_calls = len(self.solution)
        else:
            self.n_tool_calls = self._n_tool_calls
            
        # This seems to help:
        # (before sys: 81, 79.5)
        # (after sys: 79)
        if self.gen_mode == "neo_conditional":
            import copy
            tools = copy.deepcopy(tools)
            if not isinstance(tools, list):
                tools = [tools]
            for tool in tools:
                for param in tool['parameters']['properties'].values():
                    if param['type'] == 'tuple':
                        param['type'] = 'array'
                    if param['type'] == 'any':
                        param['type'] = 'string'
                    if param['type'] == 'dict':
                        param['type'] = 'string'


        # Get messages
        try:
            system_prompt, user_prompt = self.get_prompt(tools, user_query)
        except Exception as e:
            result = f'[error.message(error="{e}")]'
            print(f"ERROR:\n{e}")
            return result, {"input_tokens": 0, "output_tokens": 0, "latency": 0, "n_tool_calls": self.n_tool_calls, "tool_calls": [], "messages": ""}
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        
        # Generate tool calls
        tool_calls, output_messages = [], None
        try:
            start = time.time()
            output_messages, tool_calls = self.tool(messages, gen_mode=self.gen_mode, tools=tools, n_tool_calls=self.n_tool_calls)
            if self.user_prompt_style == "json" or self.system_prompt_style == "json" or self.gen_mode == 'neo_conditional' or self.system_prompt_style == 'art':
                result = bfcl_format(tool_calls)
            else: # python
                result = output_messages[-1]["content"]
        except Exception as e:
            result = f'[error.message(error="{e}")]'
            print(f"ERROR:\n{e}")
            return result, {"input_tokens": 0, "output_tokens": 0, "latency": 0,  "n_tool_calls": self.n_tool_calls, "tool_calls": tool_calls, "messages": messages if output_messages is None else output_messages}

        # Record info
        latency = time.time() - start
        metadata = {
            "input_tokens": 0,
            "output_tokens": 0,
            "latency": latency,
            "n_tool_calls": self.n_tool_calls,
            "tool_calls": tool_calls,
            "messages": output_messages,
            }
        return result, metadata

    def decode_ast(self, result, language="Python"):
        decoded_output = ast_parse(result, language)
        return decoded_output

    def decode_execute(self, result):

        # Parse result
        func = result.replace('", "', ",").replace('["', "[").replace('"]', "]")
        if " " == func[0]:
            func = func[1:]
        if not func.startswith("["):
            func = "[" + func
        if not func.endswith("]"):
            func = func + "]"

        # Parse AST of func
        decode_output = ast_parse(func)

        # Create execution list
        execution_list = []
        for function_call in decode_output:
            for key, value in function_call.items():
                execution_list.append(
                    f"{key}({','.join([f'{k}={repr(v)}' for k, v in value.items()])})"
                )
        return execution_list

    def write(self, result, write_path):
        os.makedirs(os.path.dirname(write_path), exist_ok=True)

        # Write path
        with open(write_path, "a+") as f:
            f.write(json.dumps(result) + "\n")

    def load_result(self, test_category):
        # This method is used to load the result from the file.
        result_list = []
        with open(
            f"./result/{self.model_name.replace('/', '_')}/gorilla_openfunctions_v1_test_{test_category}_result.json"
        ) as f:
            for line in f:
                result_list.append(json.loads(line))
        return result_list

def bfcl_format(tool_calls):
    tool_strs = []
    for tool_call in tool_calls:
        args, name = tool_call["tool_arguments"], tool_call["tool_name"]
        args_string = ', '.join([f"{key}='{value}'" if isinstance(value, str) else f"{key}={value}" for key, value in args.items()])
        tool_str = f'{name}({args_string})'
        tool_strs.append(tool_str)
    result = '[' + ', '.join(tool_strs) + ']'
    return result