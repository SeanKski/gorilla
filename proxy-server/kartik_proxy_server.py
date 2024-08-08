import logging
import json
import os
import re
from typing import Coroutine
from openai.types.chat import ChatCompletion
import requests
import time

from aiohttp import web

import openai
from openai import AsyncOpenAI
from dotenv import load_dotenv

from openai_proxy_server import BaseOpenAIProxyServer

def quote_unquoted_variables(text):
    # if the variable is None then just turn it into null
    if text == "None":
        return "null"
    # Match variable names not surrounded by quotes
    pattern = r'(?<!["\'])\b[a-zA-Z_][a-zA-Z0-9_]*\b(?!\["\'\])'
    # Replace the found variable names with them surrounded by double quotes
    return re.sub(pattern, r'"\g<0>"', text)


def remove_comprehension(text):
    pattern = r'\sfor\s+\w+\s+in\s+\w+(?:\s+if\s+.*)?'
    return re.sub(pattern, '', text)



def default_tools_system_prompt(tools_schema: str):
    tool_list_start = "<tools>"
    tool_list_end = "</tools>"

    tool_call_start = "<tool_call>"
    tool_call_end = "</tool_call>"

    return f"""Your job is to answer the user's questions, which may involve calling functions from a list of available ones. They are provided in JSON Schema format. Your task is to call one or more of these functions to help the user, when they are relevant, and answer directly when the provided functions are are not pertinent.

Please use your own judgment as to whether or not you should call a function. In particular, you must follow these guiding principles:
    1. You may assume the user has implemented the function themselves.
    2. You may assume the user will call the function on their own. You should NOT ask the user to call the function and let you know the result; they will do this on their own. You just need to pass the name and arguments.
    3. Never call a function twice with the same arguments. Do not repeat your function calls!
    4. If none of the functions are truly relevant to the user's question, do not make any unnecessary function calls.


You can only call functions according to the following formatting rules:
    Rule 1: All the functions you have access to are contained within {tool_list_start}{tool_list_end} XML tags. You cannot use any functions that are not listed between these tags.
    Rule 2: For each function call, output JSON which conforms to the schema of the function. You must wrap the function call in {tool_call_start}[...list of tool calls...]{tool_call_end} XML tags. Each call will be a JSON object with the keys "name" and "arguments". The "name" key will contain the name of the function you are calling, and the "arguments" key will contain the arguments you are passing to the function as a JSON object. The top level structure is a list of these objects. YOU MUST OUTPUT VALID JSON BETWEEN THE {tool_call_start} AND {tool_call_end} TAGS!
    Rule 3: If user decides to run the function, they will output the result of the function call in the following query. If it answers the user's question, you should incorporate the output of the function in your following message.

P.S: When specifying boolean arguments, use the strings true and false (without quotes) to represent True and False, respectively.

Here are the functions available to you:
{tool_list_start}\n{tools_schema}\n{tool_list_end}

The format of your response must be as follows:
<thinking>Your analysis of the situation and what function call(s) and arguments you will use.</thinking>""" + \
    """<tool_call>[{"name": "function_name", "arguments": {"arg1": "value1", "arg2": "value2"}}, ...]</tool_call>

NO OTHER CONTENT IS REQUIRED."""


def relevance_focussed_tools_system_prompt(tools_schema: str):
    tool_list_start = "<tools>"
    tool_list_end = "</tools>"

    tool_call_start = "<tool_call>"
    tool_call_end = "</tool_call>"

    system_prompt = f"""You are a helpful assistant.

Your job is to answer the user's questions, which may involve calling functions from a list of available ones. They are provided in JSON Schema format. Your task is to call one or more of these functions to help the user, when they are relevant, and answer directly when the provided functions are are not pertinent.

Please use your own judgment as to whether or not you should call a function. In particular, you must follow these guiding principles:
    1. You may assume the user has implemented the function themselves.
    2. You may assume the user will call the function on their own. You should NOT ask the user to call the function and let you know the result; they will do this on their own. You just need to pass the name and arguments.
    3. Never call a function twice with the same arguments. Do not repeat your function calls!
    4. If none of the functions are relevant to the user's question, DO NOT MAKE any unnecessary function calls.
    5. Do not assume access to any functions that are not listed in this prompt, no matter how simple. Do not assume access to a code interpretor either. DO NOT MAKE UP FUNCTIONS.


You can only call functions according to the following formatting rules:
    Rule 1: All the functions you have access to are contained within {tool_list_start}{tool_list_end} XML tags. You cannot use any functions that are not listed between these tags.
    Rule 2: For each function call, output JSON which conforms to the schema of the function. You must wrap the function call in {tool_call_start}[...list of tool calls...]{tool_call_end} XML tags. Each call will be a JSON object with the keys "name" and "arguments". The "name" key will contain the name of the function you are calling, and the "arguments" key will contain the arguments you are passing to the function as a JSON object. The top level structure is a list of these objects. YOU MUST OUTPUT VALID JSON BETWEEN THE {tool_call_start} AND {tool_call_end} TAGS!
    Rule 3: If user decides to run the function, they will output the result of the function call in the following query. If it answers the user's question, you should incorporate the output of the function in your following message.

P.S: When specifying boolean arguments, use the strings true and false (without quotes) to represent True and False, respectively.

The format of your response must be as follows:
<thinking>Your analysis of the situation and what function call(s) and arguments you will use.</thinking>""" + \
    """<tool_call>[{"name": "function_name", "arguments": {"arg1": "value1", "arg2": "value2"}}, ...]</tool_call>

NO OTHER CONTENT IS REQUIRED.
"""

    # add few-shot examples to handle relevance showing that the model doesn't need to always call functions.
    system_prompt += """Remember that if none of the functions given to you in this prompt are relevant, SKIP the tool_call section. DO NOT MAKE UP FUNCTIONS TO USE IN THE <tool_call> section.
Here are 5 examples of how to handle cases when none of the functions are relevant:

1. Supposed the functions available to you are:
<tools>
[{'type': 'function', 'function': {'name': 'determine_body_mass_index', 'description': 'Calculate body mass index given weight and height.', 'parameters': {'type': 'object', 'properties': {'weight': {'type': 'number', 'description': 'Weight of the individual in kilograms. This is a float type value.', 'format': 'float'}, 'height': {'type': 'number', 'description': 'Height of the individual in meters. This is a float type value.', 'format': 'float'}}, 'required': ['weight', 'height']}}}]
[{'type': 'function', 'function': {'name': 'math_prod', 'description': 'Compute the product of all numbers in a list.', 'parameters': {'type': 'object', 'properties': {'numbers': {'type': 'array', 'items': {'type': 'number'}, 'description': 'The list of numbers to be added up.'}, 'decimal_places': {'type': 'integer', 'description': 'The number of decimal places to round to. Default is 2.'}}, 'required': ['numbers']}}}]
[{'type': 'function', 'function': {'name': 'distance_calculator_calculate', 'description': 'Calculate the distance between two geographical coordinates.', 'parameters': {'type': 'object', 'properties': {'coordinate_1': {'type': 'array', 'items': {'type': 'number'}, 'description': 'The first coordinate, a pair of latitude and longitude.'}, 'coordinate_2': {'type': 'array', 'items': {'type': 'number'}, 'description': 'The second coordinate, a pair of latitude and longitude.'}}, 'required': ['coordinate_1', 'coordinate_2']}}}]
</tools>

And the user asks:
Question: What is the current time in New York?

Then you should respond with:
<thinking>
Let's start with a list of functions I have access to:
- determine_body_mass_index: since this function is not relevant to getting the current time, I will not call it.
- math_prod: since this function is not relevant to getting the current time, I will not call it.
- distance_calculator_calculate: since this function is not relevant to getting the current time, I will not call it.
None of the available functions, [determine_body_mass_index, math_prod, distance_calculator] are pertinent to the given query. Please check if you left out any relevant functions.
As a Large Language Model, without access to the appropriate tools, I am unable to provide the current time in New York.
</thinking>

2. Supposed the functions available to you are:
<tools>
[{'type': 'function', 'function': {'name': 'calculate_resistance_using_ohms_law', 'description': 'Calculate the resistance given potential difference (V) and current (I).', 'parameters': {'type': 'object', 'properties': {'voltage': {'type': 'number', 'description': 'Potential difference in volts. This is a float type value.', 'format': 'float'}, 'current': {'type': 'number', 'description': 'Current in amperes. This is a float type value.', 'format': 'float'}}, 'required': ['voltage', 'current']}}}]
</tools>

And the user asks:
Question: What is speed of light?

Then you should respond with:
<thinking>
Let's start with a list of functions I have access to:
- calculate_resistance_using_ohms_law: since this function is not relevant to getting the speed of light, I will not call it.
None of the available functions, [calculate_resistance_using_ohms_law] are pertinent to the given query. Please check if you left out any relevant functions.
However, I can answer the given question without using any tool calls. the speed of light is c=299792458 m/s
</thinking>

3. Supposed the functions available to you are:
<tools>
[{'type': 'function', 'function': {'name': 'find_roots', 'description': 'Find the roots of a quadratic equation ax^2 + bx + c = 0.', 'parameters': {'type': 'object', 'properties': {'a': {'type': 'number', 'description': 'Coefficient of x^2. This is a float type value.', 'format': 'float'}, 'b': {'type': 'number', 'description': 'Coefficient of x. This is a float type value.', 'format': 'float'}, 'c': {'type': 'number', 'description': 'Constant term. This is a float type value.', 'format': 'float'}}, 'required': ['a', 'b', 'c']}}}]
[{'type': 'function', 'function': {'name': 'draw_circle', 'description': 'Draw a circle based on the radius provided.', 'parameters': {'type': 'object', 'properties': {'radius': {'type': 'number', 'description': 'The radius of the circle. This is a float type value.', 'format': 'float'}, 'unit': {'type': 'string', 'description': "The unit of measurement for the radius. e.g. 'm' for meters, 'cm' for centimeters"}}, 'required': ['radius', 'unit']}}}]
[{'type': 'function', 'function': {'name': 'draw_square', 'description': 'Draw a square based on the side length provided.', 'parameters': {'type': 'object', 'properties': {'side': {'type': 'number', 'description': 'The side length of the circle. This is a float type value.', 'format': 'float'}, 'unit': {'type': 'string', 'description': "The unit of measurement for the side length. e.g. 'm' for meters, 'cm' for centimeters"}}, 'required': ['radius', 'unit']}}}]
</tools>

And the user asks:
Question: Draw a rectangle with length 4cm and breadth 2cm.

Then you should respond with:
<thinking>
Let's start with a list of functions I have access to:
- find_roots: since this function is not relevant to drawing a square, I will not call it.
- draw_circle: since a circle and a square are different shapes, I will not call this function.
- draw_square: while rectangles are similar to squares, the user asked for a rectangle, not a square. Since all rectangles are not squares, I cannot use this function. Therefore, I will not clal this function.
None of the available functions, [find_roots, draw_circle, draw_square] are pertinent to the given query. Please check if you left out any relevant functions.
</thinking>

4. Supposed the functions available to you are:
<tools>
[{'type': 'function', 'function': {'name': 'find_roots', 'description': 'Find the roots of a quadratic equation ax^2 + bx + c = 0.', 'parameters': {'type': 'object', 'properties': {'a': {'type': 'number', 'description': 'Coefficient of x^2. This is a float type value.', 'format': 'float'}, 'b': {'type': 'number', 'description': 'Coefficient of x. This is a float type value.', 'format': 'float'}, 'c': {'type': 'number', 'description': 'Constant term. This is a float type value.', 'format': 'float'}}, 'required': ['a', 'b', 'c']}}}]
[{'type': 'function', 'function': {'name': 'draw_circle', 'description': 'Draw a circle based on the radius provided.', 'parameters': {'type': 'object', 'properties': {'radius': {'type': 'number', 'description': 'The radius of the circle. This is a float type value.', 'format': 'float'}, 'unit': {'type': 'string', 'description': "The unit of measurement for the radius. e.g. 'm' for meters, 'cm' for centimeters"}}, 'required': ['radius', 'unit']}}}]
[{'type': 'function', 'function': {'name': 'draw_square', 'description': 'Draw a square based on the side length provided.', 'parameters': {'type': 'object', 'properties': {'side': {'type': 'number', 'description': 'The side length of the circle. This is a float type value.', 'format': 'float'}, 'unit': {'type': 'string', 'description': "The unit of measurement for the side length. e.g. 'm' for meters, 'cm' for centimeters"}}, 'required': ['radius', 'unit']}}}]
</tools>

And the user asks:
Question: Get the area of a square of side length 2cm.

Then you should respond with:
<thinking>
Let's start with a list of functions I have access to:
- find_roots: since this function is not relevant to getting the area of a square, I will not call it.
- draw_circle: since this function is not relevant to getting the area of a square, I will not call it.
- draw_square: while the user's question is related to squares, drawing it will not help in getting the area of a square. Therefore, I will not call this function.
None of the available functions, [find_roots, draw_circle, draw_square] are pertinent to the given query. Please check if you left out any relevant functions.
</thinking>

5. Supposed the functions available to you are:
<tools>
[{'type': 'function', 'function': {'name': 'find_roots', 'description': 'Find the roots of a quadratic equation ax^2 + bx + c = 0.', 'parameters': {'type': 'object', 'properties': {'a': {'type': 'number', 'description': 'Coefficient of x^2. This is a float type value.', 'format': 'float'}, 'b': {'type': 'number', 'description': 'Coefficient of x. This is a float type value.', 'format': 'float'}, 'c': {'type': 'number', 'description': 'Constant term. This is a float type value.', 'format': 'float'}}, 'required': ['a', 'b', 'c']}}}]
[{'type': 'function', 'function': {'name': 'math_prod', 'description': 'Compute the product of all numbers in a list.', 'parameters': {'type': 'object', 'properties': {'numbers': {'type': 'array', 'items': {'type': 'number'}, 'description': 'The list of numbers to be added up.'}, 'decimal_places': {'type': 'integer', 'description': 'The number of decimal places to round to. Default is 2.'}}, 'required': ['numbers']}}}]
</tools>

And the user asks:
Question: Who won the cricket world cup in the year 2007?

Then you should respond with:
<thinking>
Let's start with a list of functions I have access to:
- find_roots: since this function is not relevant to cricket, I will not call it.
- math_prod: since this function is not relevant to cricket, I will not call it.
None of the available functions, [find_roots, draw_circle, draw_square] are pertinent to the given query. Please check if you left out any relevant functions.
However, I can answer the given question without using any tool calls. The winner of the cricket world cup in 2007 was Australia.
</thinking>

"""

    # add the list of tools available to the model
    system_prompt += f"""Here are the functions available to you:\n{tool_list_start}\n{tools_schema}\n{tool_list_end}"""

    return system_prompt


def create_or_modify_system_prompt(messages, tools, tool_choice):
    # TODO support tool_choice!
    # system_prompt = default_tools_system_prompt(tools)
    system_prompt = relevance_focussed_tools_system_prompt(tools)
    if len(messages) == 0 or messages[0]['role'] != 'system':
        messages.insert(0, {
            "role": "system",
            "content": "You are a helpful assistant.\n\n" + system_prompt
        })
    else:
        messages[0]['content'] += "\n\nIn addition to the above, remember the following:\n" + system_prompt
    return messages


def get_content_calls(content):
    if content.count("<tool_call>") == 0:
        return content, None
    
    # sometimes <tool_call> gets used in thinking tags, so we need to be careful
    count = content.count("<tool_call>")
    if count == 1:
        x = content.split("<tool_call>")
        content = x[0].strip()
        tools = x[1].strip().split("</tool_call>")[0]
    elif count > 1:
        content = content.split("</thinking>")[0] + "</thinking>"
        tools = content.split("<tool_call>")[count].split("</tool_call>")[0].strip()
    
    if tools[0].strip() != "[":
        tools = f"[{tools}]"
    try:
        tools_list = _load_tool_calls(tools)
    except Exception as e:
        print(content, tools)
        print(e)
        tools_list = []
    tools = format_tools_list(tools_list)
    return content, tools


def _load_tool_calls(tool_calls: str) -> list:
    try:
        tools = json.loads(tool_calls)
    except json.decoder.JSONDecodeError as e:
        # print("JSON DECODE ERROR")
        # print(tool_calls)
        # print(type(tool_calls))
        # print("trying eval!")
        # remove any list comprehensions
        tool_calls = remove_comprehension(tool_calls)
        # quote any variables
        tool_calls = quote_unquoted_variables(tool_calls)
        print(tool_calls)
        tools = eval(tool_calls)
    except Exception as e:
        print(e)
        tools = []
    return tools

def format_tools_list(tools: list) -> list:
    new_tools = [{} for _ in range(len(tools))]
    for i, tool in enumerate(tools):
        new_tools[i]['id'] = f"call_{i}"
        new_tools[i]["type"] = "function"
        new_tools[i]["function"] = {}
        try:
            new_tools[i]["function"]["name"] = tool["name"]
            new_tools[i]["function"]["arguments"] = json.dumps(tool["arguments"])
        except Exception as e:
            # maybe it is double nested... TODO fix this
            return format_tools_list(tools[0])
    return new_tools


class KartikProxyServer(BaseOpenAIProxyServer):

    async def preprocess_request_data(self, request_data: dict) -> dict:
        request_data['messages'] = request_data.get('messages', [])
        request_data['model'] = request_data.get('model')
        request_data['tools'] = request_data.get('tools', [])
        request_data['tool_choice'] = request_data.get('tool_choice', 'auto')
        request_data['temperature'] = request_data.get('temperature', 0.0)
        request_data['max_tokens'] = request_data.get('max_tokens', 8192)
        if "llama" in request_data['model']:
            request_data['max_tokens'] = 3000

        request_data['messages'] = create_or_modify_system_prompt(
            request_data['messages'], request_data['tools'], request_data['tool_choice']
        )
        return request_data
    
    async def execute_request(self, request_data: dict) -> ChatCompletion | dict:
        self.logger.info('Initiating thinking request...')
        response = await self.client.chat.completions.create(
            model=request_data['model'],
            messages=request_data['messages'],
            temperature=request_data['temperature'],
            max_tokens=request_data['max_tokens'],
            stop=["</tool_call>"]
        )

        content_raw = response.choices[0].message.content + "</tool_call>"

        content, tool_calls = get_content_calls(content_raw)

        return {'raw_response': response, 'content': content, 'tool_calls': tool_calls}
    
    async def post_process_request_execution(self, response: dict) -> ChatCompletion:
        # update the raw response with the new content
        raw_response = response['raw_response']
        raw_response.choices[0].message.content = response['content'] if response['tool_calls'] is None else None
        raw_response.choices[0].message.tool_calls = response['tool_calls']
        return raw_response


if __name__ == '__main__':
    
    load_dotenv()
    base_url = os.getenv("DATABRICKS_URL")
    if base_url is None:
        print("DATABRICKS_URL not set, using default (DF 1 workspace url)")
        base_url = "N/A"
    api_key = os.environ["DATABRICKS_TOKEN"]
    
    proxy_server = KartikProxyServer(base_url=base_url, api_key=api_key)
    app = proxy_server.create_app()
    web.run_app(app, port=8080)
