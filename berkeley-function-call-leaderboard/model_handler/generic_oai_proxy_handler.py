from model_handler.handler import BaseHandler
from model_handler.model_style import ModelStyle
from model_handler.utils import (
    convert_to_tool,
    convert_to_function_call,
    augment_prompt_by_languge,
    language_specific_pre_processing,
    ast_parse,
)
from model_handler.constant import (
    GORILLA_TO_OPENAPI,
    GORILLA_TO_PYTHON,
    USER_PROMPT_FOR_CHAT_MODEL,
    SYSTEM_PROMPT_FOR_CHAT_MODEL,
)
from openai import OpenAI
import os, time, json


class GenericOAIProxyHandler(BaseHandler):
    def __init__(self, model_name, temperature=0.7, top_p=1, max_tokens=1000) -> None:
        super().__init__(model_name, temperature, top_p, max_tokens)
        self.model_style = ModelStyle.OpenAI
        # TODO: to avoid changing BFCL too much, just passing this as an environment variable
        if not os.environ.get("MODEL_ENDPOINT_URL"):
            raise ValueError("MODEL_ENDPOINT_URL is not set. Please set it to use the GenericOAIProxyHandler.")
        self.client = OpenAI(
            api_key='-',
            base_url=os.environ["MODEL_ENDPOINT_URL"],
        )

    def inference(self, prompt,functions,test_category):
        API_FAILURE_MESSAGE = None  # this is a flag to indicate that the API failed
        if "FC" not in self.model_name:
            prompt = augment_prompt_by_languge(prompt,test_category)
            functions = language_specific_pre_processing(functions,test_category,False)
            message = [
                {
                    "role": "user",
                    "content": f"<query>{prompt}</query>\n<functions>{str(functions)}</functions>",
                },
            ]
            start_time = time.time()
            response = self.client.chat.completions.create(
                messages=message,
                model=os.environ.get("ENDPOINT_MODEL_NAME", 'auto'),  # if model=auto, then the proxy server 
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
            )
            latency = time.time() - start_time
            result = response.choices[0].message.content
        else:
            prompt = augment_prompt_by_languge(prompt, test_category)
            functions = language_specific_pre_processing(functions, test_category, True)
            if type(functions) is not list:
                functions = [functions]
            message = [{"role": "user", "content": f"<query>{prompt}</query>"}]
            oai_tool = convert_to_tool(
                functions, GORILLA_TO_OPENAPI, self.model_style, test_category, True
            )
            start_time = time.time()
            if len(oai_tool) > 0:
                # bugfix: the cards tool is an "object" without "properties", which breaks outlines
                """
                # looks like this happens in more cases. can I make my solution more generic?
                Functions=[{'name': 'calculate_standard_deviation', 'description': 'This function calculates the standard deviation across different scores for a specific student.', 'parameters': {'type': 'object', 'properties': {'gradeDict': {'type': 'object', 'description': 'A dictionary where keys represent subjects and values represent scores'}}, 'required': ['gradeDict']}}, {'name': 'calculate_average', 'description': 'This function calculates the average grade across different subjects for a specific student.', 'parameters': {'type': 'object', 'properties': {'gradeDict': {'type': 'object', 'description': 'A dictionary where keys represent subjects and values represent scores'}}, 'required': ['gradeDict']}}, {'name': 'highest_grade', 'description': 'This function finds the subject where the student got the highest score.', 'parameters': {'type': 'object', 'properties': {'gradeDict': {'type': 'object', 'description': 'A dictionary where keys represent subjects and values represent scores'}}, 'required': ['gradeDict']}}]
                """
                try:
                    for tool in oai_tool:
                        if 'cards' in tool['function']['parameters']['properties'].keys():
                            tool['function']['parameters']['properties']['cards'] =\
                                {'type': 'object',
                                 'description': 'An object containing the player name as key and the cards as values in a list.',
                                 'properties': {'player_name': {'type': 'string', 'description': 'The name of the player.'},
                                 'cards': {'type': 'array',
                                 'items': {'type': 'string',},
                                 'description': 'List of cards that the player has.'}},
                                }
                        if 'gradeDict' in tool['function']['parameters']['properties'].keys():
                            tool['function']['parameters']['properties']['gradeDict'] =\
                                {'type': 'string',
                                 'description': 'A dictionary where keys represent subjects and values represent scores',
                                 }
                        if 'population' in tool['function']['parameters']['properties'].keys():
                            tool['function']['parameters']['properties']['population'] =\
                                {'type': 'object',
                                 'description': "The description of population. 'adults' is the number of adults in the household. 'children' is the number of children. 'singles' is the number of single adults living alone.",
                                 'required': ['adults', 'children', 'singles'],
                                    'properties': {'adults': {'type': 'integer', 'description': 'The number of adults in the household.'},
                                                   'children': {'type': 'integer', 'description': 'The number of children in the household.'},
                                                   'singles': {'type': 'integer', 'description': 'The number of single adults living alone.'},
                                                  }
                                 }
                except Exception as e:
                    pass
                try:
                    response = self.client.chat.completions.create(
                        messages=message,
                        model=os.environ.get("ENDPOINT_MODEL_NAME", 'auto-FC').replace("-FC", ""),
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        top_p=self.top_p,
                        tools=oai_tool,
                        tool_choice='auto', # this is important as it let's the model decide when to use FC
                    )
                except Exception as e:
                    print(f"\nError while trying to do FC: {e}\n")
                    print(f"Messages={message}")
                    print(f"Functions={functions}\n")
                    API_FAILURE_MESSAGE = f"API Failure: {e}"
            else:
                # @KS: TODO: Gorilla decided not to use the tool? What's going on here.
                print(f"DEBUG: BFCL decided to not use the tool.")
                print(f"Prompt = {prompt}")
                print(f"Functions = {functions}")
                response = self.client.chat.completions.create(
                    messages=message,
                    model=os.environ.get("ENDPOINT_MODEL_NAME", 'auto-FC').replace("-FC", ""),
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                )
            latency = time.time() - start_time
            try:
                result = [
                    {func_call.function.name: func_call.function.arguments}
                    for func_call in response.choices[0].message.tool_calls
                ]
            except Exception as e:
                print("Error while trying to extract function calls from response:", e)
                if API_FAILURE_MESSAGE:
                    result = API_FAILURE_MESSAGE
                else:
                    result = response.choices[0].message.content
        metadata = {}
        if API_FAILURE_MESSAGE:
            # do something
            metadata["input_tokens"] = -1
            metadata["output_tokens"] = -1
        else:
            metadata["input_tokens"] = response.usage.prompt_tokens
            metadata["output_tokens"] = response.usage.completion_tokens
        metadata["latency"] = latency
        return result,metadata
    
    def decode_ast(self,result,language="Python"):
        if "FC" not in self.model_name:
            decoded_output = ast_parse(result,language)
        else:
            decoded_output = []
            for invoked_function in result:
                name = list(invoked_function.keys())[0]
                if name == "weather_get_by_coordinates_date":
                    date_field = invoked_function[name].split('"date": ')[1].replace("}", "")
                    invoked_function[name].replace(date_field, f'"{date_field}"')
                params = json.loads(invoked_function[name])
                if language == "Python":
                    pass
                else:
                    # all values of the json are casted to string for java and javascript
                    for key in params:
                        params[key] = str(params[key])
                decoded_output.append({name: params})
        return decoded_output
    
    def decode_execute(self,result):
        if "FC" not in self.model_name:
            decoded_output = ast_parse(result)
            execution_list = []
            for function_call in decoded_output:
                for key, value in function_call.items():
                    execution_list.append(
                        f"{key}({','.join([f'{k}={repr(v)}' for k, v in value.items()])})"
                    )
            return execution_list
        else:
            function_call = convert_to_function_call(result)
            return function_call
