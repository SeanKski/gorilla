import logging
import json
import os
import requests
import time
from typing import Optional
from abc import ABC, abstractmethod

from aiohttp import web

import openai
from openai.types.chat import ChatCompletion
from openai import AsyncOpenAI
from dotenv import load_dotenv

class BaseOpenAIProxyServer(ABC):
    """
    A proxy server that can be used to wrap around an OpenAI chat API client. This is helpful for quickly writing a 
    custom inference server (e.g., for tool-use).
    """    
    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None,
                 client: Optional[openai.AsyncOpenAI] = None,
                 required_request_parameters: Optional[list] = ['messages', 'model']):
        """
        Initializes the ProxyServer object.
        Parameters:
        - base_url: The base URL of the server. Default is None, if this is not provided then client must be provided.
        - api_key: The API key for authentication. Default is None, if this is not provided then client must be provided.
        - client: An instance of the AsyncOpenAI client. Default is None, if this is not provided then base_url and api_key must be provided.
        - required_request_parameters (list): A list of keys which are required for a request to be considered valid.
            The default is ['messages', 'model'].
        """ 
        # either client or base_url and api_key should be provided
        if client is None:
            if base_url is None or api_key is None:
                raise ValueError("Either client or base_url and api_key should be provided.")
            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url
            )
        else:
            self.client = client
        self.required_request_parameters = required_request_parameters

        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    async def _handle_request(self, request: web.Request):
        """
        The main handler for incoming requests. This method extracts the request data, processes it, and returns the response.
        Ideally this shouldn't be overridden, and instead you should override the components of this pipeline. 
        The components are:
          1. `preprocess_request_data`: This takes in the raw request dictionary and allows for custom preprocessing
          2.  `execute_request`: This takes in the preprocessed request dictionary and does the actual inference execution
          3. `post_process_request_execution`: This takes in the executed response and allows for custom post-processing
          4. `format_chat_completion_response`: This takes in the post-processed dictionary and formats to the form expected
                by the OpenAI chat completion API. This probably doesn't need to be overridden, but put here just in case.
        """
        self.logger.info(f"Received request from {request.remote}")
        try:
            request_data = await request.json()
            self.validate_request(request_data)            
            request_data = await self.preprocess_request_data(request_data)
            executed_response_data = await self.execute_request(request_data)
            executed_response_data = await self.post_process_request_execution(executed_response_data)
            formatted_json_response = self.format_chat_completion_response(executed_response_data)
            return web.json_response(formatted_json_response)
        except Exception as e:
            # at some point we might want to pass the relevant data_dict here for better error handling
            # but for now, we'll just pass None
            return self._request_exception_handler(error=e, data_dict=None)

    async def preprocess_request_data(self, request_data: dict) -> dict:
        """
        A method for preprocessing the request data before sending it to the `execute_request` method.
        This can be subclassed to modify the input request data (e.g., adding tools, modifying system messages, etc.).
        """
        return request_data
    
    async def execute_request(self, request_data: dict) -> ChatCompletion | dict:
        """
        A method which takes in the pre-processed request dictionary and does the actual inference execution.
        This can be subclassed to modify the actual execution (e.g., implementing conditional generation, etc.).
        """
        self.logger.info(f"Sending request to OpenAI with data: {json.dumps(request_data, indent=2)}")
        response = await self.client.chat.completions.create(**request_data)
        return response
    
    async def post_process_request_execution(self, response: dict | ChatCompletion) -> dict | ChatCompletion:
        """
        A method for post-processing the response from the OpenAI client before returning it to the user.
        This can be subclassed to modify the output response (e.g., adding metadata, unifying messages, etc.).
        """
        return response

    def format_chat_completion_response(self, response: dict | ChatCompletion) -> dict:
        """
        A method for formatting the response from the OpenAI client into the form expected by the OpenAI chat completion API.
        NOTE: By default this only works for OpenAI ChatCompletion objects. If you are passing in a dictionary,
              you will need to override this method.
        """
        if isinstance(response, ChatCompletion):
            # Response is a subclass of the Pydantic BaseModel, so all we need to do is call model_dump() to convert it to JSON
           return response.model_dump()
        
            # # Leaving this old code here for reference in case someone wants to implement a custom formatting
            # # extract any tools calls from the response
            # if hasattr(response.choices[0].message, 'tool_calls'):
            #     tool_calls = [tool_call.model_dump() for tool_call in response.choices[0].message.tool_calls]
            # formatted_response = {
            #     "id": response.id,
            #     "object": "chat.completion",
            #     "created": response.created,
            #     "model": response.model,
            #     "choices": [{
            #         "index": 0,
            #         "message": {
            #             "role": "assistant",
            #             "content": response.choices[0].message.content,
            #             "tool_calls": tool_calls
            #         },
            #         "finish_reason": response.choices[0].finish_reason
            #     }],
            #     "usage": {
            #         "prompt_tokens": response.usage.prompt_tokens,
            #         "completion_tokens": response.usage.completion_tokens,
            #         "total_tokens": response.usage.total_tokens,
            #     }
            # }
        else:
            raise ValueError('The default `format_chat_completion_response` method only works for OpenAI ChatCompletion objects.'
                       'If you are returning a dictionary, you will need to override this method.')
        return formatted_response

    def create_app(self):
        app = web.Application()
        app.on_startup.append(self._on_startup)
        self._setup_routes(app)
        return app
    
    def _setup_routes(self, app):
        # app.router.add_post('/', self._handle_request)
        app.router.add_post('/chat/completions', self._handle_request)
        app.router.add_get('/', self._health_check)
    
    async def _on_startup(self, app):
        self.logger.info("Server has started successfully")

    async def _health_check(self, request: web.Request):
        return web.json_response({"status": "healthy"})
    
    def _request_exception_handler(self, error: Exception, data_dict=None):
        self.logger.error(f"Error encountered in request: {str(error)}", exc_info=True)
        if isinstance(error, openai.BadRequestError):
            # If the error is a BadRequestError, we want to return a specific response
            return self._output_bad_request_error(error, data_dict)
        elif isinstance(error, openai.OpenAIError):
            # For OpenAI errors (that aren't BadRequestErrors, since that could be from our end), return the error as-is
            return self._output_openai_error(error, data_dict)
        else:
            return self._output_general_error(error, data_dict)

    def _output_openai_error(self, error: openai.OpenAIError, data_dict: dict = None):
            return web.json_response(
                {
                    "error": {
                        "message": str(error),
                        "type": type(error).__name__,
                        "code": error.status_code if hasattr(error, 'status_code') else 500
                    }
                },
                status=error.status_code if hasattr(error, 'status_code') else 500
            )

    def _output_general_error(self, error: Exception, data_dict: dict = None):
        return web.json_response(
            {
                "error": {
                    "message": str(error),
                    "type": 'Internal Server Error',
                    "param": None,
                    "code": 500
                }
            },
            status=500
        )
    
    def _output_bad_request_error(self, error: Exception, data_dict: dict = None):
        assert isinstance(error, openai.BadRequestError), f"Error must be a BadRequestError. Got {type(error)} instead."
        # # okay, we have a bad request error, let's build a response
        # for now we'll just return the error as is, but this can be customized (e.g., adding more info)
        return self._output_openai_error(error, data_dict)

    def validate_request(self, request_data):
        """
        Validates the incoming request data to ensure it contains the required parameters.
        """
        for key in self.required_request_parameters:
            if key not in request_data:
                logging.error(f"Incoming request data failed to validate due to missing a required parameter. It must contain a '{key}' key.")
                raise ValueError(f"Incoming request data is missing the required '{key}' key.")


def check_if_server_is_running(url='http://localhost:8080', retries=5, delay=2):
    for i in range(retries):
        try:
            response = requests.get(url)
            response.raise_for_status()
            print("Server is up and responding!")
            print("Response:", response.json())
            return True
        except requests.exceptions.RequestException as e:
            print(f"Attempt {i+1}: Failed to connect to the server: {e}")
            if i < retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
    
    print("Failed to connect to the server after multiple attempts.")
    return False

if __name__ == '__main__':

    load_dotenv()

    base_url = os.getenv("DATABRICKS_URL")
    if base_url is None:
        print("DATABRICKS_URL not set, using default (DF 1 workspace url)")
        base_url = "https://dbc-559ffd80-2bfc.cloud.databricks.com/serving-endpoints/"
    api_key = os.environ["DATABRICKS_TOKEN"]
    
    proxy_server = BaseOpenAIProxyServer(base_url=base_url, api_key=api_key)
    app = proxy_server.create_app()
    web.run_app(app, port=8080)