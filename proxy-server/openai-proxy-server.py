import logging
import json
import os
import requests
import time

from aiohttp import web

import openai
from openai import AsyncOpenAI
from dotenv import load_dotenv

class OpenAIProxyServer:

    def __init__(self, base_url=None, api_key=None, client=None,
                 required_request_parameters=['messages', 'model']
                 ):
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

    def create_app(self):
        app = web.Application()
        app.on_startup.append(self._on_startup)
        self.setup_routes(app)
        return app
    
    def setup_routes(self, app):
        # app.router.add_post('/', self._handle_request)
        app.router.add_post('/v1/chat/completions', self._handle_request)
        app.router.add_get('/', self._health_check)
    
    async def _on_startup(self, app):
        self.logger.info("Server has started successfully")

    async def _health_check(self, request: web.Request):
        return web.json_response({"status": "healthy"})
    
    async def _handle_request(self, request: web.Request):
        self.logger.info(f"Received request from {request.remote}")
        request_data = await self.extract_request_data(request)
        if isinstance(request_data, web.Response):
            # and error has occurred and we should return early
            return request_data
        return await self.process_request(request_data)
    
    async def extract_request_data(self, request: web.Request):
        request_data = await request.json()
        self.validate_request(request_data)
        request_data['tools'] = request_data.get('tools', [])
        request_data['tool_choice'] = request_data.get('tool_choice', 'auto')
        request_data['stop'] = request_data.get('stop', ["</tool_call>"])

        # try:
        #     request_data['messages'] = create_or_modify_system_prompt(request_data['messages'], request_data['tools'], request_data['tool_choice'])
        # except Exception as e:
        #     return self._request_exception_handler(request_data=request_data, error=e)

        return request_data
    
    async def process_request(self, request_data: dict):
        try:
            response = await self.client.chat.completions.create(
                **request_data
            )
        except Exception as e:
            return self._request_exception_handler(request_data=request_data, error=e)
        
        # Simulate the 'usage' data (this should ideally come from real usage metrics)
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

        message = {
            "role": "assistant",
            "content": response.choices[0].message.content,
            "tool_calls": None
        }

        # Construct a response matching the OpenAI API structure
        formatted_response = {
            "id": response.id,
            "object": "chat.completion",
            "created": response.created,
            "model": request_data['model'],
            "choices": [{
                "index":0,
                "message": message,
                "logprobs": None,
                "finish_reason": response.choices[0].finish_reason
            }],
            "usage": usage
        }
        return web.json_response(formatted_response)
    
    def _request_exception_handler(self, request_data: dict, error: Exception):
        if isinstance(error, openai.BadRequestError):
            print(error)
            bad_response_json, status = self._build_bad_response(request_data, error)
            return web.json_response(bad_response_json, status=status)
        else:
            return web.json_response(self._build_general_error_response(error), status=400)

    def _build_general_error_response(self, error: Exception):
        return {
            "error": {
                  # extract the error message
                "message": str(error),
                "type": type(error).__name__,
                "param": None,
                "code": 400
            }
        }
    
    def _build_bad_response(self, request_data, error):

        assert isinstance(error, openai.BadRequestError), f"Error must be a BadRequestError. Got {type(error)} instead."
        # okay, we have a bad request error, let's build a response
        
        # TODO: this lol
        print('warning: a bad request has been made, but the handler is not implemented yet so generic response inbound.')
        return ({
            "id": 0,
            "object": "chat.completion",
            "created": 0,
            "model": "XXX",
            "choices": [{
                "index":0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": []
                },
                "logprobs": None,
                "finish_reason": "ERROR"
            }],
            "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 0,
            "total_tokens": 100,
            }
        },
        400
        )

    def validate_request(self, request_data):
        for key in self.required_request_parameters:
            if key not in request_data:
                raise ValueError(f"Request must contain a '{key}' key.")


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
    api_key = os.environ["DATABRICKS_TOKEN"]
    # api_key = os.getenv("OPENAI_API_KEY")
    
    proxy_server = OpenAIProxyServer(base_url=base_url, api_key=api_key)
    app = proxy_server.create_app()
    web.run_app(app, port=8080)