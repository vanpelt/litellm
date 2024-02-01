import sys, os
import traceback
from dotenv import load_dotenv

load_dotenv()
import os, io

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest, asyncio
import litellm
from litellm import embedding, completion, completion_cost, Timeout, acompletion
from litellm import RateLimitError
import json
import os
import tempfile
from typing import Union, Dict, List

litellm.num_retries = 3
litellm.cache = None
user_message = "Write a short poem about the sky"
messages = [{"content": user_message, "role": "user"}]


def load_vertex_ai_credentials(personal_account=False):
    # Define the path to the vertex_key.json file
    print("loading vertex ai credentials")
    filepath = os.path.dirname(os.path.abspath(__file__))
    if personal_account == False:
        vertex_key_path = filepath + "/vertex_key.json"
    else:
        vertex_key_path = filepath + "/personal_account.json"

    # Read the existing content of the file or create an empty dictionary
    try:
        with open(vertex_key_path, "r") as file:
            # Read the file content
            print("Read vertexai file path")
            content = file.read()

            # If the file is empty or not valid JSON, create an empty dictionary
            if not content or not content.strip():
                service_account_key_data = {}
            else:
                # Attempt to load the existing JSON content
                file.seek(0)
                service_account_key_data = json.load(file)
    except FileNotFoundError:
        # If the file doesn't exist, create an empty dictionary
        service_account_key_data = {}

    # Update the service_account_key_data with environment variables
    if personal_account == False:
        private_key_id = os.environ.get("VERTEX_AI_PRIVATE_KEY_ID", "")
        private_key = os.environ.get("VERTEX_AI_PRIVATE_KEY", "")
    else:
        private_key_id = os.environ.get("VERTEX_AI_PRIVATE_KEY_ID_2", "")
        private_key = os.environ.get("VERTEX_AI_PRIVATE_KEY_2", "")
    private_key = private_key.replace("\\n", "\n")
    service_account_key_data["private_key_id"] = private_key_id
    service_account_key_data["private_key"] = private_key

    print(f"service_account_key_data: {service_account_key_data}")
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
        # Write the updated content to the temporary file
        json.dump(service_account_key_data, temp_file, indent=2)

    # Export the temporary file as GOOGLE_APPLICATION_CREDENTIALS
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(temp_file.name)


# @pytest.mark.asyncio
# async def test_model_garden_raw():
#     load_vertex_ai_credentials(personal_account=True)
#     from google.cloud import aiplatform
#     from google.protobuf import json_format
#     from google.protobuf.struct_pb2 import Value

#     def predict_custom_trained_model_sample(
#         project: str,
#         endpoint_id: str,
#         instances: Union[Dict, List[Dict]],
#         location: str = "us-east1",
#         api_endpoint: str = "us-east1-aiplatform.googleapis.com",
#     ):
#         """
#         `instances` can be either single instance of type dict or a list
#         of instances.
#         """
#         # The AI Platform services require regional API endpoints.
#         client_options = {"api_endpoint": api_endpoint}
#         # Initialize client that will be used to create and send requests.
#         # This client only needs to be created once, and can be reused for multiple requests.
#         client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
#         # The format of each instance should conform to the deployed model's prediction input schema.
#         instances = instances if isinstance(instances, list) else [instances]
#         instances = [
#             json_format.ParseDict(instance_dict, Value()) for instance_dict in instances
#         ]
#         parameters_dict = {}
#         parameters = json_format.ParseDict(parameters_dict, Value())
#         endpoint = client.endpoint_path(
#             project=project, location=location, endpoint=endpoint_id
#         )
#         response = client.predict(
#             endpoint=endpoint, instances=instances, parameters=parameters
#         )
#         print("response")
#         print(" deployed_model_id:", response.deployed_model_id)
#         # The predictions are a google.protobuf.Value representation of the model's predictions.
#         predictions = response.predictions
#         print(f" prediction: {predictions}")

#     predict_custom_trained_model_sample(
#         project="510528649030",
#         endpoint_id="177109333002158080",
#         location="us-east1",
#         instances=[
#             {
#                 "prompt": "My favourite condiment is",
#                 "n": 1,
#                 "max_tokens": 200,
#                 "temperature": 1.0,
#                 "top_p": 1.0,
#                 "top_k": 10,
#             }
#         ],
#     )


@pytest.mark.asyncio
async def test_model_garden_with_litellm():
    load_vertex_ai_credentials(personal_account=True)
    ## set ENV variables
    os.environ["VERTEXAI_PROJECT"] = "510528649030"
    os.environ["VERTEXAI_LOCATION"] = "us-east1"

    response = completion(
        model="vertex_ai/177109333002158080",
        messages=[{"content": "Hello, how are you?", "role": "user"}],
    )
    print(f"response: {response}")


@pytest.mark.asyncio
async def get_response():
    load_vertex_ai_credentials()
    prompt = '\ndef count_nums(arr):\n    """\n    Write a function count_nums which takes an array of integers and returns\n    the number of elements which has a sum of digits > 0.\n    If a number is negative, then its first signed digit will be negative:\n    e.g. -123 has signed digits -1, 2, and 3.\n    >>> count_nums([]) == 0\n    >>> count_nums([-1, 11, -11]) == 1\n    >>> count_nums([1, 1, 2]) == 3\n    """\n'
    try:
        response = await acompletion(
            model="gemini-pro",
            messages=[
                {
                    "role": "system",
                    "content": "Complete the given code with no more explanation. Remember that there is a 4-space indent before the first line of your generated code.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        return response
    except litellm.UnprocessableEntityError as e:
        pass
    except Exception as e:
        pytest.fail(f"An error occurred - {str(e)}")


def test_vertex_ai():
    import random

    load_vertex_ai_credentials()
    test_models = (
        litellm.vertex_chat_models
        + litellm.vertex_code_chat_models
        + litellm.vertex_text_models
        + litellm.vertex_code_text_models
    )
    litellm.set_verbose = False
    vertex_ai_project = "reliablekeys"
    # litellm.vertex_project = "reliablekeys"

    test_models = random.sample(test_models, 1)
    # test_models += litellm.vertex_language_models  # always test gemini-pro
    test_models = litellm.vertex_language_models  # always test gemini-pro
    for model in test_models:
        try:
            if model in [
                "code-gecko",
                "code-gecko@001",
                "code-gecko@002",
                "code-gecko@latest",
                "code-bison@001",
                "text-bison@001",
            ]:
                # our account does not have access to this model
                continue
            print("making request", model)
            response = completion(
                model=model,
                messages=[{"role": "user", "content": "hi"}],
                temperature=0.7,
                vertex_ai_project=vertex_ai_project,
            )
            print("\nModel Response", response)
            print(response)
            assert type(response.choices[0].message.content) == str
            assert len(response.choices[0].message.content) > 1
        except Exception as e:
            pytest.fail(f"Error occurred: {e}")


# test_vertex_ai()


def test_vertex_ai_stream():
    load_vertex_ai_credentials()
    litellm.set_verbose = False
    litellm.vertex_project = "reliablekeys"
    import random

    test_models = (
        litellm.vertex_chat_models
        + litellm.vertex_code_chat_models
        + litellm.vertex_text_models
        + litellm.vertex_code_text_models
    )
    test_models = random.sample(test_models, 1)
    test_models += litellm.vertex_language_models  # always test gemini-pro
    for model in test_models:
        try:
            if model in [
                "code-gecko",
                "code-gecko@001",
                "code-gecko@002",
                "code-gecko@latest",
                "code-bison@001",
                "text-bison@001",
            ]:
                # our account does not have access to this model
                continue
            print("making request", model)
            response = completion(
                model=model,
                messages=[
                    {"role": "user", "content": "write 10 line code code for saying hi"}
                ],
                stream=True,
            )
            completed_str = ""
            for chunk in response:
                print(chunk)
                content = chunk.choices[0].delta.content or ""
                print("\n content", content)
                completed_str += content
                assert type(content) == str
                # pass
            assert len(completed_str) > 4
        except Exception as e:
            pytest.fail(f"Error occurred: {e}")


# test_vertex_ai_stream()


@pytest.mark.asyncio
async def test_async_vertexai_response():
    import random

    load_vertex_ai_credentials()
    test_models = (
        litellm.vertex_chat_models
        + litellm.vertex_code_chat_models
        + litellm.vertex_text_models
        + litellm.vertex_code_text_models
    )
    test_models = random.sample(test_models, 1)
    test_models += litellm.vertex_language_models  # always test gemini-pro
    for model in test_models:
        print(f"model being tested in async call: {model}")
        if model in [
            "code-gecko",
            "code-gecko@001",
            "code-gecko@002",
            "code-gecko@latest",
            "code-bison@001",
            "text-bison@001",
        ]:
            # our account does not have access to this model
            continue
        try:
            user_message = "Hello, how are you?"
            messages = [{"content": user_message, "role": "user"}]
            response = await acompletion(
                model=model, messages=messages, temperature=0.7, timeout=5
            )
            print(f"response: {response}")
        except litellm.Timeout as e:
            pass
        except Exception as e:
            pytest.fail(f"An exception occurred: {e}")


# asyncio.run(test_async_vertexai_response())


@pytest.mark.asyncio
async def test_async_vertexai_streaming_response():
    import random

    load_vertex_ai_credentials()
    test_models = (
        litellm.vertex_chat_models
        + litellm.vertex_code_chat_models
        + litellm.vertex_text_models
        + litellm.vertex_code_text_models
    )
    test_models = random.sample(test_models, 1)
    test_models += litellm.vertex_language_models  # always test gemini-pro
    for model in test_models:
        if model in [
            "code-gecko",
            "code-gecko@001",
            "code-gecko@002",
            "code-gecko@latest",
            "code-bison@001",
            "text-bison@001",
        ]:
            # our account does not have access to this model
            continue
        try:
            user_message = "Hello, how are you?"
            messages = [{"content": user_message, "role": "user"}]
            response = await acompletion(
                model="gemini-pro",
                messages=messages,
                temperature=0.7,
                timeout=5,
                stream=True,
            )
            print(f"response: {response}")
            complete_response = ""
            async for chunk in response:
                print(f"chunk: {chunk}")
                complete_response += chunk.choices[0].delta.content
            print(f"complete_response: {complete_response}")
            assert len(complete_response) > 0
        except litellm.Timeout as e:
            pass
        except Exception as e:
            print(e)
            pytest.fail(f"An exception occurred: {e}")


# asyncio.run(test_async_vertexai_streaming_response())


def test_gemini_pro_vision():
    try:
        load_vertex_ai_credentials()
        litellm.set_verbose = True
        litellm.num_retries = 3
        resp = litellm.completion(
            model="vertex_ai/gemini-pro-vision",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Whats in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "gs://cloud-samples-data/generative-ai/image/boats.jpeg"
                            },
                        },
                    ],
                }
            ],
        )
        print(resp)

        prompt_tokens = resp.usage.prompt_tokens

        # DO Not DELETE this ASSERT
        # Google counts the prompt tokens for us, we should ensure we use the tokens from the orignal response
        assert prompt_tokens == 263  # the gemini api returns 263 to us

    except Exception as e:
        if "500 Internal error encountered.'" in str(e):
            pass
        else:
            pytest.fail(f"An exception occurred - {str(e)}")


# test_gemini_pro_vision()


def gemini_pro_function_calling():
    load_vertex_ai_credentials()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    messages = [{"role": "user", "content": "What's the weather like in Boston today?"}]
    completion = litellm.completion(
        model="gemini-pro", messages=messages, tools=tools, tool_choice="auto"
    )
    print(f"completion: {completion}")


# gemini_pro_function_calling()


async def gemini_pro_async_function_calling():
    load_vertex_ai_credentials()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    messages = [{"role": "user", "content": "What's the weather like in Boston today?"}]
    completion = await litellm.acompletion(
        model="gemini-pro", messages=messages, tools=tools, tool_choice="auto"
    )
    print(f"completion: {completion}")


asyncio.run(gemini_pro_async_function_calling())

# Extra gemini Vision tests for completion + stream, async, async + stream
# if we run into issues with gemini, we will also add these to our ci/cd pipeline
# def test_gemini_pro_vision_stream():
#     try:
#         litellm.set_verbose = False
#         litellm.num_retries=0
#         print("streaming response from gemini-pro-vision")
#         resp = litellm.completion(
#             model = "vertex_ai/gemini-pro-vision",
#             messages=[
#                 {
#                     "role": "user",
#                     "content": [
#                                     {
#                                         "type": "text",
#                                         "text": "Whats in this image?"
#                                     },
#                                     {
#                                         "type": "image_url",
#                                         "image_url": {
#                                         "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
#                                         }
#                                     }
#                                 ]
#                 }
#             ],
#             stream=True
#         )
#         print(resp)
#         for chunk in resp:
#             print(chunk)
#     except Exception as e:
#         import traceback
#         traceback.print_exc()
#         raise e
# test_gemini_pro_vision_stream()

# def test_gemini_pro_vision_async():
#     try:
#         litellm.set_verbose = True
#         litellm.num_retries=0
#         async def test():
#             resp = await litellm.acompletion(
#                 model = "vertex_ai/gemini-pro-vision",
#                 messages=[
#                     {
#                         "role": "user",
#                         "content": [
#                                         {
#                                             "type": "text",
#                                             "text": "Whats in this image?"
#                                         },
#                                         {
#                                             "type": "image_url",
#                                             "image_url": {
#                                             "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
#                                             }
#                                         }
#                                     ]
#                     }
#                 ],
#             )
#             print("async response gemini pro vision")
#             print(resp)
#         asyncio.run(test())
#     except Exception as e:
#         import traceback
#         traceback.print_exc()
#         raise e
# test_gemini_pro_vision_async()


# def test_gemini_pro_vision_async_stream():
#     try:
#         litellm.set_verbose = True
#         litellm.num_retries=0
#         async def test():
#             resp = await litellm.acompletion(
#                 model = "vertex_ai/gemini-pro-vision",
#                 messages=[
#                     {
#                         "role": "user",
#                         "content": [
#                                         {
#                                             "type": "text",
#                                             "text": "Whats in this image?"
#                                         },
#                                         {
#                                             "type": "image_url",
#                                             "image_url": {
#                                             "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
#                                             }
#                                         }
#                                     ]
#                     }
#                 ],
#                 stream=True
#             )
#             print("async response gemini pro vision")
#             print(resp)
#             for chunk in resp:
#                 print(chunk)
#         asyncio.run(test())
#     except Exception as e:
#         import traceback
#         traceback.print_exc()
#         raise e
# test_gemini_pro_vision_async()
