from openai import OpenAI
import jsonpickle

from utils.vl_utils import make_interleave_content
from utils.build_conversation import build_conversation
from config.config_wrapper import config_wrapper


def load_model(model_name="GPT4", base_url="", api_key="", model="", call_type='api_chat'):
    model_components = {}
    model_components['model_name'] = model_name
    model_components['model'] = model
    model_components['base_url'] = base_url
    model_components['api_key'] = api_key
    model_components['call_type'] = call_type
    return model_components

def request(messages, timeout=6000, max_tokens=2000, base_url="", api_key="", model="", model_name=None):
    client = OpenAI(base_url=base_url, api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages = messages,
        stream=False,
        max_tokens=max_tokens,
        timeout=timeout,
        temperature=config_wrapper.temperatrue,
    )
    return response

# 参考调用方式 https://bailian.console.aliyun.com/?tab=model&accounttraceid=22feddcc6aca4b3bb315678301ffa716jemp#/model-market/detail/qwen3-235b-a22b
def request_thinking_model(messages, timeout=6000, max_tokens=2000, base_url="", api_key="", model="", model_name=None, enable_thinking=True):
    client = OpenAI(base_url=base_url, api_key=api_key)
    
    
    # 根据thinking模式调整采样参数 Qwen3推荐设置
    if enable_thinking:
        temperature, top_p= 0.6, 0.95
    else:
        temperature, top_p = 0.7, 0.8

     # 构建extra_body参数
    extra_body = {
        "top_k": 20,
        "min_p": 0,
        "chat_template_kwargs": {"enable_thinking": enable_thinking}
    }
    # Qwen3 thinking推荐设置 temperature, top_p, top_k, min_p = 0.6, 0.95, 20, 0
    # Qwen3 no-thinking推荐设置 temperature, top_p, top_k, min_p = 0.7, 0.8, 20, 0

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False,
        max_tokens=max_tokens,
        timeout=timeout,
        temperature=temperature,
        top_p=top_p,
        extra_body=extra_body  # 添加这个参数
    )
    return response

def request_to_base_model(prompt, timeout=6000, max_tokens=2000, base_url="", api_key="", model="", model_name=None):
    client = OpenAI(base_url=base_url, api_key=api_key)
    response = client.completions.create(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        timeout=timeout
    )
    print(response)
    
    return response

def request_with_images(texts_or_image_paths, timeout=60, max_tokens=2000, base_url="", api_key="", model="gpt-4o", model_name=None):
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": make_interleave_content(texts_or_image_paths),
            }  
        ],  
        max_tokens=max_tokens,
        timeout=timeout,
    )
    return response

def infer(prompts, historys=[{}], **kwargs):
    model = kwargs.get('model')
    base_url = kwargs.get('base_url')
    api_key = kwargs.get('api_key')
    model_name = kwargs.get('model_name', None)
    call_type = kwargs.get('call_type', 'api_chat')
    try:
        if call_type == 'api_chat':
            if isinstance(prompts, list):
                if len(prompts) > 1:
                    print(f'[Warning] infer/models/openai_api.py: Multiple prompts detected, only the first one will be processed')
                prompts = prompts[0]
                historys = historys[0]
            if isinstance(prompts, dict) and 'images' in prompts:
                prompts, images = prompts['prompt'], prompts['images']
                images = ["<|image|>" + image for image in images]
                response = request_with_images([prompts, *images], max_tokens=config_wrapper.max_tokens, base_url=base_url, api_key=api_key, model=model, model_name=model_name)
                meta_response = jsonpickle.encode(response, unpicklable=True)
                response = response.choices[0].message.content
            else:
                messages = build_conversation(historys, prompts)
                if 'Qwen3' in model_name and ('Instruct' in model_name or 'no-thinking' in model_name):
                    response = request_thinking_model(messages, max_tokens=config_wrapper.max_tokens, base_url=base_url, api_key=api_key, model=model, model_name=model_name, enable_thinking=False)
                elif 'Qwen3' in model_name and 'Instruct' not in model_name and 'no-thinking' not in model_name:
                    response = request_thinking_model(messages, max_tokens=config_wrapper.max_tokens, base_url=base_url, api_key=api_key, model=model, model_name=model_name, enable_thinking=True)
                else:
                    response = request(messages, max_tokens=config_wrapper.max_tokens, base_url=base_url, api_key=api_key, model=model, model_name=model_name)
                meta_response = jsonpickle.encode(response, unpicklable=True)
                response = response.choices[0].message.content
        elif call_type == 'api_base':
            response = request_to_base_model(prompts, max_tokens=config_wrapper.max_tokens, base_url=base_url, api_key=api_key, model=model, model_name=model_name)
            meta_response = jsonpickle.encode(response, unpicklable=True)
            response = response.choices[0].choices[0].text
        else:
            raise ValueError(f'Invalid call_type: {call_type}')
    except Exception as e:
        response = {"error": str(e)}
        meta_response = response

    if config_wrapper.print_response:
        print(response)
    if config_wrapper.print_meta_response:
        print(meta_response)
    return [response], [meta_response]

# def infer(prompts, historys=[{}], **kwargs):
#     model = kwargs.get('model')
#     base_url = kwargs.get('base_url')
#     api_key = kwargs.get('api_key')
#     model_name = kwargs.get('model_name', None)
#     call_type = kwargs.get('call_type', 'api_chat')
#     try:
#         if call_type == 'api_chat':
#             if isinstance(prompts, list):
#                 if len(prompts) > 1:
#                     print(f'[Warning] infer/models/openai_api.py: Multiple prompts detected, only the first one will be processed')
#                 prompts = prompts[0]
#                 historys = historys[0]
#             if isinstance(prompts, dict) and 'images' in prompts:
#                 prompts, images = prompts['prompt'], prompts['images']
#                 images = ["<|image|>" + image for image in images]
#                 response = request_with_images([prompts, *images], max_tokens=config_wrapper.max_tokens, base_url=base_url, api_key=api_key, model=model, model_name=model_name)
#                 meta_response = jsonpickle.encode(response, unpicklable=True)
#                 response = response.choices[0].message.content
#             else:
#                 messages = build_conversation(historys, prompts)
#                 response = request(messages, max_tokens=config_wrapper.max_tokens, base_url=base_url, api_key=api_key, model=model, model_name=model_name)
#                 meta_response = jsonpickle.encode(response, unpicklable=True)
#                 response = response.choices[0].message.content
#         elif call_type == 'api_base':
#             response = request_to_base_model(prompts, max_tokens=config_wrapper.max_tokens, base_url=base_url, api_key=api_key, model=model, model_name=model_name)
#             meta_response = jsonpickle.encode(response, unpicklable=True)
#             response = response.choices[0].choices[0].text
#         else:
#             raise ValueError(f'Invalid call_type: {call_type}')
#     except Exception as e:
#         response = {"error": str(e)}
#         meta_response = response

#     if config_wrapper.print_response:
#         print(response)
#     if config_wrapper.print_meta_response:
#         print(meta_response)
#     return [response], [meta_response]

    

