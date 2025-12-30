from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from utils.build_conversation import build_conversation
from config.config_wrapper import config_wrapper

def load_model(model_name, model_args, use_accel=False):
    model_path = model_args.get('model_path_or_name')
    tp = model_args.get('tp', 8)
    model_components = {}
    if use_accel:
        model_components['use_accel'] = True
        model_components['tokenizer'] = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # model_components['model'] = LLM(model=model_path, tokenizer=model_path, gpu_memory_utilization=0.95, tensor_parallel_size=tp, trust_remote_code=True, disable_custom_all_reduce=True, enforce_eager=True)
        if 'olmo' in model_path.lower():
            # model_components['model'] = LLM(model=model_path, tokenizer=model_path, gpu_memory_utilization=0.95, tensor_parallel_size=tp, trust_remote_code=True, disable_custom_all_reduce=True, enforce_eager=True, max_model_len=32768, rope_scaling={"type": "dynamic", "factor": 8.0})
            model_components['model'] = LLM(model=model_path, tokenizer=model_path, gpu_memory_utilization=0.95, tensor_parallel_size=tp, trust_remote_code=True, disable_custom_all_reduce=True, enforce_eager=True, max_model_len=32768, rope_scaling = {"type": "yarn", "factor": 8.0, "original_max_position_embeddings": 4096, "attention_factor": 1.0, "beta_fast": 32, "beta_slow": 1})
        else:
            model_components['model'] = LLM(model=model_path, tokenizer=model_path, gpu_memory_utilization=0.95, tensor_parallel_size=tp, trust_remote_code=True, disable_custom_all_reduce=True, enforce_eager=True)
        model_components['model_name'] = model_name
    else:
        model_components['use_accel'] = False
        model_components['tokenizer'] = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model_components['model'] = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map='auto')
        model_components['model_name'] = model_name
    return model_components

def infer(prompts, historys=[{}], **kwargs):
    model = kwargs.get('model')
    tokenizer = kwargs.get('tokenizer', None)
    model_name = kwargs.get('model_name', None)
    use_accel = kwargs.get('use_accel', False)
    

    if isinstance(prompts[0], str):
        messages = [build_conversation(history, prompt) for history, prompt in zip(historys, prompts)]
    else:
        raise ValueError("Invalid prompts format")
    
    # if use_accel:
    #     # prompt_token_ids = [tokenizer.apply_chat_template(message, add_generation_prompt=True) for message in messages]
    #     prompt_token_ids = [tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in messages]
    #     stop_token_ids=[tokenizer.eos_token_id]
    #     if 'Llama-3' in model_name:
    #         stop_token_ids.append(tokenizer.convert_tokens_to_ids("<|eot_id|>"))
    #     sampling_params = SamplingParams(max_tokens=config_wrapper.max_tokens, stop_token_ids=stop_token_ids, temperature=config_wrapper.temperatrue)
    #     if 'Qwen3' in model_name and 'Instruct' not in model_name and 'no-thinking' not in model_name:
    #         sampling_params.top_p, sampling_params.top_k, sampling_params.min_p = config_wrapper.top_p, config_wrapper.top_k, config_wrapper.min_p
    #     # outputs = model.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)
    #     outputs = model.generate(prompt_token_ids, sampling_params) # vllm 0.10 # https://qwen.readthedocs.io/zh-cn/latest/deployment/vllm.html
    #     responses = []
    #     for output in outputs:
    #         response = output.outputs[0].text
    #         responses.append(response)
    if use_accel:
        # prompt_token_ids = [tokenizer.apply_chat_template(message, add_generation_prompt=True) for message in messages]
        text = [tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in messages]
        stop_token_ids=[tokenizer.eos_token_id]
        if 'Llama-3' in model_name:
            stop_token_ids.append(tokenizer.convert_tokens_to_ids("<|eot_id|>"))

        sampling_params = SamplingParams(max_tokens=config_wrapper.max_tokens, stop_token_ids=stop_token_ids, temperature=config_wrapper.temperatrue)
        if 'Qwen3' in model_name and 'Instruct' not in model_name and 'no-thinking' not in model_name:
            sampling_params.temperature, sampling_params.top_p, sampling_params.top_k, sampling_params.min_p = 0.6, 0.95, 20, 0
        if 'Qwen3' in model_name and ('Instruct' in model_name or 'no-thinking' in model_name):
            sampling_params.temperature, sampling_params.top_p, sampling_params.top_k, sampling_params.min_p = 0.7, 0.95, 40, 0
        if 'QwQ' in model_name:
            sampling_params.temperature, sampling_params.top_p, sampling_params.top_k, sampling_params.min_p = 0.6, 0.95, 20, 0
        if 'Qwen3' in model_name and 'no-thinking' in model_name:
            text = [tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True, enable_thinking=False) for message in messages] # Switches default thinking to non-thinking modes
        # outputs = model.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)
        outputs = model.generate(text, sampling_params) # vllm 0.10 # https://qwen.readthedocs.io/zh-cn/latest/deployment/vllm.html
        responses = []
        for output in outputs:
            response = output.outputs[0].text
            responses.append(response)
    else:
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, padding=True, truncation=True, return_dict=True, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=config_wrapper.max_tokens, do_sample=False)
        responses = []
        for i, prompt in enumerate(prompts):
            response = tokenizer.decode(outputs[i, len(inputs['input_ids'][i]):], skip_special_tokens=True)
            responses.append(response)

    return responses, [None] * len(responses)

if __name__ == '__main__':

    prompts = [
        '''Who are you?''',
        '''only answer with "I am a chatbot"''',
    ]
    model_args = {
        'model_path_or_name': '01-ai/Yi-1.5-6B-Chat',
        'model_type': 'local',
        'tp': 8
    }
    model_components = load_model("Yi-1.5-6B-Chat", model_args, use_accel=True)
    responses = infer(prompts, None, **model_components)
    for response in responses:
        print(response)