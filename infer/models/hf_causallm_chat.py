from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from utils.build_conversation import build_conversation
from config.config_wrapper import config_wrapper
import os
import json
from pathlib import Path

# def load_model(model_name, model_args, use_accel=False):
#     model_path = model_args.get('model_path_or_name')
#     tp = model_args.get('tp', 8)
#     model_components = {}
#     if use_accel:
#         model_components['use_accel'] = True
#         # model_components['model'] = LLM(model=model_path, tokenizer=model_path, gpu_memory_utilization=0.95, tensor_parallel_size=tp, trust_remote_code=True, disable_custom_all_reduce=True, enforce_eager=True)
#         model_components['model'] = LLM(model=model_path, gpu_memory_utilization=0.9, tensor_parallel_size=tp, trust_remote_code=True, disable_custom_all_reduce=True, enforce_eager=True)

#     else:
#         model_components['use_accel'] = False
#         model_components['model'] = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map='auto')
    
#     model_components['tokenizer'] = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
#     model_components['model_name'] = model_name
#     model_components['model_path'] = model_path

#     return model_components

def load_model(model_name, model_args, use_accel=False):
    """加载模型并智能读取配置"""
    model_path = model_args.get('model_path_or_name')
    tp = model_args.get('tp', 8)
    
    model_components = {
        'use_accel': use_accel,
        'tokenizer': AutoTokenizer.from_pretrained(model_path, trust_remote_code=True),
        'model_name': model_name,
        'model_path': model_path
    }
    
    if use_accel:
        # 读取 config.json 检查 max_position_embeddings
        extra_kwargs = {}
        try:
            config_path = os.path.join(model_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                max_pos_embeddings = config.get('max_position_embeddings', None)
                
                if max_pos_embeddings and max_pos_embeddings < 32768:
                    target_length = 32768
                    rope_factor = target_length / max_pos_embeddings
                    
                    extra_kwargs['max_model_len'] = target_length
                    extra_kwargs['rope_scaling'] = {
                        "type": "yarn",
                        "factor": rope_factor,
                        "original_max_position_embeddings": max_pos_embeddings
                    }
                    
                    print("=" * 70)
                    print("⚠️  警告：检测到模型原始上下文长度较短，已自动扩展")
                    print(f"📌 原始 max_position_embeddings: {max_pos_embeddings}")
                    print(f"📌 扩展后 max_model_len: {target_length}")
                    print(f"📌 RoPE scaling 类型: yarn")
                    print(f"📌 RoPE scaling factor: {rope_factor:.4f}")
                    print("💡 提示：这将使模型支持更长的上下文，但可能影响性能")
                    print("=" * 70)
                else:
                    print(f"✓ 模型上下文长度: {max_pos_embeddings or '未指定'}")
            else:
                print(f"⚠ 未找到 config.json，使用默认配置")
        except Exception as e:
            print(f"⚠ 读取 config.json 失败: {e}")
        # 直接初始化 LLM
        model_components['model'] = LLM(model=model_path, tokenizer=model_path, gpu_memory_utilization=0.95, tensor_parallel_size=tp, trust_remote_code=True, disable_custom_all_reduce=True, enforce_eager=True, **extra_kwargs)

    else:
        model_components['model'] = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map='auto')
    
    return model_components


def build_SamplingParams(model_path, model_name=None, config_wrapper=None):
    """
    构建 SamplingParams 对象
    优先级: 模型配置 > config_wrapper > 默认值
    """
    import json
    from pathlib import Path
    from transformers import AutoTokenizer
    
    # 默认参数（贪婪解码）
    temperature = 0.0
    top_p = 0.95
    max_tokens = 4096
    
    # 1. 尝试从模型配置加载 sampling 参数
    config_path = Path(model_path) / "generation_config.json"
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            temperature = config.get('temperature', temperature)
            top_p = config.get('top_p', top_p)
            max_tokens = config.get('max_new_tokens', config.get('max_length', max_tokens))
        except Exception as e:
            print(f"Warning: Failed to load generation_config.json: {e}")
    
    # 2. 从 config_wrapper 覆盖（如果存在）
    if config_wrapper:
        temperature = getattr(config_wrapper, 'temperatrue', temperature)
        top_p = getattr(config_wrapper, 'top_p', top_p)
        max_tokens = getattr(config_wrapper, 'max_tokens', max_tokens)
    
    # 3. 获取 stop_token_ids（独立逻辑，三种来源）
    stop_token_ids = []
    
    # 3.1 从 generation_config.json 读取 eos_token_id
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            eos_token_id = config.get('eos_token_id')
            if eos_token_id is not None:
                if isinstance(eos_token_id, list):
                    stop_token_ids.extend(eos_token_id)
                else:
                    stop_token_ids.append(eos_token_id)
        except Exception as e:
            print(f"Warning: Failed to read eos_token_id from generation_config.json: {e}")
    
    # 3.2 从 tokenizer 读取 eos_token_id（如果还没有）
    if not stop_token_ids:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
                stop_token_ids.append(tokenizer.eos_token_id)
        except Exception as e:
            print(f"Warning: Failed to load tokenizer for eos_token_id: {e}")
    
    # # 3.3 添加额外的 stop tokens
    # if model_name and 'Llama-3' in model_name:
    #     try:
    #         tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    #         eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    #         stop_token_ids.append(eot_id)
    #     except Exception as e:
    #         print(f"Warning: Failed to add Llama-3 eot_id: {e}")
    
    # 3.4 去重
    stop_token_ids = list(set(stop_token_ids))

    # 有的模型输出短
    if model_name and ('gemma-2-27b-it' in model_name.lower() or 'qwen3-14b-no-thinking' in model_name.lower() or 'olmo-2-1124-13b-instruct' in model_name.lower()):
        return SamplingParams(temperature=0.6, top_p=0.95, max_tokens=max_tokens, stop_token_ids=stop_token_ids, min_tokens=16)
    
    # 4. 创建 SamplingParams    
    if temperature == 0.0:
        return SamplingParams(temperature=0.0, max_tokens=max_tokens, stop_token_ids=stop_token_ids)
    else:
        return SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens, stop_token_ids=stop_token_ids)

def infer(prompts, historys=[{}], **kwargs):
    model = kwargs.get('model')
    tokenizer = kwargs.get('tokenizer', None)
    model_name = kwargs.get('model_name', None)
    use_accel = kwargs.get('use_accel', False)
    model_path = kwargs.get('model_path', None)

    if isinstance(prompts[0], str):
        messages = [build_conversation(history, prompt) for history, prompt in zip(historys, prompts)]
    else:
        raise ValueError("Invalid prompts format")
    
    if use_accel:
        text = [tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in messages]

        print(f"查看第一条prompt:\n{text[0]}")

        # sampling_params = SamplingParams(max_tokens=config_wrapper.max_tokens, stop_token_ids=stop_token_ids, temperature=config_wrapper.temperature)
        # sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=32768)
        sampling_params = build_SamplingParams(model_path, model_name, config_wrapper)

        # hybrid model non-thinking model setting in apply_chat_template function and sampling_params
        if model_name and 'qwen3' in model_name.lower() and 'no-thinking' in model_name.lower():
            sampling_params.temperature, sampling_params.top_p, sampling_params.top_k, sampling_params.min_p = 0.6, 0.95, 20, 0
            text = [tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True, enable_thinking=False) for message in messages]
        
        print(f"sampling_params:{sampling_params}")
        outputs = model.generate(text, sampling_params)
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
    # Initialize configuration
    from config.config_wrapper import initialize_config
    import config.config_wrapper as cw_module
    
    # Initialize with default config
    initialize_config('config/config_default.yaml')
    
    # Update the global config_wrapper reference so infer() uses the initialized one
    config_wrapper = cw_module.config_wrapper

    input="""Answer the following multiple choice question. Please carefully analyze each statement and option, and end your response with 'Answer: \$OPTION_NUMBER' (no quotes), where OPTION_NUMBER is A, B, C, D, E, F.\n\nReview the following statements and identify those that are accurate: \ni.Descartes and Locke's Philosophy: Descartes argued that systematic doubt was the beginning of firm knowledge. Fifty years later, John Locke provided an account of the psychology of knowledge, reducing its primary constituents to the impressions conveyed by the senses to the mind; he argued against Descartes that there were no innate ideas in human nature. The mind contained only sense - data and the connections it made between them. This implied that mankind had no fixed ideas of right and wrong; moral values, according to Locke, arose as the mind experienced pain and pleasure. There was a huge future for the development of such ideas, from which theories about education, society's duty to regulate material conditions and many other derivations from environmentalism would flow. There was also a huge past behind them: the dualism expressed by Descartes and Locke in their distinctions of body and mind, physical and moral, had its roots in Plato and Christian metaphysics. Perhaps most strikingly, Locke's ideas could still be associated with the traditional framework of Christian belief.\nii.Post-Invasion Iraq Turmoil: Bush and Blair would have escaped some of the criticism they came in for after the invasion of Iraq if the occupation of that country had been better planned. Instead, parts of the country fell into anarchy after the collapse of the regime, as basic services stopped and the economy faltered. Looting and lawlessness were widespread for months before Iraqis - much helped by an American tank - toppled Saddam’s statue in the centre of Baghdad. Even though relations between the main ethnic and religious groups in Iraq would have been difficult to handle for any post-Saddam authority, the lack of security and the economic chaos helped inflame the situation. The majority Shia Muslims - who held significant influence under the former Ba’ath regime’s mainly Sunni leaders - flocked to their religious guides for direction, many of whom advocated for democratic reforms and integration into the new political system rather than establishing an Islamic state similar to that in Iran. Meanwhile, a number of revolts started in the Sunni parts of the country, based both on Saddam loyalists, and increasingly, on Sunni Islamists both from Iraq and other Arab countries. The new Iraqi authorities - a weak coalition government dominated by Shias - remained dependent on United States military support, while the Kurdish northern part of the country set up its own institutions separate from those in Baghdad.\niii.Theodoric, a judicious Ostrogothic ruler, maintained good ties with other barbarians (marrying Clovis’s sister) and held primacy. However, not sharing his people’s Arian faith, religious division weakened Ostrogothic power long-term. Unlike Franks (who eschewed Roman heritage), Ostrogoths embraced it. After Theodoric, they were expelled from Italy by Eastern Roman generals, leaving it ruined. Soon, Italy faced another invasion – by the Lombards, marking a turbulent chapter in its post-Ostrogothic history.\niv.The Coming of Agriculture in China: As in other parts of the world, the coming of agriculture in the area between the Yellow and the Yangzi rivers meant a revolution. This happened not long after 9000 BC in small sections of this area. In a larger area, people exploited vegetation for fibres and food. Rice was harvested in some areas along the Yangzi before the eighth millennium BC, and around the same time, evidence of agriculture (probably millet - growing) appeared just above the flood - level of the Yellow River. The first Chinese agriculture was somewhat like that of early Egypt, being exhaustive or semi - exhaustive, with land cleared, used for a few years, then left to revert to nature while cultivators moved on. Forms of agricultural techniques from the ‘nuclear area of North China’ later spread north, west and south. Complex cultures soon appeared within this area, combining agriculture with the use of jade and wood for carving, the domestication of silk - worms, the making of traditional - form ceremonial vessels, and perhaps even the use of chopsticks. In Neolithic times, this area was already the home of much that was characteristic of later Chinese tradition in the historic area.\nv.Egyptian Medicine: Only in medicine is there indisputable originality and achievement and it can be traced back at least as far as the Old Kingdom. By 1000 BC an Egyptian pre-eminence in this art was internationally recognized. While Egyptian medicine was never wholly separable from magic (magical prescriptions and amulets survive in great numbers), it had an appreciable content of rationality and pure empirical observation. It extended as far as a knowledge of contraceptive techniques. Its indirect contribution to subsequent history was great, too, whatever its efficiency in its own day; much of our knowledge of drugs and of plants furnishing materia medica was first established by the Egyptians and passed directly to the scientists of medieval Europe. It is a considerable thing to have initiated the use of a remedy effective as long as castor oil.\nvi.European Cultural Hegemony: A remarkable aspect of European cultural hegemony is how quickly other peoples responded to it, creating amalgams of their own cultures and foreign imports. By the late nineteenth century, the first stages of such hybrid societies could be found in Asia, with Japan being the clearest example, and parts of China, Southeast Asia, India, Persia, and the Middle East not far behind. Some of this was based on 'defensive modernization' (acquiring European weapons and methods of organization to defend aspects of independence and sovereignty), but more importantly, millions of indigenous populations took what they admired from the colonial or predominant power and gradually made it their own. In ports like Tangier, Cairo, Istanbul, Bombay, Singapore, and Shanghai, young non - Europeans lived very different lives from their fathers, putting immense pressure on politics and value - systems and leading to revolutions that would dominate the twenty - first - century world.\nvii.The eclipse of the imperial power in Japan: The eclipse of the imperial power in Japan was different from what occurred in China, the model of the seventh - century reformers. It was complex, with a progression through the centuries from the exercise of a usurped central authority in the emperor’s name to the virtual disappearance of any central authority at all. There was a fundamental bias in the traditional clan loyalties of Japanese society and the topography of Japan against any central power; remote valleys provided lodgements for great magnates. However, other countries like eighteenth - century Britain (Hanoverian governments) tamed the Scottish highlands with punitive expeditions and military roads.\nviii.Enduring Influence of Ancient Egyptian Religion: Ancient Egyptian religious life’s structure/solidity was sustained by the political forms that supported it, which facilitated its impression on others. Herodotus correctly thought Greeks got god names from Egypt, demonstrating its direct influence. Roman emperors initially tolerated Egyptian god cults but later forbade them due to perceived threats. Even in the 18th century, Egyptian-themed “mumbo-jumbo” charmed Europeans; modern examples like Shriners’ rituals reflect ongoing fascination. Egyptian religion had continuing vigour, preserved by the political frameworks that outlived it, highlighting its long-lasting impact across cultures and time, from ancient Greek accurate perceptions to Roman initial tolerance and subsequent prohibition, 18th-century European fascination, and modern fraternal rituals, with its endurance being a direct consequence of the political systems that once sustained it, demonstrating the profound and persistent allure of ancient Egyptian religious and cultural elements in shaping cross-cultural perceptions and practices over millennia, as its structural solidity and unique mythology were perpetuated by later societies long after the political systems of ancient Egypt had faded, leaving a lasting legacy in the collective imagination and cultural expressions of later civilizations, from classical antiquity to the modern era, where the mystique of Egyptian religion persisted as a source of inspiration, curiosity, and even ritual adaptation, underscoring its remarkable ability to be reinforced within its original political and temporal boundaries.\nix.Optimism and confidence grew as the USSR showed signs of growing division and difficulty in reforming its affairs, primarily driven by the US government's new defensive measures in space. Though thousands of scientists said the project was unrealistic, the Soviet government enthusiastically embraced the challenge and invested heavily to compete. Americans were heartened, too, in 1986 when American bombers were launched from England on a punitive mission against Libya, whose unbalanced ruler had been supporting anti-American terrorists (significantly, the Soviet Union expressed deep concern about this, surpassing the reactions of many west Europeans). President Reagan was largely successful in convincing most of his countrymen that more enthusiastic assertions of American interests in Central America were truly to their advantage. But he remained remarkably popular; only during his presidency did it become apparent that the decade had been one in which the gap between rich and poor in the United States had widened even further.\n\nA) vi.i.ii.\nB) i.ii.viii.ix.\nC) i.ix.\nD) i.vii.iv.vi.\nE) iv.ix.iii.viii.\nF) ix.ii.v.\n"""
    prompts = [input]

    # Prepare histories for each prompt (list of dicts)
    historys = [{} for _ in prompts]

    model_args = {
        'model_path_or_name': '/volume/basedata/users/yzli02/EncycloK_models/Olmo-3-7B-Instruct',
        'model_type': 'local',
        'tp': 8
    }
    
    model_components = load_model("Olmo-3-7B-Instruct", model_args, use_accel=True)
    
    # infer returns (responses, metas)
    responses, _ = infer(prompts, historys, **model_components)
    
    for response in responses:
        print(response)


# from transformers import AutoTokenizer, AutoModelForCausalLM
# from vllm import LLM, SamplingParams
# from utils.build_conversation import build_conversation
# from config.config_wrapper import config_wrapper

# def load_model(model_name, model_args, use_accel=False):
#     model_path = model_args.get('model_path_or_name')
#     tp = model_args.get('tp', 8)
#     model_components = {}
#     if use_accel:
#         model_components['use_accel'] = True
#         model_components['tokenizer'] = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
#         # model_components['model'] = LLM(model=model_path, tokenizer=model_path, gpu_memory_utilization=0.95, tensor_parallel_size=tp, trust_remote_code=True, disable_custom_all_reduce=True, enforce_eager=True)
#         if 'olmo' in model_path.lower():
#             # model_components['model'] = LLM(model=model_path, tokenizer=model_path, gpu_memory_utilization=0.95, tensor_parallel_size=tp, trust_remote_code=True, disable_custom_all_reduce=True, enforce_eager=True, max_model_len=32768, rope_scaling={"type": "dynamic", "factor": 8.0})
#             model_components['model'] = LLM(model=model_path, tokenizer=model_path, gpu_memory_utilization=0.95, tensor_parallel_size=tp, trust_remote_code=True, disable_custom_all_reduce=True, enforce_eager=True, max_model_len=32768, rope_scaling = {"type": "yarn", "factor": 8.0, "original_max_position_embeddings": 4096, "attention_factor": 1.0, "beta_fast": 32, "beta_slow": 1})
#         else:
#             model_components['model'] = LLM(model=model_path, tokenizer=model_path, gpu_memory_utilization=0.95, tensor_parallel_size=tp, trust_remote_code=True, disable_custom_all_reduce=True, enforce_eager=True)
#         model_components['model_name'] = model_name
#     else:
#         model_components['use_accel'] = False
#         model_components['tokenizer'] = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
#         model_components['model'] = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map='auto')
#         model_components['model_name'] = model_name
#     return model_components

# def infer(prompts, historys=[{}], **kwargs):
#     model = kwargs.get('model')
#     tokenizer = kwargs.get('tokenizer', None)
#     model_name = kwargs.get('model_name', None)
#     use_accel = kwargs.get('use_accel', False)
    

#     if isinstance(prompts[0], str):
#         messages = [build_conversation(history, prompt) for history, prompt in zip(historys, prompts)]
#     else:
#         raise ValueError("Invalid prompts format")
    
#     # if use_accel:
#     #     # prompt_token_ids = [tokenizer.apply_chat_template(message, add_generation_prompt=True) for message in messages]
#     #     prompt_token_ids = [tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in messages]
#     #     stop_token_ids=[tokenizer.eos_token_id]
#     #     if 'Llama-3' in model_name:
#     #         stop_token_ids.append(tokenizer.convert_tokens_to_ids("<|eot_id|>"))
#     #     sampling_params = SamplingParams(max_tokens=config_wrapper.max_tokens, stop_token_ids=stop_token_ids, temperature=config_wrapper.temperatrue)
#     #     if 'Qwen3' in model_name and 'Instruct' not in model_name and 'no-thinking' not in model_name:
#     #         sampling_params.top_p, sampling_params.top_k, sampling_params.min_p = config_wrapper.top_p, config_wrapper.top_k, config_wrapper.min_p
#     #     # outputs = model.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)
#     #     outputs = model.generate(prompt_token_ids, sampling_params) # vllm 0.10 # https://qwen.readthedocs.io/zh-cn/latest/deployment/vllm.html
#     #     responses = []
#     #     for output in outputs:
#     #         response = output.outputs[0].text
#     #         responses.append(response)
#     if use_accel:
#         # prompt_token_ids = [tokenizer.apply_chat_template(message, add_generation_prompt=True) for message in messages]
#         text = [tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in messages]
#         stop_token_ids=[tokenizer.eos_token_id]
#         if 'Llama-3' in model_name:
#             stop_token_ids.append(tokenizer.convert_tokens_to_ids("<|eot_id|>"))

#         sampling_params = SamplingParams(max_tokens=config_wrapper.max_tokens, stop_token_ids=stop_token_ids, temperature=config_wrapper.temperatrue)
#         if 'Qwen3' in model_name and 'Instruct' not in model_name and 'no-thinking' not in model_name:
#             sampling_params.temperature, sampling_params.top_p, sampling_params.top_k, sampling_params.min_p = 0.6, 0.95, 20, 0
#         if 'Qwen3' in model_name and ('Instruct' in model_name or 'no-thinking' in model_name):
#             sampling_params.temperature, sampling_params.top_p, sampling_params.top_k, sampling_params.min_p = 0.7, 0.95, 40, 0
#         if 'QwQ' in model_name:
#             sampling_params.temperature, sampling_params.top_p, sampling_params.top_k, sampling_params.min_p = 0.6, 0.95, 20, 0
#         if 'Qwen3' in model_name and 'no-thinking' in model_name:
#             text = [tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True, enable_thinking=False) for message in messages] # Switches default thinking to non-thinking modes
#         # outputs = model.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)
#         outputs = model.generate(text, sampling_params) # vllm 0.10 # https://qwen.readthedocs.io/zh-cn/latest/deployment/vllm.html
#         responses = []
#         for output in outputs:
#             response = output.outputs[0].text
#             responses.append(response)
#     else:
#         inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, padding=True, truncation=True, return_dict=True, return_tensors="pt").to(model.device)
#         outputs = model.generate(**inputs, max_new_tokens=config_wrapper.max_tokens, do_sample=False)
#         responses = []
#         for i, prompt in enumerate(prompts):
#             response = tokenizer.decode(outputs[i, len(inputs['input_ids'][i]):], skip_special_tokens=True)
#             responses.append(response)

#     return responses, [None] * len(responses)

# if __name__ == '__main__':

#     prompts = [
#         '''Who are you?''',
#         '''only answer with "I am a chatbot"''',
#     ]
#     model_args = {
#         'model_path_or_name': '01-ai/Yi-1.5-6B-Chat',
#         'model_type': 'local',
#         'tp': 8
#     }
#     model_components = load_model("Yi-1.5-6B-Chat", model_args, use_accel=True)
#     responses = infer(prompts, None, **model_components)
#     for response in responses:
#         print(response)