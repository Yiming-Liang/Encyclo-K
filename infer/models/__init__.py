import importlib

class ModelLoader:
    def __init__(self, model_name, config, use_accel=False):
        self.model_name = model_name
        self.config = config
        self.use_accel = use_accel
        self._model = None

    def _lazy_import(self, module_name, func_name):
        if module_name.startswith('.'):
            module_name = __package__ + module_name
        module = importlib.import_module(module_name)
        return getattr(module, func_name)

    def load_model(self):
        if self._model is None:
            load_func = self._lazy_import(self.config['load'][0], self.config['load'][1])
            if 'api' in self.config.get('call_type'):
                self._model = load_func(
                    self.config['model_path_or_name'], 
                    self.config['base_url'], 
                    self.config['api_key'], 
                    self.config['model'],
                    self.config['call_type']
                )
            else:
                self._model = load_func(self.model_name, self.config, self.use_accel)
        return self._model

    @property
    def model(self):
        return self.load_model()

    @property
    def infer(self):
        return self._lazy_import(self.config['infer'][0], self.config['infer'][1])

class ModelRegistry:
    def __init__(self):
        self.models = {}

    def register_model(self, name, config):
        """Register a model configuration."""
        self.models[name] = ModelLoader(name, config, use_accel=False)

    def load_model(self, choice, use_accel=False):
        """Load a model based on the choice."""
        if choice in self.models:
            self.models[choice].use_accel = use_accel
            return self.models[choice].model
        else:
            raise ValueError(f"Model choice '{choice}' is not supported.")

    def infer(self, choice):
        """Get the inference function for a given model."""
        if choice in self.models:
            return self.models[choice].infer
        else:
            raise ValueError(f"Inference choice '{choice}' is not supported.")

# Initialize model registry
model_registry = ModelRegistry()

# Configuration of models
model_configs = {
    'gpt-4o-2024-11-20': {
        'load': ('.openai_api', 'load_model'),
        'infer': ('.openai_api', 'infer'),
        'model_path_or_name': 'GPT4o',
        'base_url': '',
        'api_key': '',
        'model': 'gpt-4o-2024-11-20',
        'call_type': 'api_chat'
    },
    'gpt-4o-2024-08-06': {
        'load': ('.openai_api', 'load_model'),
        'infer': ('.openai_api', 'infer'),
        'model_path_or_name': 'GPT4o-2024-08-06',
        'base_url': '',
        'api_key': '',
        'model': 'gpt-4o-2024-08-06',
        'call_type': 'api_chat'
    },
    'claude-3-5-sonnet-20241022': {
        'load': ('.openai_api', 'load_model'),
        'infer': ('.openai_api', 'infer'),
        'model_path_or_name': 'Claude-3-5-Sonnet-20241022',
        'base_url': '',
        'api_key': '',
        'model': 'claude-3-5-sonnet-20241022',
        'call_type': 'api_chat'
    },
    'o1-mini': {
        'load': ('.openai_api', 'load_model'),
        'infer': ('.openai_api', 'infer'),
        'model_path_or_name': 'o1-mini',
        'base_url': '',
        'api_key': '',
        'model': 'o1-mini',
        'call_type': 'api_chat'
    },
    'gemini-1.5-pro-002': {
        'load': ('.openai_api', 'load_model'),
        'infer': ('.openai_api', 'infer'),
        'model_path_or_name': 'Gemini-1.5-pro-002',
        'base_url': '',
        'api_key': '',
        'model': 'gemini-1.5-pro-002',
        'call_type': 'api_chat'
    },
    'gpt-4-turbo-2024-04-09': {
        'load': ('.openai_api', 'load_model'),
        'infer': ('.openai_api', 'infer'),
        'model_path_or_name': 'GPT4',
        'base_url': '',
        'api_key': '',
        'model': 'gpt-4-turbo',
        'call_type': 'api_chat'
    },
    'gpt-5.1-2025-11-13': {
        'load': ('.openai_api', 'load_model'),
        'infer': ('.openai_api', 'infer'),
        'model_path_or_name': 'gpt-5.1-2025-11-13',
        'base_url': '',
        'api_key': '',
        'model': 'gpt-5.1-2025-11-13',
        'call_type': 'api_chat'
    },
    'claude-3-5-sonnet-20240620': {
        'load': ('.openai_api', 'load_model'),
        'infer': ('.openai_api', 'infer'),
        'model_path_or_name': 'Claude-3-5-Sonnet-20240620',
        'base_url': '',
        'api_key': '',
        'model': 'claude-3-5-sonnet-20240620',
        'call_type': 'api_chat'
    },
    'claude-opus-4-1-20250805-thinking': {
        'load': ('.openai_api', 'load_model'),
        'infer': ('.openai_api', 'infer'),
        'model_path_or_name': 'claude-opus-4-1-20250805-thinking',
        'base_url': '',
        'api_key': '',
        'model': 'claude-opus-4-1-20250805-thinking',
        'call_type': 'api_chat'
    },
    'gemini-2.0-flash-exp': {
        'load': ('.openai_api', 'load_model'),
        'infer': ('.openai_api', 'infer'),
        'model_path_or_name': 'Gemini-2.0-Flash-Exp',
        'base_url': '',
        'api_key': '',
        'model': 'gemini-2.0-flash-exp',
        'call_type': 'api_chat'
    },
    'Llama-3.1-405B': {
        'load': ('.openai_api', 'load_model'),
        'infer': ('.openai_api', 'infer'),
        'model_path_or_name': 'meta-llama/Llama-3.1-405B',
        'base_url': '',
        'api_key': '',
        'model': 'meta-llama/Llama-3.1-405B',
        'call_type': 'api_base'
    },
    'Llama-3.1-405B-Instruct': {
        'load': ('.openai_api', 'load_model'),
        'infer': ('.openai_api', 'infer'),
        'model_path_or_name': 'meta-llama/Llama-3.1-405B-Instruct',
        'base_url': '',
        'api_key': '',
        'model': 'meta-llama/Llama-3.1-405B-Instruct',
        'call_type': 'api_chat'
    },
    'o3-mini-2025-01-31': {
        'load': ('.openai_api', 'load_model'),
        'infer': ('.openai_api', 'infer'),
        'model_path_or_name': 'o3-mini-2025-01-31',
        'base_url': '',
        'api_key': '',
        'model': 'o3-mini-2025-01-31',
        'call_type': 'api_chat'
    },
    'Doubao-1.5-pro-32k-250115': {
        'load': ('.openai_api', 'load_model'),
        'infer': ('.openai_api', 'infer'),
        'model_path_or_name': 'Doubao-1.5-pro-32k-250115',
        'base_url': "", 
        'api_key': "",
        'model': "",
        'call_type': 'api_chat'
    },
    'DeepSeek-R1': {
        'load': ('.openai_api', 'load_model'),
        'infer': ('.openai_api', 'infer'),
        'model_path_or_name': 'DeepSeek-R1',
        'base_url': '',
        'api_key': '',
        'model': 'deepseek-reasoner',
        'call_type': 'api_chat'
    },
    'DeepSeek-R1-0528': {
        'load': ('.openai_api', 'load_model'),
        'infer': ('.openai_api', 'infer'),
        'model_path_or_name': 'DeepSeek-R1-0528',
        'base_url': '',
        'api_key': '',
        'model': 'DeepSeek-R1-0528',
        'call_type': 'api_chat'
    },
    'DeepSeek-V3.2': {
        'load': ('.openai_api', 'load_model'),
        'infer': ('.openai_api', 'infer'),
        'model_path_or_name': 'DeepSeek-V3.2',
        'base_url': '',
        'api_key': '',
        'model': 'DeepSeek-V3.2',
        'call_type': 'api_chat'
    },
    'DeepSeek-V3.1-Terminus': {
        'load': ('.openai_api', 'load_model'),
        'infer': ('.openai_api', 'infer'),
        'model_path_or_name': 'DeepSeek-V3.1-Terminus',
        'base_url': '',
        'api_key': '',
        'model': 'DeepSeek-V3.1-Terminus',
        'call_type': 'api_chat'
    },
    'DeepSeek-V3.1-Terminus-non-thinking': {
        'load': ('.openai_api', 'load_model'),
        'infer': ('.openai_api', 'infer'),
        'model_path_or_name': 'DeepSeek-V3.1-Terminus-non-thinking',
        'base_url': '',
        'api_key': '',
        'model': 'DeepSeek-V3.1-Terminus-non-thinking',
        'call_type': 'api_chat'
    },
    'DeepSeek-V3': {
        'load': ('.openai_api', 'load_model'),
        'infer': ('.openai_api', 'infer'),
        'model_path_or_name': 'DeepSeek-V3',
        'base_url': '',
        'api_key': '',
        'model': 'DeepSeek-V3',
        'call_type': 'api_chat'
    },
    'DeepSeek-V3-0324': {
        'load': ('.openai_api', 'load_model'),
        'infer': ('.openai_api', 'infer'),
        'model_path_or_name': 'DeepSeek-V3-0324',
        'base_url': '',
        'api_key': '',
        'model': 'DeepSeek-V3-0324',
        'call_type': 'api_chat'
    },
    'claude-3-7-sonnet-20250219': {
        'load': ('.anthropic_api', 'load_model'),
        'infer': ('.anthropic_api', 'infer'),
        'model_path_or_name': 'claude-3-7-sonnet-20250219',
        'base_url': '',
        'api_key': '',
        'model': 'claude-3-7-sonnet-20250219',
        'call_type': 'api_chat'
    },

    ####### Local Language Aligned models #######
    'phi-4': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'microsoft/phi-4',
        'call_type': 'local',
        'tp': 8
    },
    'granite-3.1-8b-instruct': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'ibm-granite/granite-3.1-8b-instruct',
        'call_type': 'local',
        'tp': 8
    },
    'granite-3.1-2b-instruct': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'ibm-granite/granite-3.1-2b-instruct',
        'call_type': 'local',
        'tp': 8
    },
    'QwQ-32B-Preview': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'Qwen/QwQ-32B-Preview',
        'call_type': 'local',
        'tp': 8
    },
    'Qwen2.5-0.5B-Instruct': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'Qwen/Qwen2.5-0.5B-Instruct',
        'call_type': 'local',
        'tp': 1
    },
    'Qwen2.5-1.5B-Instruct': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'Qwen/Qwen2.5-1.5B-Instruct',
        'call_type': 'local',
        'tp': 2
    },
    'Qwen2.5-3B-Instruct': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'Qwen/Qwen2.5-3B-Instruct',
        'call_type': 'local',
        'tp': 2
    },
    'Qwen2.5-7B-Instruct': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'Qwen/Qwen2.5-7B-Instruct',
        'call_type': 'local',
        'tp': 4
    },
    'Qwen2.5-14B-Instruct': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'Qwen/Qwen2.5-14B-Instruct',
        'call_type': 'local',
        'tp': 8
    },
    'Qwen2.5-32B-Instruct': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'Qwen/Qwen2.5-32B-Instruct',
        'call_type': 'local',
        'tp': 8
    },
    'Qwen2.5-72B-Instruct': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'Qwen/Qwen2.5-72B-Instruct',
        'call_type': 'local',
        'tp': 8
    },
    'K2-Chat': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'LLM360/K2-Chat',
        'call_type': 'local',
        'tp': 8
    },
    'gemma-2-2b-it': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'google/gemma-2-2b-it',
        'call_type': 'local',
        'tp': 1
    },
    'gemma-2-9b-it': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'google/gemma-2-9b-it',
        'call_type': 'local',
        'tp': 2
    },
    'gemma-2-27b-it': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'google/gemma-2-27b-it',
        'call_type': 'local',
        'tp': 8
    },
    'gemma-3-1b-it': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'google/gemma-3-1b-it',
        'call_type': 'local',
        'tp': 1
    },
    'gemma-3-4b-it': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'google/gemma-3-4b-it',
        'call_type': 'local',
        'tp': 2
    },
    'gemma-3-12b-it': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'google/gemma-3-12b-it',
        'call_type': 'local',
        'tp': 4
    },
    'gemma-3-27b-it': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'google/gemma-3-27b-it',
        'call_type': 'local',
        'tp': 8
    },
    'Llama-3.1-8B-Instruct': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'meta-llama/Llama-3.1-8B-Instruct',
        'call_type': 'local',
        'tp': 8
    },
    'Llama-3.1-70B-Instruct': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'meta-llama/Llama-3.1-70B-Instruct',
        'call_type': 'local',
        'tp': 8
    },
    'Llama-3.3-70B-Instruct': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'meta-llama/Llama-3.3-70B-Instruct',
        'call_type': 'local',
        'tp': 8
    },
    'Llama-3.2-3B-Instruct': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'meta-llama/Llama-3.2-3B-Instruct',
        'call_type': 'local',
        'tp': 8
    },
    'Llama-3.2-1B-Instruct': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'meta-llama/Llama-3.2-1B-Instruct',
        'call_type': 'local',
        'tp': 8
    },
    'Yi-1.5-6B-Chat': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': '01-ai/Yi-1.5-6B-Chat',
        'call_type': 'local',
        'tp': 8
    },
    'Yi-1.5-9B-Chat': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': '01-ai/Yi-1.5-9B-Chat',
        'call_type': 'local',
        'tp': 8
    },
    'Yi-1.5-34B-Chat': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': '01-ai/Yi-1.5-34B-Chat',
        'call_type': 'local',
        'tp': 8
    },
    'MAP-Neo-7B-Instruct-v0.1': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'm-a-p/neo_7b_instruct_v0.1',
        'call_type': 'local',
        'tp': 8
    },
    'Mistral-7B-Instruct-v0.3': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'mistralai/Mistral-7B-Instruct-v0.3',
        'call_type': 'local',
        'tp': 8
    },
    'Mistral-Large-Instruct-2411': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'mistralai/Mistral-Large-Instruct-2411',
        'call_type': 'local',
        'tp': 8
    },
    'Mistral-Small-Instruct-2409': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'mistralai/Mistral-Small-Instruct-2409',
        'call_type': 'local',
        'tp': 8
    },
    'Mixtral-8x22B-Instruct-v0.1': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'mistralai/Mixtral-8x22B-Instruct-v0.1',
        'call_type': 'local',
        'tp': 8
    },
    'Mixtral-8x7B-Instruct-v0.1': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
        'call_type': 'local',
        'tp': 8
    },
    'OLMo-2-1124-13B-Instruct': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'allenai/OLMo-2-1124-13B-Instruct',
        'call_type': 'local',
        'tp': 8
    },
    'OLMo-2-1124-7B-Instruct': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'allenai/OLMo-2-1124-7B-Instruct',
        'call_type': 'local',
        'tp': 8
    },

    ####### Local Language Base models #######
    'Qwen2.5-0.5B': {
        'load': ('.hf_causallm_base', 'load_model'),
        'infer': ('.hf_causallm_base', 'infer'),
        'model_path_or_name': 'Qwen/Qwen2.5-0.5B',
        'call_type': 'local',
        'tp': 1
    },
    'Qwen2.5-1.5B': {
        'load': ('.hf_causallm_base', 'load_model'),
        'infer': ('.hf_causallm_base', 'infer'),
        'model_path_or_name': 'Qwen/Qwen2.5-1.5B',
        'call_type': 'local',
        'tp': 2
    },
    'Qwen2.5-3B': {
        'load': ('.hf_causallm_base', 'load_model'),
        'infer': ('.hf_causallm_base', 'infer'),
        'model_path_or_name': 'Qwen/Qwen2.5-3B',
        'call_type': 'local',
        'tp': 2
    },
    'Qwen2.5-7B': {
        'load': ('.hf_causallm_base', 'load_model'),
        'infer': ('.hf_causallm_base', 'infer'),
        'model_path_or_name': 'Qwen/Qwen2.5-7B',
        'call_type': 'local',
        'tp': 4
    },
    'Qwen2.5-14B': {
        'load': ('.hf_causallm_base', 'load_model'),
        'infer': ('.hf_causallm_base', 'infer'),
        'model_path_or_name': 'Qwen/Qwen2.5-14B',
        'call_type': 'local',
        'tp': 8
    },
    'Qwen2.5-32B': {
        'load': ('.hf_causallm_base', 'load_model'),
        'infer': ('.hf_causallm_base', 'infer'),
        'model_path_or_name': 'Qwen/Qwen2.5-32B',
        'call_type': 'local',
        'tp': 8
    },
    'Qwen2.5-72B': {
        'load': ('.hf_causallm_base', 'load_model'),
        'infer': ('.hf_causallm_base', 'infer'),
        'model_path_or_name': 'Qwen/Qwen2.5-72B',
        'call_type': 'local',
        'tp': 8
    },
    'K2': {
        'load': ('.hf_causallm_base', 'load_model'),
        'infer': ('.hf_causallm_base', 'infer'),
        'model_path_or_name': 'LLM360/K2',
        'call_type': 'local',
        'tp': 8
    },
    'gemma-2-2b': {
        'load': ('.hf_causallm_base', 'load_model'),
        'infer': ('.hf_causallm_base', 'infer'),
        'model_path_or_name': 'google/gemma-2-2b',
        'call_type': 'local',
        'tp': 8
    },
    'gemma-2-9b': {
        'load': ('.hf_causallm_base', 'load_model'),
        'infer': ('.hf_causallm_base', 'infer'),
        'model_path_or_name': 'google/gemma-2-9b',
        'call_type': 'local',
        'tp': 8
    },
    'gemma-2-27b': {
        'load': ('.hf_causallm_base', 'load_model'),
        'infer': ('.hf_causallm_base', 'infer'),
        'model_path_or_name': 'google/gemma-2-27b',
        'call_type': 'local',
        'tp': 8
    },
    'gemma-3-1b-pt': {
        'load': ('.hf_causallm_base', 'load_model'),
        'infer': ('.hf_causallm_base', 'infer'),
        'model_path_or_name': 'google/gemma-3-1b-pt',
        'call_type': 'local',
        'tp': 1
    },
    'gemma-3-4b-pt': {
        'load': ('.hf_causallm_base', 'load_model'),
        'infer': ('.hf_causallm_base', 'infer'),
        'model_path_or_name': 'google/gemma-3-4b-pt',
        'call_type': 'local',
        'tp': 2
    },
    'gemma-3-12b-pt': {
        'load': ('.hf_causallm_base', 'load_model'),
        'infer': ('.hf_causallm_base', 'infer'),
        'model_path_or_name': 'google/gemma-3-12b-pt',
        'call_type': 'local',
        'tp': 4
    },
    'gemma-3-27b-pt': {
        'load': ('.hf_causallm_base', 'load_model'),
        'infer': ('.hf_causallm_base', 'infer'),
        'model_path_or_name': 'google/gemma-3-27b-pt',
        'call_type': 'local',
        'tp': 8
    },
    'Llama-3.1-8B': {
        'load': ('.hf_causallm_base', 'load_model'),
        'infer': ('.hf_causallm_base', 'infer'),
        'model_path_or_name': 'meta-llama/Llama-3.1-8B',
        'call_type': 'local',
        'tp': 8
    },
    'Llama-3.1-70B': {
        'load': ('.hf_causallm_base', 'load_model'),
        'infer': ('.hf_causallm_base', 'infer'),
        'model_path_or_name': 'meta-llama/Llama-3.1-70B',
        'call_type': 'local',
        'tp': 8
    },
    'Yi-1.5-6B': {
        'load': ('.hf_causallm_base', 'load_model'),
        'infer': ('.hf_causallm_base', 'infer'),
        'model_path_or_name': '01-ai/Yi-1.5-6B',
        'call_type': 'local',
        'tp': 8
    },
    'Yi-1.5-9B': {
        'load': ('.hf_causallm_base', 'load_model'),
        'infer': ('.hf_causallm_base', 'infer'),
        'model_path_or_name': '01-ai/Yi-1.5-9B',
        'call_type': 'local',
        'tp': 8
    },
    'Yi-1.5-34B': {
        'load': ('.hf_causallm_base', 'load_model'),
        'infer': ('.hf_causallm_base', 'infer'),
        'model_path_or_name': '01-ai/Yi-1.5-34B',
        'call_type': 'local',
        'tp': 8
    },
    'MAP-Neo-7B': {
        'load': ('.hf_causallm_base', 'load_model'),
        'infer': ('.hf_causallm_base', 'infer'),
        'model_path_or_name': 'm-a-p/neo_7b',
        'call_type': 'local',
        'tp': 8
    },
    'Mistral-7B-v0.3': {
        'load': ('.hf_causallm_base', 'load_model'),
        'infer': ('.hf_causallm_base', 'infer'),
        'model_path_or_name': 'mistralai/Mistral-7B-v0.3',
        'call_type': 'local',
        'tp': 8
    },
    'Mixtral-8x22B-v0.1': {
        'load': ('.hf_causallm_base', 'load_model'),
        'infer': ('.hf_causallm_base', 'infer'),
        'model_path_or_name': 'mistralai/Mixtral-8x22B-v0.1',
        'call_type': 'local',
        'tp': 8
    },
    'Mixtral-8x7B-v0.1': {
        'load': ('.hf_causallm_base', 'load_model'),
        'infer': ('.hf_causallm_base', 'infer'),
        'model_path_or_name': 'mistralai/Mixtral-8x7B-v0.1',
        'call_type': 'local',
        'tp': 8
    },
    'OLMo-2-1124-13B': {
        'load': ('.hf_causallm_base', 'load_model'),
        'infer': ('.hf_causallm_base', 'infer'),
        'model_path_or_name': 'allenai/OLMo-2-1124-13B',
        'call_type': 'local',
        'tp': 8
    },
    'OLMo-2-1124-7B': {
        'load': ('.hf_causallm_base', 'load_model'),
        'infer': ('.hf_causallm_base', 'infer'),
        'model_path_or_name': 'allenai/OLMo-2-1124-7B',
        'call_type': 'local',
        'tp': 8
    },
    'granite-3.1-2b-base': {
        'load': ('.hf_causallm_base', 'load_model'),
        'infer': ('.hf_causallm_base', 'infer'),
        'model_path_or_name': 'ibm-granite/granite-3.1-2b-base',
        'call_type': 'local',
        'tp': 8
    },
    'granite-3.1-8b-base': {
        'load': ('.hf_causallm_base', 'load_model'),
        'infer': ('.hf_causallm_base', 'infer'),
        'model_path_or_name': 'ibm-granite/granite-3.1-8b-base',
        'call_type': 'local',
        'tp': 8
    },
    ####### The new model is up to November 6, 2025 #######
    'Qwen3-0.6B': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'Qwen/Qwen3-0.6B',
        'call_type': 'local',
        'tp': 8
    },
    'Qwen3-1.7B': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'Qwen/Qwen3-1.7B',
        'call_type': 'local',
        'tp': 8
    },
    'Qwen3-4B': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'Qwen/Qwen3-4B',
        'call_type': 'local',
        'tp': 8
    },
    'Qwen3-8B': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'Qwen/Qwen3-8B',
        'call_type': 'local',
        'tp': 8
    },
    'Qwen3-14B': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'Qwen/Qwen3-14B',
        'call_type': 'local',
        'tp': 8
    },
    'Qwen3-32B': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'Qwen/Qwen3-32B',
        'call_type': 'local',
        'tp': 8
    },
    'Qwen3-0.6B-Base': {
        'load': ('.hf_causallm_base', 'load_model'),
        'infer': ('.hf_causallm_base', 'infer'),
        'model_path_or_name': 'Qwen/Qwen3-0.6B-Base',
        'call_type': 'local',
        'tp': 8
    },
    'Qwen3-1.7B-Base': {
        'load': ('.hf_causallm_base', 'load_model'),
        'infer': ('.hf_causallm_base', 'infer'),
        'model_path_or_name': 'Qwen/Qwen3-1.7B-Base',
        'call_type': 'local',
        'tp': 8
    },
    'Qwen3-4B-Base': {
        'load': ('.hf_causallm_base', 'load_model'),
        'infer': ('.hf_causallm_base', 'infer'),
        'model_path_or_name': 'Qwen/Qwen3-4B-Base',
        'call_type': 'local',
        'tp': 8
    },
    'Qwen3-8B-Base': {
        'load': ('.hf_causallm_base', 'load_model'),
        'infer': ('.hf_causallm_base', 'infer'),
        'model_path_or_name': 'Qwen/Qwen3-8B-Base',
        'call_type': 'local',
        'tp': 8
    },
    'Qwen3-14B-Base': {
        'load': ('.hf_causallm_base', 'load_model'),
        'infer': ('.hf_causallm_base', 'infer'),
        'model_path_or_name': 'Qwen/Qwen3-14B-Base',
        'call_type': 'local',
        'tp': 8
    },
    'Qwen3-4B-Thinking-2507': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'Qwen/Qwen3-4B-Instruct-2507',
        'call_type': 'local',
        'tp': 8
    },
    'Qwen3-0.6B-non-thinking': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'Qwen/Qwen3-0.6B',
        'call_type': 'local',
        'tp': 8
    },
    'Qwen3-1.7B-non-thinking': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'Qwen/Qwen3-1.7B',
        'call_type': 'local',
        'tp': 8
    },
    'Qwen3-4B-non-thinking': {
        'load': ('.openai_api', 'load_model'),
        'infer': ('.openai_api', 'infer'),
        'model_path_or_name': 'Qwen3-4B-non-thinking',
        'base_url': '',
        'api_key': '',
        'model': 'Qwen3-4B-non-thinking',
        'call_type': 'api_chat'
    },
    'Qwen3-4B-Instruct-2507': {
        'load': ('.openai_api', 'load_model'),
        'infer': ('.openai_api', 'infer'),
        'model_path_or_name': 'Qwen3-4B-Instruct-2507',
        'base_url': '',
        'api_key': '',
        'model': 'Qwen3-4B-Instruct-2507',
        'call_type': 'api_chat'
    },
    'Qwen3-8B-non-thinking': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'Qwen/Qwen3-8B',
        'call_type': 'local',
        'tp': 8
    },
    'Qwen3-14B-non-thinking': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'Qwen/Qwen3-14B',
        'call_type': 'local',
        'tp': 8
    },
    'Qwen3-32B-non-thinking': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'Qwen/Qwen3-32B',
        'call_type': 'local',
        'tp': 8
    },
    'Qwen3-30B-A3B-Thinking-2507': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'Qwen/Qwen3-30B-A3B-Thinking-2507',
        'call_type': 'local',
        'tp': 8
    },
    'Qwen3-30B-A3B-Instruct-2507': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'Qwen/Qwen3-30B-A3B-Instruct-2507',
        'call_type': 'local',
        'tp': 8
    },
    'Qwen3-235B-A22B': {
        'load': ('.openai_api', 'load_model'),
        'infer': ('.openai_api', 'infer'),
        'model_path_or_name': 'Qwen3-235B-A22B',
        'base_url': '',
        'api_key': '',
        'model': 'Qwen/Qwen3-235B-A22B',
        'call_type': 'api_chat'
    },
    'Qwen3-235B-A22B-non-thinking': {
        'load': ('.openai_api', 'load_model'),
        'infer': ('.openai_api', 'infer'),
        'model_path_or_name': 'Qwen3-235B-A22B-non-thinking',
        'base_url': '',
        'api_key': '',
        'model': 'Qwen3-235B-A22B-non-thinking',
        'call_type': 'api_chat'
    },
    'Qwen3-30B-A3B': {
        'load': ('.openai_api', 'load_model'),
        'infer': ('.openai_api', 'infer'),
        'model_path_or_name': 'Qwen3-30B-A3B',
        'base_url': '',
        'api_key': '',
        'model': 'Qwen3-30B-A3B',
        'call_type': 'api_chat'
    },
    'Qwen3-30B-A3B-non-thinking': {
        'load': ('.openai_api', 'load_model'),
        'infer': ('.openai_api', 'infer'),
        'model_path_or_name': 'Qwen3-30B-A3B-non-thinking',
        'base_url': '',
        'api_key': '',
        'model': 'Qwen3-30B-A3B-non-thinking',
        'call_type': 'api_chat'
    },
    'Qwen3-235B-A22B-Thinking-2507': {
        'load': ('.openai_api', 'load_model'),
        'infer': ('.openai_api', 'infer'),
        'model_path_or_name': 'Qwen3-235B-A22B-Thinking-2507',
        'base_url': '',
        'api_key': '',
        'model': 'Qwen3-235B-A22B-Thinking-2507',
        'call_type': 'api_chat'
    },
    'Qwen3-235B-A22B-Instruct-2507': {
        'load': ('.openai_api', 'load_model'),
        'infer': ('.openai_api', 'infer'),
        'model_path_or_name': 'Qwen3-235B-A22B-Instruct-2507',
        'base_url': '',
        'api_key': '',
        'model': 'Qwen/Qwen3-235B-A22B-Instruct-2507',
        'call_type': 'api_chat'
    },
    'GLM-4.6': {
        'load': ('.openai_api', 'load_model'),
        'infer': ('.openai_api', 'infer'),
        'model_path_or_name': 'GLM-4.6',
        'base_url': '',
        'api_key': '',
        'model': 'zai-org/GLM-4.6',
        'call_type': 'api_chat'
    },
    'GLM-4.6-non-thinking': {
        'load': ('.openai_api', 'load_model'),
        'infer': ('.openai_api', 'infer'),
        'model_path_or_name': 'GLM-4.6-non-thinking',
        'base_url': '',
        'api_key': '',
        'model': 'zai-org/GLM-4.6',
        'call_type': 'api_chat'
    },
    'GLM-4.5': {
        'load': ('.openai_api', 'load_model'),
        'infer': ('.openai_api', 'infer'),
        'model_path_or_name': 'GLM-4.5',
        'base_url': '',
        'api_key': '',
        'model': 'zai-org/GLM-4.5',
        'call_type': 'api_chat'
    },
    'Kimi-K2-Instruct-0905': {
        'load': ('.openai_api', 'load_model'),
        'infer': ('.openai_api', 'infer'),
        'model_path_or_name': 'Kimi-K2-Instruct-0905',
        'base_url': '',
        'api_key': '',
        'model': 'moonshotai/Kimi-K2-Instruct-0905',
        'call_type': 'api_chat'
    },
    'QwQ-32B': {
        'load': ('.hf_causallm_chat', 'load_model'),
        'infer': ('.hf_causallm_chat', 'infer'),
        'model_path_or_name': 'Qwen/QwQ-32B',
        'call_type': 'local',
        'tp': 8
    },

}

# replace model paths with local paths if available
LOCAL_MODEL_MAPPING = {
    
    "Qwen2.5-0.5B": "",
    "Qwen2.5-1.5B": "",
    "Qwen2.5-3B": "",
    "Qwen2.5-7B": "",
    "Qwen2.5-14B": "",
    "Qwen2.5-32B": "",
    "Qwen2.5-72B": "",
    "Qwen2.5-0.5B-Instruct": "",
    "Qwen2.5-1.5B-Instruct": "",
    "Qwen2.5-3B-Instruct": "",
    "Qwen2.5-7B-Instruct": "",
    "Qwen2.5-14B-Instruct": "",
    "Qwen2.5-32B-Instruct": "",
    "Qwen2.5-72B-Instruct": "",
    
    "Yi-1.5-6B": "",
    "Yi-1.5-9B": "",
    "Yi-1.5-34B": "", 
    'Yi-1.5-6B-Chat': "",
    'Yi-1.5-9B-Chat': "",
    'Yi-1.5-34B-Chat': "",
    
    'gemma-2-2b': "",
    'gemma-2-9b': "",
    'gemma-2-27b': "",
    'gemma-2-2b-it': "",
    'gemma-2-9b-it': "",
    'gemma-2-27b-it': "",


    'gemma-3-1b-pt':  "",
    'gemma-3-4b-pt':  "",
    'gemma-3-12b-pt': "",
    'gemma-3-27b-pt': "",
    'gemma-3-1b-it':  "",
    'gemma-3-4b-it':  "",
    'gemma-3-12b-it': "",
    'gemma-3-27b-it': "",
    
    'Llama-3.1-8B': "",
    'Llama-3.1-70B': "",
    'Llama-3.1-8B-Instruct': "",
    'Llama-3.1-70B-Instruct': "",
    'Llama-3.3-70B-Instruct': "",
    'Llama-3.2-3B-Instruct': "",
    'Llama-3.2-1B-Instruct': "",

    'DeepSeek-V3': "",

    'Qwen3-0.6B': "",
    'Qwen3-1.7B': "",
    'Qwen3-4B': "",
    'Qwen3-8B': "",
    'Qwen3-14B': "",
    'Qwen3-32B': "",
    'Qwen3-0.6B-Base': "",
    'Qwen3-1.7B-Base': "",
    'Qwen3-4B-Base': "",
    'Qwen3-8B-Base': "",
    'Qwen3-14B-Base': "",
    'Qwen3-4B-Instruct-2507': "",
    'Qwen3-4B-Thinking-2507': "",
    'Qwen3-0.6B-non-thinking': "",
    'Qwen3-1.7B-non-thinking': "",
    'Qwen3-4B-non-thinking': "",
    'Qwen3-8B-non-thinking': "",
    'Qwen3-14B-non-thinking': "",
    'Qwen3-32B-non-thinking': "",
    'Qwen3-30B-A3B': "",
    'Qwen3-30B-A3B-non-thinking': "",
    'Qwen3-30B-A3B-Thinking-2507': "",
    'Qwen3-30B-A3B-Instruct-2507': "",
    'QwQ-32B': "",

    'OLMo-2-1124-7B': "",
    'OLMo-2-1124-13B': "",
    'OLMo-2-1124-7B-Instruct': "",
    'OLMo-2-1124-13B-Instruct': "",

    'MAP-Neo-7B-Instruct-v0.1': "",
    'MAP-Neo-7B': "",

    'Mistral-7B-v0.3': "",
    'Mistral-7B-v0.3-Instruct': "",
    'Mistral-Large-Instruct-2411': "",
    'Mistral-Large-Instruct-2411-Instruct': "",
    'Mistral-Small-Instruct-2409': "",
    'Mistral-Small-Instruct-2409-Instruct': "",

    'phi-4': "",



}

for model_name in model_configs:
    if model_name in LOCAL_MODEL_MAPPING:
        model_configs[model_name]['model_path_or_name'] = LOCAL_MODEL_MAPPING[model_name]

# # Register all models
# for model_name, config in model_configs.items():
#     model_registry.register_model(model_name, config)

def load_model(choice, use_accel=False):
    """Load a specific model based on the choice."""
    model_registry.register_model(choice, model_configs[choice])
    return model_registry.load_model(choice, use_accel)

def infer(choice):
    """Get the inference function for a specific model."""
    return model_registry.infer(choice)

