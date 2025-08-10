from padai.llms.base import ChatModelDescriptionEx
from padai.config.huggingface import get_default_device_int
from typing import List, Dict

MAX_NEW_TOKENS = 16 * 1024

_BASE_MODELS: List[ChatModelDescriptionEx] = [
    ChatModelDescriptionEx(id="gpt-5",          label="OpenAI: GPT-5",          engine="openai",    params={"model": "gpt-5"},      tags={"no-temperature"}),
    ChatModelDescriptionEx(id="gpt-5-mini",     label="OpenAI: GPT-5-mini",     engine="openai",    params={"model": "gpt-5-mini"}, tags={"no-temperature"}),
    ChatModelDescriptionEx(id="gpt-5-nano",     label="OpenAI: GPT-5-nano",     engine="openai",    params={"model": "gpt-5-nano"}, tags={"no-temperature"}),
    ChatModelDescriptionEx(id="o3",             label="OpenAI: o3",             engine="openai",    params={"model": "o3"},         tags={"no-temperature"}),
    ChatModelDescriptionEx(id="gpt-4.1-mini",   label="OpenAI: GPT-4.1 mini",   engine="openai",    params={"model": "gpt-4.1-mini"}),

    ChatModelDescriptionEx(id="gpt-oss-120b",           label="AWS BedRock: GPT-OSS-120B",              engine="bedrock",   params={"model": "openai.gpt-oss-120b-1:0", "region_name": "us-west-2"}),
    ChatModelDescriptionEx(id="gpt-oss-20b",            label="AWS BedRock: GPT-OSS-20B",               engine="bedrock",   params={"model": "openai.gpt-oss-20b-1:0", "region_name": "us-west-2"}),
    ChatModelDescriptionEx(id="nova-pro",               label="AWS BedRock: Nova Pro",                  engine="bedrock",   params={"model": "amazon.nova-pro-v1:0", "region_name": "us-east-1"}),
    ChatModelDescriptionEx(id="deep-seek",              label="AWS BedRock: DeepSeek-R1",               engine="bedrock",   params={"model": "us.deepseek.r1-v1:0", "region_name": "us-west-2"}),
    ChatModelDescriptionEx(id="llama3-3-70b-instruct",  label="AWS BedRock: Llama 3.3 70B Instruct",    engine="bedrock",   params={"model": "us.meta.llama3-3-70b-instruct-v1:0", "region_name": "us-west-2"}),

    ChatModelDescriptionEx(id="gemini-2.5-pro",         label="Google: Gemini 2.5 Pro",         engine="google",    params={"model": "gemini-2.5-pro"}),
    ChatModelDescriptionEx(id="gemini-2.5-flash",       label="Google: Gemini 2.5 Flash",       engine="google",    params={"model": "gemini-2.5-flash"}),
]

_CPU_MODELS: List[ChatModelDescriptionEx] = [
    ChatModelDescriptionEx(id="Llama-3.2-1B-Instruct",  label="CPU: Meta Llama-3.2-1B-Instruct",    engine="huggingface",   params={"model_id": "meta-llama/Llama-3.2-1B-Instruct", "max_new_tokens": MAX_NEW_TOKENS}),
    ChatModelDescriptionEx(id="gemma-3-1b-it",          label="CPU: Google Gemma 3 1B",             engine="huggingface",   params={"model_id": "google/gemma-3-1b-it", "max_new_tokens": MAX_NEW_TOKENS}),
]

_GPU_MODELS: List[ChatModelDescriptionEx] = [
    ChatModelDescriptionEx(id="phi-4",                      label="GPU: Microsoft Phi-4",               engine="huggingface",   params={"model_id": "microsoft/phi-4", "max_new_tokens": MAX_NEW_TOKENS}),
    ChatModelDescriptionEx(id="gemma-3-12b-it",             label="GPU: Google Gemma 3 12B Instruct",   engine="huggingface",   params={"model_id": "google/gemma-3-12b-it", "max_new_tokens": MAX_NEW_TOKENS}),
    ChatModelDescriptionEx(id="Llama-3.1-8B-Instruct",      label="GPU: Meta Llama 3.1 8B Instruct",    engine="huggingface",   params={"model_id": "meta-llama/Llama-3.1-8B-Instruct", "max_new_tokens": MAX_NEW_TOKENS}),
    ChatModelDescriptionEx(id="deepseek-llm-7b-chat",       label="GPU: DeepSeek LLM 7B Chat",          engine="huggingface",   params={"model_id": "deepseek-ai/deepseek-llm-7b-chat", "max_new_tokens": MAX_NEW_TOKENS}),
    ChatModelDescriptionEx(id="Mistral-7B-Instruct-v0.3",   label="GPU: Mistral 7B Instruct v0.3",      engine="huggingface",   params={"model_id": "mistralai/Mistral-7B-Instruct-v0.3", "max_new_tokens": MAX_NEW_TOKENS}),
    ChatModelDescriptionEx(id="Qwen3-14B",                  label="GPU: Qwen3-14B",                     engine="huggingface",   params={"model_id": "Qwen/Qwen3-14B", "max_new_tokens": MAX_NEW_TOKENS}),
]


def _build_models() -> List["ChatModelDescriptionEx"]:
    out: List[ChatModelDescriptionEx] = list(_BASE_MODELS)
    out.extend(_CPU_MODELS if get_default_device_int() == -1 else _GPU_MODELS)
    return out


models: List[ChatModelDescriptionEx] = _build_models()


models_registry: Dict[str, ChatModelDescriptionEx] = {
    m.full_name: m for m in models
}
