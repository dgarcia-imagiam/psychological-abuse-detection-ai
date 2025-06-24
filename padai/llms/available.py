from padai.llms.base import ChatModelDescriptionEx
from typing import List, Dict


default_available_models: List[ChatModelDescriptionEx] = [
    ChatModelDescriptionEx(id="o3-pro",          label="OpenAI: o3-pro",             engine="openai",    params={"model": "o3-pro", "use_responses_api": True},     tags={"no-temperature"}),
    ChatModelDescriptionEx(id="o3",              label="OpenAI: o3",                 engine="openai",    params={"model": "o3"},                                    tags={"no-temperature"}),
    ChatModelDescriptionEx(id="o3-mini",         label="OpenAI: o3-mini",            engine="openai",    params={"model": "o3-mini"},                               tags={"no-temperature"}),
    ChatModelDescriptionEx(id="o4-mini",         label="OpenAI: o4-mini",            engine="openai",    params={"model": "o4-mini"},                               tags={"no-temperature"}),
    ChatModelDescriptionEx(id="gpt-4.5-preview", label="OpenAI: gpt-4.5-preview",    engine="openai",    params={"model": "gpt-4.5-preview"}),
    ChatModelDescriptionEx(id="gpt-4.1",         label="OpenAI: gpt-4.1",            engine="openai",    params={"model": "gpt-4.1"}),
    ChatModelDescriptionEx(id="gpt-4.1-mini",    label="OpenAI: gpt-4.1-mini",       engine="openai",    params={"model": "gpt-4.1-mini"}),
    ChatModelDescriptionEx(id="gpt-4.1-nano",    label="OpenAI: gpt-4.1-nano",       engine="openai",    params={"model": "gpt-4.1-nano"}),
    ChatModelDescriptionEx(id="gpt-4o",          label="OpenAI: gpt-4o",             engine="openai",    params={"model": "gpt-4o"}),
    ChatModelDescriptionEx(id="gpt-4o-mini",     label="OpenAI: gpt-4o-mini",        engine="openai",    params={"model": "gpt-4o-mini"}),

    ChatModelDescriptionEx(id="nova-premier",    label="AWS BedRock: Nova Premier",  engine="bedrock",   params={"model": "us.amazon.nova-premier-v1:0", "region_name": "us-east-1"}),
    ChatModelDescriptionEx(id="nova-pro",        label="AWS BedRock: Nova Pro",      engine="bedrock",   params={"model": "amazon.nova-pro-v1:0", "region_name": "us-east-1"}),
    ChatModelDescriptionEx(id="nova-lite",       label="AWS BedRock: Nova Lite",     engine="bedrock",   params={"model": "amazon.nova-lite-v1:0", "region_name": "us-east-1"}),
    ChatModelDescriptionEx(id="nova-micro",      label="AWS BedRock: Nova Micro",    engine="bedrock",   params={"model": "amazon.nova-micro-v1:0", "region_name": "us-east-1"}),
    ChatModelDescriptionEx(id="deep-seek",       label="AWS BedRock: DeepSeek",      engine="bedrock",   params={"model": "us.deepseek.r1-v1:0", "region_name": "us-west-2"}),

    ChatModelDescriptionEx(id="gemini-2.5-pro-preview",      label="Google: gemini-2.5-pro-preview",     engine="google",    params={"model": "gemini-2.5-pro-preview-06-05"}),
    ChatModelDescriptionEx(id="gemini-2.5-flash-preview",    label="Google: gemini-2.5-flash-preview",   engine="google",    params={"model": "gemini-2.5-flash-preview-05-20"}),
    ChatModelDescriptionEx(id="gemini-2.0-flash",            label="Google: gemini-2.0-flash",           engine="google",    params={"model": "gemini-2.0-flash"}),
    ChatModelDescriptionEx(id="gemini-2.0-flash-lite",       label="Google: gemini-2.0-flash-lite",      engine="google",    params={"model": "gemini-2.0-flash-lite"}),
]


default_available_models_registry: Dict[str, ChatModelDescriptionEx] = {
    m.id: m for m in default_available_models
}
