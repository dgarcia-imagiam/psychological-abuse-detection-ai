from padai.config.settings import settings
from padai.config.huggingface import get_default_device_int
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from typing import Dict, Any


def get_default_chat_huggingface() -> ChatHuggingFace:
    return get_chat_huggingface(settings.huggingface.chat.as_kwargs())


def get_chat_huggingface(params: Dict[str, Any]) -> ChatHuggingFace:
    gen_keys = {
        "temperature",
        "max_new_tokens",
        "top_p",
        "top_k",
        "do_sample",
        "num_beams",
        "repetition_penalty",
        "return_full_text",
    }

    pipeline_kwargs = {k: v for k, v in params.items() if k in gen_keys}
    model_kwargs = {k: v for k, v in params.items() if k not in gen_keys | {"model_id", "task", "device"}}

    pipeline_kwargs.setdefault("return_full_text", False)  # strip prompt

    llm = HuggingFacePipeline.from_model_id(
        model_id=params["model_id"],
        task=params.get("task", "text-generation"),
        device=params.get("device", get_default_device_int()),
        model_kwargs=model_kwargs,
        pipeline_kwargs=pipeline_kwargs,
    )
    return ChatHuggingFace(llm=llm)
