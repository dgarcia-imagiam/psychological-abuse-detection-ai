from padai.config.settings import settings
from langchain_openai import ChatOpenAI
from typing import Dict, Any


def get_default_chat_openai() -> ChatOpenAI:
    return get_chat_openai(settings.openai.chat.as_kwargs())


def get_chat_openai(params: Dict[str, Any]) -> ChatOpenAI:
    return ChatOpenAI(
        **params,
        api_key=settings.openai.api_key.get_secret_value()
    )
