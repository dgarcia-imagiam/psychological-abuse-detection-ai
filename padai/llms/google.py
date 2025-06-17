from padai.config.settings import settings
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Dict, Any


def get_default_chat_google() -> ChatGoogleGenerativeAI:
    return get_chat_google(settings.google.chat.as_kwargs())


def get_chat_google(params: Dict[str, Any]) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        **params,
        google_api_key=settings.google.api_key.get_secret_value()
    )
