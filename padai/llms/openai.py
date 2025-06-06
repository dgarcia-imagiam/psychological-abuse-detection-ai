from padai.config.settings import settings
from langchain_openai import ChatOpenAI


def get_default_chat_openai():
    return ChatOpenAI(
        **settings.openai.chat.as_kwargs(),
        api_key=settings.openai.api_key.get_secret_value()
    )
