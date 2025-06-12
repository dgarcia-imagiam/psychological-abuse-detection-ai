from padai.config.settings import settings
from padai.llms.aws import get_default_chat_bedrock
from padai.llms.openai import get_default_chat_openai


def get_default_chat_model():

    if settings.default_chat_model == "bedrock":
        return get_default_chat_bedrock()

    if settings.default_chat_model == "openai":
        return get_default_chat_openai()

    raise ValueError("Unknown default chat model")

