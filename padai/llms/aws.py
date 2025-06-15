from padai.config.settings import settings
from langchain_aws import ChatBedrockConverse
from typing import Dict, Any


def get_default_chat_bedrock() -> ChatBedrockConverse:
    return get_chat_bedrock(settings.bedrock.chat.as_kwargs())


def get_chat_bedrock(params: Dict[str, Any]) -> ChatBedrockConverse:
    return ChatBedrockConverse(
        **params,
        aws_access_key_id=settings.bedrock.aws_access_key_id.get_secret_value(),
        aws_secret_access_key=settings.bedrock.aws_secret_access_key.get_secret_value(),
    )
