from padai.config.settings import settings
from langchain_aws import ChatBedrockConverse
from typing import Dict, Any
from botocore.config import Config

cfg = Config(
    read_timeout=300,           # 5 minutes; raise if needed
    connect_timeout=10,         # handshake timeout
    retries={"max_attempts": 10, "mode": "standard"}
)


def get_default_chat_bedrock() -> ChatBedrockConverse:
    return get_chat_bedrock(settings.bedrock.chat.as_kwargs())


def get_chat_bedrock(params: Dict[str, Any]) -> ChatBedrockConverse:
    return ChatBedrockConverse(
        **params,
        aws_access_key_id=settings.bedrock.aws_access_key_id.get_secret_value(),
        aws_secret_access_key=settings.bedrock.aws_secret_access_key.get_secret_value(),
        # config=cfg
    )
