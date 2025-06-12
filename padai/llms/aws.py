from padai.config.settings import settings
from langchain_aws import ChatBedrockConverse


def get_default_chat_bedrock():
    return ChatBedrockConverse(
        **settings.bedrock.chat.as_kwargs(),
        aws_access_key_id=settings.bedrock.aws_access_key_id.get_secret_value(),
        aws_secret_access_key=settings.bedrock.aws_secret_access_key.get_secret_value(),
    )