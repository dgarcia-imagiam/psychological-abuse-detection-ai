from padai.config.settings import settings
from padai.llms.aws import get_default_chat_bedrock, get_chat_bedrock
from padai.llms.openai import get_default_chat_openai, get_chat_openai
from typing import Dict, Any, Callable
from pydantic import BaseModel, ConfigDict
from padai.llms.types import ChatEngine


_FACTORIES: dict[str, Callable[[Dict[str, Any]], Any]] = {
    "bedrock": get_chat_bedrock,
    "openai":  get_chat_openai,
}

_DEFAULT_FACTORIES: dict[str, Callable[[], Any]] = {
    "bedrock": get_default_chat_bedrock,
    "openai":  get_default_chat_openai,
}


def get_chat_model(engine: ChatEngine, params: Dict[str, Any]):
    try:
        return _FACTORIES[engine](params)
    except KeyError:
        raise ValueError(f"Unknown chat model: {engine!r}") from None


def get_default_chat_model():
    try:
        return _DEFAULT_FACTORIES[settings.default_chat_model]()
    except KeyError:
        raise ValueError(
            f"Unknown default chat model: {settings.default_chat_model!r}"
        ) from None


class ChatModelDescription(BaseModel):
    engine: ChatEngine
    params: Dict[str, Any]

    model_config = ConfigDict(extra="forbid")
