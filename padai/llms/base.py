from padai.config.settings import settings
from padai.llms.aws import get_default_chat_bedrock, get_chat_bedrock
from padai.llms.openai import get_default_chat_openai, get_chat_openai
from padai.llms.google import get_default_chat_google, get_chat_google
from typing import Dict, Any, Callable, Set
from pydantic import BaseModel, ConfigDict, Field, computed_field
from padai.llms.engine import ChatEngine
import pandas as pd


_FACTORIES: dict[str, Callable[[Dict[str, Any]], Any]] = {
    "bedrock": get_chat_bedrock,
    "openai":  get_chat_openai,
    "google": get_chat_google,
}

_DEFAULT_FACTORIES: dict[str, Callable[[], Any]] = {
    "bedrock": get_default_chat_bedrock,
    "openai":  get_default_chat_openai,
    "google": get_default_chat_google,
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

    @computed_field
    @property
    def full_name(self) -> str:
        return f"{self.engine}.{self.params['model']}"

    model_config = ConfigDict(extra="forbid")


class ChatModelDescriptionEx(ChatModelDescription):
    id: str
    label: str
    tags: Set[str] = Field(default_factory=set)

    model_config = ConfigDict(extra="forbid")

    @staticmethod
    def nice_index(
        df: pd.DataFrame,
        registry: dict[str, "ChatModelDescriptionEx"]

    ) -> pd.DataFrame:
        """
        Return *df* with its index replaced by the human-readable labels
        found in *registry* (falling back to the original key when a label
        is missing).

        The order and length of the DataFrame stay the same.
        """
        return df.rename(
            index=lambda idx: registry[idx].label  # preferred
            if idx in registry and registry[idx].label
            else idx  # fallback
        )
