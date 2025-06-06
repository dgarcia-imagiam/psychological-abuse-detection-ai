from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr, BaseModel
from typing import Optional, Dict, Any


class ChatModelDefaults(BaseModel):
    model: str = "gpt-4.1-mini"
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None

    def as_kwargs(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True)


class OpenAISettings(BaseSettings):
    api_key: SecretStr
    chat: ChatModelDefaults = ChatModelDefaults()

    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
    )

