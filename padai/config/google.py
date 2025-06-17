from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr, BaseModel
from typing import Optional, Dict, Any


class ChatModelDefaults(BaseModel):
    model: str = "gemini-2.0-flash-lite"
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None

    def as_kwargs(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True)


class GoogleSettings(BaseSettings):
    api_key: SecretStr
    chat: ChatModelDefaults = ChatModelDefaults()

    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
    )

