from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr, BaseModel
from typing import Optional, Dict, Any


class ChatModelDefaults(BaseModel):
    model: str = "amazon.nova-micro-v1:0"
    temperature: float = 0
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    region_name: Optional[str] = None

    def as_kwargs(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True)


class BedrockSettings(BaseSettings):
    aws_access_key_id: SecretStr
    aws_secret_access_key: SecretStr
    chat: ChatModelDefaults = ChatModelDefaults()

    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
    )

