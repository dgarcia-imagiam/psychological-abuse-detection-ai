from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr, BaseModel
from typing import Optional, Dict, Any


class HuggingFaceChatModelDefaults(BaseModel):
    model_id: str = "meta-llama/Llama-3.2-1B-Instruct"
    task: str = "text-generation"
    device: int = -1,
    temperature: Optional[float] = None
    max_new_tokens: Optional[int] = None
    top_p: Optional[float] = None

    def as_kwargs(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True)


class HuggingFaceSettings(BaseSettings):
    hub_token: SecretStr
    chat: HuggingFaceChatModelDefaults = HuggingFaceChatModelDefaults()

    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
    )

