from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr, BaseModel, Field
from typing import Optional, Dict, Any
import torch


def get_default_device_int() -> int:
    """Return 0 if a CUDA GPU is available, else -1 (CPU)."""
    return 0 if torch.cuda.is_available() else -1


class HuggingFaceChatModelDefaults(BaseModel):
    model_id: str = "meta-llama/Llama-3.2-1B-Instruct"
    task: str = "text-generation"
    device: int = Field(default_factory=get_default_device_int)
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

