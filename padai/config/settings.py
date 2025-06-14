from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal, List
from pathlib import Path
from pydantic import Field, SecretStr
from padai.config.logging import LoggingSettings
from padai.config.openai import OpenAISettings
from padai.config.aws import BedrockSettings
from padai.config.language import Language

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class AppSettings(BaseSettings):
    name: str = "App"
    environment: Literal["dev", "staging", "prod"] = "dev"
    debug: bool = True

    language: Language = Language.ES
    available_languages: List[Language] = Field(
        default_factory=lambda: list(Language)
    )

    secret: SecretStr

    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    bedrock: BedrockSettings = Field(default_factory=BedrockSettings)

    default_chat_model: Literal["openai", "bedrock"] = "openai"

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        env_prefix="APP_",
        env_nested_delimiter="__",
    )

    def init_logging(self) -> None:
        import logging.config
        logging.config.dictConfig(self.logging.as_dict())


settings = AppSettings()

