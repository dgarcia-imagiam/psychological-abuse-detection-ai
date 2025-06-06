from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal, Dict, Any, List
from pathlib import Path
from pydantic import Field, SecretStr, BaseModel, ConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class FormatterConfig(BaseModel):
    format: str = "%(levelname)s | %(name)s | %(message)s"


class ConsoleHandlerConfig(BaseModel):
    class_: str = Field("logging.StreamHandler", alias="class")
    level: str = "INFO"
    formatter: str = "simple"
    stream: str = "ext://sys.stdout"

    model_config = ConfigDict(populate_by_name=True)


class RootLoggerConfig(BaseModel):
    level: str = "INFO"
    handlers: List[str] = ["console"]


class LoggingSettings(BaseModel):
    version: Literal[1] = 1
    disable_existing_loggers: bool = False

    formatters: Dict[str, FormatterConfig] = Field(
        default_factory=lambda: {"simple": FormatterConfig()}
    )
    handlers: Dict[str, ConsoleHandlerConfig] = Field(
        default_factory=lambda: {"console": ConsoleHandlerConfig()}
    )
    root: RootLoggerConfig = RootLoggerConfig()

    def as_dict(self) -> Dict[str, Any]:
        return self.model_dump(by_alias=True)


class OpenAISettings(BaseSettings):
    api_key: SecretStr


class AppSettings(BaseSettings):
    name: str = "App"
    environment: Literal["dev", "staging", "prod"] = "dev"
    debug: bool = True

    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    openai: OpenAISettings = Field(default_factory=OpenAISettings)

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
