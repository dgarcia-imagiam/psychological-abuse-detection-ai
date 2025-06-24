from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal, List
from pathlib import Path
from pydantic import Field, SecretStr, computed_field
from padai.config.logging import LoggingSettings
from padai.config.openai import OpenAISettings
from padai.config.aws import BedrockSettings
from padai.config.google import GoogleSettings
from padai.config.language import Language
from padai.llms.engine import ChatEngine
from slugify import slugify


BASE_DIR = Path(__file__).resolve().parent.parent.parent


class AppSettings(BaseSettings):
    name: str = "PADAI"
    environment: Literal["dev", "staging", "prod"] = "dev"
    debug: bool = True
    home: Path | None = None

    language: Language = Language.ES
    available_languages: List[Language] = Field(
        default_factory=lambda: list(Language)
    )

    secret: SecretStr = Field(...)

    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    bedrock: BedrockSettings = Field(default_factory=BedrockSettings)
    google: GoogleSettings = Field(default_factory=GoogleSettings)

    default_chat_model: ChatEngine = "openai"

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        env_prefix="APP_",
        env_nested_delimiter="__",
    )

    @computed_field
    @property
    def safe_name(self) -> str:
        return slugify(self.name, separator="_").lower()

    def model_post_init(self, __context) -> None:
        if self.home is None:
            self.home = Path.home() / f"{self.safe_name}"

    def init_logging(self) -> None:
        import logging.config
        logging.config.dictConfig(self.logging.as_dict())

    @staticmethod
    def _ensure_parent_dirs(path: Path, *, is_file: bool) -> None:
        target = path.parent if is_file else path
        target.mkdir(parents=True, exist_ok=True)

    def path_in_home(
            self,
            relative: str | Path,
            *,
            is_file: bool = True,
            create: bool = True,
    ) -> Path:

        rel_path = Path(relative)
        if rel_path.is_absolute():
            raise ValueError("`relative` must be a relative path, not absolute")
        if ".." in rel_path.parts:
            raise ValueError("`relative` may not contain upward navigation ('..')")

        full_path = (self.home / rel_path).resolve()

        if create:
            self._ensure_parent_dirs(full_path, is_file=is_file)

        return full_path

    def path_in_cache(
            self,
            relative: str | Path,
            *,
            is_file: bool = True,
            create: bool = True,
    ) -> Path:
        return self.path_in_home(Path("cache") / relative, is_file=is_file, create=create)


settings = AppSettings()

