from typing import Literal, Dict, Any, List
from pydantic import Field, BaseModel, ConfigDict


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
