from pydantic import BaseModel, Field
from typing import Set


class ExperimentSettings(BaseModel):
    figure: Set[str] = Field(default_factory=set)

    model_config = dict(extra="forbid")