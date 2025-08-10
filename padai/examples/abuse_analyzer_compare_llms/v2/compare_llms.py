import padai.config.bootstrap  # noqa: F401 always first import in main entry points

from padai.examples.abuse_analyzer_compare_llms.v2.models import models, models_registry
from padai.examples.abuse_analyzer_compare_llms.common.compare_llms import run
from pathlib import Path


def main() -> None:

    run(
        models,
        models_registry,
        "abuse_analyzer_compare_llms/v2",
    )


if __name__ == "__main__":
    main()
