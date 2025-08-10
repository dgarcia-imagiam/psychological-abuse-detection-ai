import padai.config.bootstrap  # noqa: F401 always first import in main entry points

from padai.llms.available import default_available_models, default_available_models_registry
from padai.examples.abuse_analyzer_compare_llms.common.compare_llms import run
from pathlib import Path


def main() -> None:

    run(
        default_available_models,
        default_available_models_registry,
        "abuse_analyzer_compare_llms/v1",
    )


if __name__ == "__main__":
    main()