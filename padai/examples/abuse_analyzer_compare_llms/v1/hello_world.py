import padai.config.bootstrap  # noqa: F401 always first import in main entry points

from padai.llms.available import default_available_models
from padai.examples.abuse_analyzer_compare_llms.common.hello_world import log_hello


def main() -> None:

    for model in default_available_models:
        log_hello(model)


if __name__ == "__main__":
    main()
