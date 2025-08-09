import padai.config.bootstrap  # noqa: F401 always first import in main entry points

from padai.examples.abuse_analyzer_compare_llms.v2.models import models
from padai.examples.abuse_analyzer_compare_llms.common.abuse_analyzer import log_models


def main() -> None:

    log_models(models)


if __name__ == "__main__":
    main()
