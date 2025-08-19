import padai.config.bootstrap  # noqa: F401 always first import in main entry points

from padai.examples.abuse_analyzer_compare_llms.v2.models import models, models_registry
from padai.examples.abuse_analyzer_compare_llms.common.compare_llms import run


def main() -> None:

    run(
        models,
        models_registry,
        "abuse_analyzer_compare_llms/v2",
        ignore_referees={
            "huggingface.microsoft/phi-4",
            "huggingface.deepseek-ai/deepseek-llm-7b-chat",
            "huggingface.mistralai/Mistral-7B-Instruct-v0.3",
        }
    )


if __name__ == "__main__":
    main()
