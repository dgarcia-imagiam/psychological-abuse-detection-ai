import padai.config.bootstrap  # noqa: F401 always first import in main entry points

from padai.datasets.psychological_abuse import get_communications_df, get_or_create_communication
from padai.utils.llm_cache import set_llm_sqlite_cache
from padai.llms.available import default_available_models
from padai.utils.text import strip_text
from padai.config.language import Language
from padai.llms.base import ChatModelDescriptionEx
from padai.chains.abuse_analyzer import (
    get_abuse_analyzer_params,
    get_abuse_analyzer_prompts,
    get_abuse_analyzer_compare_llm_params,
    get_abuse_analyzer_compare_llm_prompts,
)
from padai.prompts.psychological_abuse import compare_llm_responses
from typing import Dict, MutableMapping, List
from itertools import combinations
from padai.chains.base import build_prompt_llm_parser_chain
from padai.plots.compare_llms import (
    create_compare_llm_figure,
    create_empty_compare_llm_dataframe,
    get_scores,
    get_scores_many,
    normalize_scores,
    create_compare_llm_barplot_figure,
)
from padai.config.settings import settings
from padai.utils.path import safe_file_name
import hashlib
import logging
import pandas as pd


logger = logging.getLogger(__name__)

LLMCache = MutableMapping[tuple[str, str], str]


def invoke(
        severity: str,
        text: str,
        context: str,
        language: Language,
        description: ChatModelDescriptionEx,
):
    params: Dict[str, str] = get_abuse_analyzer_params(text, user_context=context)

    system_prompt, human_prompt = get_abuse_analyzer_prompts(language, severity, user_context=context)

    chain = build_prompt_llm_parser_chain(description, system_prompt, human_prompt)
    response: str = chain.invoke(params)

    return response


def _fingerprint(text: str, context: str) -> str:
    h = hashlib.sha256()
    h.update(text.encode())
    h.update(b"\0")
    h.update(context.encode())
    return h.hexdigest()


def invoke_cached(
    llm_cache: LLMCache,
    severity: str,
    text: str,
    context: str,
    language: Language,
    model: ChatModelDescriptionEx,
) -> str:
    key = (_fingerprint(text, context), model.full_name)

    if key in llm_cache:          # hit
        return llm_cache[key]

    response = invoke(severity, text, context, language, model)
    llm_cache[key] = response
    return response


def main() -> None:

    set_llm_sqlite_cache()

    severity = "extreme_vigilant_with_history"

    communications_df = get_communications_df()

    llm_cache: LLMCache = {}

    model_names = [description.full_name for description in default_available_models]

    scores_many: List[pd.DataFrame] = []

    cache_path = settings.path_in_cache("abuse_analyzer_compare_llms", is_file=False)

    for id_ in communications_df.index:
        communication = get_or_create_communication(id_, communications_df)

        text = communication["text"]
        context = strip_text(communication["context"])
        language = Language(communication["language"])

        for referee in default_available_models:
            logger.info(f"Referee: {referee.full_name}")

            df_path = cache_path / safe_file_name(f"df.{id_}.{referee.full_name}.pkl")

            if df_path.exists():
                df = pd.read_pickle(df_path)
            else:
                df = create_empty_compare_llm_dataframe(model_names)

                for left, right in combinations(default_available_models, 2):
                    logger.info(f"{left.full_name} vs {right.full_name}")

                    left_response: str = invoke_cached(llm_cache, severity, text, context, language, left)
                    right_response: str = invoke_cached(llm_cache, severity, text, context, language, right)

                    params = get_abuse_analyzer_compare_llm_params(text, left_response, right_response, context=context)

                    system_prompt, human_prompt = get_abuse_analyzer_compare_llm_prompts(language)

                    chain = build_prompt_llm_parser_chain(
                        referee,
                        system_prompt,
                        human_prompt,
                        temperature=0,
                        top_p=1,
                    )
                    response: str = chain.invoke(params)

                    logger.info(f"Response: {response}")

                    if response.strip().startswith(compare_llm_responses[language]["left"]):
                        df.at[left.full_name, right.full_name] = 2
                        df.at[right.full_name, left.full_name] = 0
                    elif response.strip().startswith(compare_llm_responses[language]["right"]):
                        df.at[left.full_name, right.full_name] = 0
                        df.at[right.full_name, left.full_name] = 2
                    elif response.strip().startswith(compare_llm_responses[language]["tie"]):
                        df.at[left.full_name, right.full_name] = 1
                        df.at[right.full_name, left.full_name] = 1
                    else:
                        df.at[left.full_name, right.full_name] = -2
                        df.at[right.full_name, left.full_name] = -2

                df.to_pickle(df_path)

            fig = create_compare_llm_figure(df)
            fig.show()

            scores = get_scores(df)
            scores_many.append(scores)

            scores_many_normalized = normalize_scores(get_scores_many(scores_many))

            barplot = create_compare_llm_barplot_figure(scores_many_normalized)
            barplot.show()


if __name__ == "__main__":
    main()
