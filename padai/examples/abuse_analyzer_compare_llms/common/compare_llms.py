from padai.datasets.psychological_abuse import get_communications_df, get_or_create_communication
from padai.utils.llm_cache import set_llm_sqlite_cache
from padai.utils.text import strip_text, process_response
from padai.config.language import Language
from padai.llms.base import ChatModelDescriptionEx
from padai.chains.abuse_analyzer import (
    get_abuse_analyzer_params,
    get_abuse_analyzer_prompts,
    get_abuse_analyzer_compare_llm_params,
    get_abuse_analyzer_compare_llm_prompts,
)
from padai.prompts.psychological_abuse import compare_llm_responses
from typing import Dict, MutableMapping, List, Set, cast
from itertools import combinations
from padai.chains.base import build_prompt_llm_parser_chain
from padai.plots.compare_llms import (
    create_compare_llm_figure,
    create_empty_compare_llm_dataframe,
    get_row_scores,
    get_row_scores_many,
    normalize_scores,
    create_compare_llm_barplot_figure,
    get_scores,
    get_average_scores,
    get_mode_scores,
    mse_nonneg,
    different_nonneg,
    barplot_with_outliers,
)
from padai.config.settings import settings
from padai.utils.path import safe_file_name
from pathlib import Path
import hashlib
import logging
import pandas as pd
from padai.experiments.base import Experiments


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

    chain, disposable = build_prompt_llm_parser_chain(description, system_prompt, human_prompt)

    try:
        return chain.invoke(params)

    finally:
        # Important: drop chain reference so GC can break the link to llm
        del chain
        disposable.dispose()


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

    response: str = cast(str, invoke(severity, text, context, language, model))
    llm_cache[key] = response
    return response


def get_normalized_row_scores(scores: Dict[int, Dict[str, pd.DataFrame]]) -> pd.DataFrame:

    row_scores: List[pd.DataFrame] = []

    for id_, dict_ in scores.items():
        for full_name, df in dict_.items():
            row_scores.append(get_row_scores(df))

    return normalize_scores(
        get_row_scores_many(
            row_scores
        )
    )


def get_total_scores(scores: Dict[int, Dict[str, pd.DataFrame]]) -> pd.DataFrame:

    dfs: List[pd.DataFrame] = []

    for id_, dict_ in scores.items():
        for full_name, df in dict_.items():
            dfs.append(df)

    return get_scores(dfs)


def get_total_mode_scores(scores: Dict[int, Dict[str, pd.DataFrame]]) -> pd.DataFrame:

    dfs: List[pd.DataFrame] = []

    for id_, dict_ in scores.items():
        for full_name, df in dict_.items():
            dfs.append(df)

    return get_mode_scores(dfs)


def get_referee_errors(scores: Dict[int, Dict[str, pd.DataFrame]]) -> pd.DataFrame:

    full_names: Set[str] = set()

    for id_, dict_ in scores.items():
        for full_name, df in dict_.items():
            full_names.add(full_name)

    errors = pd.DataFrame(0.0, index=sorted(full_names), columns=["sum_mse", "sum_mode", "n"])

    for id_, dict_ in scores.items():
        values = list(dict_.values())

        average_df = get_average_scores(values)
        mode_df = get_mode_scores(values)

        for full_name, df in dict_.items():
            mse = mse_nonneg(df, average_df)
            different = different_nonneg(df, mode_df)

            errors.at[full_name, "sum_mse"] = errors.at[full_name, "sum_mse"] + mse
            errors.at[full_name, "sum_mode"] = errors.at[full_name, "sum_mode"] + different

            errors.at[full_name, "n"] = errors.at[full_name, "n"] + 1

    errors["mse"] = errors["sum_mse"] / errors["n"]
    errors["mode"] = errors["sum_mode"] / errors["n"]

    return errors


def run(
        descriptions: List[ChatModelDescriptionEx],
        descriptions_registry: Dict[str, ChatModelDescriptionEx],
        relative: str | Path,
) -> None:

    set_llm_sqlite_cache()

    severity = "extreme_vigilant_with_history"

    communications_df = get_communications_df()

    llm_cache: LLMCache = {}

    model_names = [description.full_name for description in descriptions]

    scores: Dict[int, Dict[str, pd.DataFrame]] = {}

    cache_path = settings.path_in_cache(relative, is_file=False)

    experiments: Experiments = Experiments(relative)

    for id_ in communications_df.index:
        communication = get_or_create_communication(id_, communications_df)

        text = communication["text"]
        context = strip_text(communication["context"])
        language = Language(communication["language"])

        scores[id_] = {}

        for referee in descriptions:
            logger.info(f"Referee: {referee.full_name}")

            df_path = cache_path / safe_file_name(f"df.{id_}.{referee.full_name}.pkl")

            if df_path.exists():
                df = pd.read_pickle(df_path)
            else:
                df = create_empty_compare_llm_dataframe(model_names)

                for left, right in combinations(descriptions, 2):
                    logger.info(f"{left.full_name} vs {right.full_name}")

                    left_response: str = process_response(invoke_cached(llm_cache, severity, text, context, language, left))
                    right_response: str = process_response(invoke_cached(llm_cache, severity, text, context, language, right))

                    params = get_abuse_analyzer_compare_llm_params(text, left_response, right_response, context=context)

                    system_prompt, human_prompt = get_abuse_analyzer_compare_llm_prompts(language)

                    chain, disposable = build_prompt_llm_parser_chain(
                        referee,
                        system_prompt,
                        human_prompt,
                        temperature=0,
                        top_p=1,
                    )
                    try:
                        response: str = process_response(chain.invoke(params))

                    finally:
                        # Important: drop chain reference so GC can break the link to llm
                        del chain
                        disposable.dispose()

                    logger.info(f"Response: {response}")

                    if response.startswith(compare_llm_responses[language]["left"]):
                        df.at[left.full_name, right.full_name] = 2
                        df.at[right.full_name, left.full_name] = 0
                    elif response.startswith(compare_llm_responses[language]["right"]):
                        df.at[left.full_name, right.full_name] = 0
                        df.at[right.full_name, left.full_name] = 2
                    elif response.startswith(compare_llm_responses[language]["tie"]):
                        df.at[left.full_name, right.full_name] = 1
                        df.at[right.full_name, left.full_name] = 1
                    else:
                        df.at[left.full_name, right.full_name] = -2
                        df.at[right.full_name, left.full_name] = -2

                df.to_pickle(df_path)

            scores[id_][referee.full_name] = df

            fig = create_compare_llm_figure(
                ChatModelDescriptionEx.nice_index(
                    df,
                    descriptions_registry
                ),
                title=f"LLM Score Matrix ({referee.full_name}, {id_})"
            )
            experiments.add_figure(fig, "llm_score_matrix")

            total_df = get_total_scores(scores)
            total_mode_df = get_total_mode_scores(scores)

            total_fig = create_compare_llm_figure(
                ChatModelDescriptionEx.nice_index(
                    total_df,
                    descriptions_registry
                ),
                title="LLM Score Matrix (Average)"
            )
            experiments.add_figure(total_fig, "llm_score_matrix_average")

            total_mode_fig = create_compare_llm_figure(
                ChatModelDescriptionEx.nice_index(
                    total_mode_df,
                    descriptions_registry
                ),
                title="LLM Score Matrix (Mode)"
            )
            experiments.add_figure(total_mode_fig, "llm_score_matrix_mode")

            errors = get_referee_errors(scores)

            errors_mse_barplot = barplot_with_outliers(
                ChatModelDescriptionEx.nice_index(
                    errors[["mse"]].sort_values(by="mse", ascending=True),
                    descriptions_registry
                ),
                title="LLM Referee Errors (MSE)"
            )
            experiments.add_figure(errors_mse_barplot, "llm_referee_errors_mse")

            errors_mode_barplot = barplot_with_outliers(
                ChatModelDescriptionEx.nice_index(
                    errors[["mode"]].sort_values(by="mode", ascending=True),
                    descriptions_registry
                ),
                title="LLM Referee Errors (Mode)",
                decimals=0
            )
            experiments.add_figure(errors_mode_barplot, "llm_referee_errors_mode")

            barplot = create_compare_llm_barplot_figure(
                ChatModelDescriptionEx.nice_index(
                    get_normalized_row_scores(scores),
                    descriptions_registry
                ),
                title="LLM Ranking",
                dpi=200,
            )
            experiments.add_figure(barplot, "llm_ranking")