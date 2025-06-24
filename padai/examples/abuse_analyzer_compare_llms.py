import padai.config.bootstrap  # noqa: F401 always first import in main entry points

from padai.datasets.psychological_abuse import get_communications_df, get_or_create_communication
from padai.utils.llm_cache import set_llm_sqlite_cache
from padai.llms.available import default_available_models
from padai.utils.text import strip_text
from padai.prompts.psychological_abuse import abuse_analyzer_prompts, abuse_analyzer_prompts_with_context
from padai.config.language import Language
from padai.llms.base import ChatModelDescription, get_chat_model
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from padai.chains.abuse_analyzer import get_abuse_analyzer_params
from typing import Dict, Any, MutableMapping
from itertools import combinations
import hashlib
import logging


logger = logging.getLogger(__name__)

LLMCache = MutableMapping[tuple[str, str], str]


def build_chain(
    description: ChatModelDescription,
    system_prompt: str,
    human_prompt: str,
):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", human_prompt),
        ]
    )

    llm = get_chat_model(description.engine, description.params)
    parser = StrOutputParser()

    return prompt | llm | parser


def invoke(
        severity: str,
        text: str,
        context: str,
        language: Language,
        description: ChatModelDescription,
):
    params: Dict[str, str] = get_abuse_analyzer_params(text, user_context=context)

    if context:
        system_prompt = abuse_analyzer_prompts_with_context[language]["system"][severity]
        human_prompt = abuse_analyzer_prompts_with_context[language]["human"]["default"]
    else:
        system_prompt = abuse_analyzer_prompts[language]["system"][severity]
        human_prompt = abuse_analyzer_prompts[language]["human"]["default"]

    chain = build_chain(description, system_prompt, human_prompt)
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
    model: ChatModelDescription,
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

    for id_ in communications_df.index:
        communication = get_or_create_communication(id_, communications_df)

        text = communication["text"]
        context = strip_text(communication["context"])
        language = Language(communication["language"])

        for referee in default_available_models:
            logger.info(f"Referee: {referee.full_name}")

            for left, right in combinations(default_available_models, 2):
                logger.info(f"{left.full_name} vs {right.full_name}")

                left_response: str = invoke_cached(llm_cache, severity, text, context, language, left)
                right_response: str = invoke_cached(llm_cache, severity, text, context, language, right)


if __name__ == "__main__":
    main()
