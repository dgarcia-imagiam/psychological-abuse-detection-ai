from padai.chains.base import build_prompt_llm_parser_chain
from padai.utils.text import process_response

import logging
logger = logging.getLogger(__name__)


def log_hello(description):
    chain, disposable = build_prompt_llm_parser_chain(
        description,
        "You are a concise assistant. Respond briefly.",
        "{user_input}",
    )
    try:
        response: str = process_response(
            chain.invoke({
                "user_input": "Hello, world!",
            })
        )

        logger.info("%s: %s", description.full_name, response)

    finally:
        # Important: drop chain reference so GC can break the link to llm
        del chain
        disposable.dispose()
