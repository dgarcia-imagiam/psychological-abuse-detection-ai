import padai.config.bootstrap  # noqa: F401 always first import in main entry points

from padai.llms.base import get_default_chat_model
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from padai.datasets.psychological_abuse import get_communications_df, get_communications_sample
from padai.prompts.psychological_abuse import abuse_analyzer_prompts, abuse_analyzer_prompts_with_context
from padai.config.settings import settings
from typing import Dict

import logging
logger = logging.getLogger(__name__)


def build_chain(params: Dict[str, str]):
    source = (
        abuse_analyzer_prompts_with_context
        if "user_context" in params
        else abuse_analyzer_prompts
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", source[settings.language]["system"]["vigilant"]),
            ("human", source[settings.language]["human"]["default"]),
        ]
    )
    llm = get_default_chat_model()
    parser = StrOutputParser()

    return prompt | llm | parser


def main() -> None:

    communications_df = get_communications_df()
    user_input, user_context = get_communications_sample(communications_df, language=settings.language)

    params = {
        "user_input": user_input,
    }

    if user_context:
        params["user_context"] = user_context

    chain = build_chain(params)
    response: str = chain.invoke(params)

    logger.info(response)


if __name__ == "__main__":
    main()
