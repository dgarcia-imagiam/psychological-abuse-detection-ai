import padai.config.bootstrap  # noqa: F401 always first import in main entry points

from padai.llms.openai import get_default_chat_openai
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from padai.datasets.psychological_abuse import get_communications_df, get_communications_text_sample
from padai.prompts.psychological_abuse import abuse_analyzer_prompts
from padai.config.settings import settings

import logging
logger = logging.getLogger(__name__)


def build_chain():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", abuse_analyzer_prompts[settings.language]["system"]["vigilant"]),
            ("human", abuse_analyzer_prompts[settings.language]["human"]["default"]),
        ]
    )
    llm = get_default_chat_openai()
    parser = StrOutputParser()

    return prompt | llm | parser


def main() -> None:

    communications_df = get_communications_df()
    user_input = get_communications_text_sample(communications_df, settings.language)

    chain = build_chain()
    response: str = chain.invoke({"user_input": user_input})

    logger.info(response)


if __name__ == "__main__":
    main()
