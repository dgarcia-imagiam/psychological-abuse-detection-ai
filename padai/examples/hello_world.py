import padai.config.bootstrap  # noqa: F401 always first import in main entry points

from padai.llms.openai import get_default_chat_openai
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import logging
logger = logging.getLogger(__name__)


def build_chain():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a concise assistant. Respond briefly."),
            ("human", "{user_input}"),
        ]
    )
    llm = get_default_chat_openai()
    parser = StrOutputParser()

    return prompt | llm | parser


def main(argv: list[str] | None = None) -> None:
    user_input = "Hello, world!"

    chain = build_chain()
    response: str = chain.invoke({"user_input": user_input})

    logger.info(response)


if __name__ == "__main__":
    main()
