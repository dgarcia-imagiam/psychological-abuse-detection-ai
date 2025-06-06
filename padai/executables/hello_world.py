import padai.config.bootstrap  # noqa: F401 always first import in main entry points

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from padai.config.settings import settings

import logging
logger = logging.getLogger(__name__)


def build_chain(model_name: str = "gpt-4o-mini", temperature: float = 0.0):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a concise assistant. Respond briefly."),
            ("human", "{user_input}"),
        ]
    )

    llm = ChatOpenAI(model=model_name, temperature=temperature, api_key=settings.openai.api_key)

    parser = StrOutputParser()

    return prompt | llm | parser


def main(argv: list[str] | None = None) -> None:
    user_input = "Hello, world!"

    chain = build_chain()
    response: str = chain.invoke({"user_input": user_input})

    logger.info(response)


if __name__ == "__main__":
    main()
