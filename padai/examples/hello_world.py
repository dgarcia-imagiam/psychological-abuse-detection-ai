import padai.config.bootstrap  # noqa: F401 always first import in main entry points

from padai.llms.openai import get_default_chat_openai
from padai.llms.aws import get_default_chat_bedrock
from padai.llms.google import get_default_chat_google
from padai.llms.huggingface import get_default_chat_huggingface
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import logging
logger = logging.getLogger(__name__)


def build_chain(llm):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a concise assistant. Respond briefly."),
            ("human", "{user_input}"),
        ]
    )
    parser = StrOutputParser()

    return prompt | llm | parser


def log_hello(name, llm):
    user_input = "Hello, world!"

    chain = build_chain(llm)
    response: str = chain.invoke({"user_input": user_input})

    logger.info("%s: %s", name, response)


def main() -> None:
    log_hello("HuggingFace", get_default_chat_huggingface())
    log_hello("OpenAI", get_default_chat_openai())
    log_hello("Bedrock", get_default_chat_bedrock())
    log_hello("Google", get_default_chat_google())


if __name__ == "__main__":
    main()
