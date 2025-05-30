from __future__ import annotations

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def build_chain(model_name: str = "gpt-4o-mini", temperature: float = 0.0):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a concise assistant. Respond briefly."),
            ("human", "{user_input}"),
        ]
    )

    llm = ChatOpenAI(model=model_name, temperature=temperature)

    parser = StrOutputParser()

    return prompt | llm | parser


def main(argv: list[str] | None = None) -> None:
    user_input = "Hello, world!"

    load_dotenv()

    chain = build_chain()
    response: str = chain.invoke({"user_input": user_input})

    print(response)


if __name__ == "__main__":
    main()
