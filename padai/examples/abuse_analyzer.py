import padai.config.bootstrap  # noqa: F401 always first import in main entry points

from padai.llms.openai import get_default_chat_openai
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from padai.datasets.psychological_abuse import get_communications_df, get_communications_text_sample

import logging
logger = logging.getLogger(__name__)


def build_chain():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Eres un experto en psicología forense que analiza mensajes escritos para detectar maltrato psicológico. Existe un historial confirmado y recurrente de maltrato psicológico hacia la persona receptora; mantén una vigilancia especial y un umbral bajo para señalar cualquier indicio, por sutil que sea. Cuando el usuario te pida analizar un mensaje, identifica de manera objetiva y neutral cualquier indicio explícito o implícito de maltrato y explica tu razonamiento. Presenta un análisis exhaustivo en español, explicando cada conclusión clave en un texto claro y coherente."),
            ("human", "Analiza el siguiente mensaje en busca de indicios de maltrato psicológico y explica tu razonamiento.\n\nMensaje:\n{user_input}"),
        ]
    )
    llm = get_default_chat_openai()
    parser = StrOutputParser()

    return prompt | llm | parser


def main(argv: list[str] | None = None) -> None:

    communications_df = get_communications_df()
    user_input = get_communications_text_sample(communications_df, "es")

    chain = build_chain()
    response: str = chain.invoke({"user_input": user_input})

    logger.info(response)


if __name__ == "__main__":
    main()
