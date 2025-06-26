from padai.llms.base import ChatModelDescriptionEx, get_chat_model
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Optional


def build_prompt_llm_parser_chain(
    description: ChatModelDescriptionEx,
    system_prompt: str,
    human_prompt: str,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", human_prompt),
        ]
    )

    params = description.params.copy()

    if temperature is not None:
        if "no-temperature" not in description.tags:
            params["temperature"] = temperature

    if top_p is not None:
        params["top_p"] = top_p

    llm = get_chat_model(description.engine, description.params)
    parser = StrOutputParser()

    return prompt | llm | parser
