from typing import Optional, Dict, Any
from padai.prompts.psychological_abuse import abuse_analyzer_prompts, abuse_analyzer_prompts_with_context
from padai.config.settings import settings
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def get_abuse_analyzer_chain(llm, params: Dict[str, str], severity: Optional[str] = "vigilant"):
    source = (
        abuse_analyzer_prompts_with_context
        if "user_context" in params
        else abuse_analyzer_prompts
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", source[settings.language]["system"][severity]),
            ("human", source[settings.language]["human"]["default"]),
        ]
    )
    parser = StrOutputParser()

    return prompt | llm | parser


def get_abuse_analyzer_params(user_input: str, user_context: Optional[str] = None) -> Dict[str, str]:
    params = {
        "user_input": user_input,
    }

    if user_context:
        params["user_context"] = user_context

    return params
