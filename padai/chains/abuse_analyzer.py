from typing import Optional, Dict, Tuple
from padai.prompts.psychological_abuse import abuse_analyzer_prompts, abuse_analyzer_prompts_with_context, abuse_analyzer_compare_prompts
from padai.config.settings import settings
from padai.config.language import Language
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


def get_abuse_analyzer_prompts(language: Language, severity: str, user_context: Optional[str] = None) -> Tuple[str, str]:
    if user_context:
        system_prompt = abuse_analyzer_prompts_with_context[language]["system"][severity]
        human_prompt = abuse_analyzer_prompts_with_context[language]["human"]["default"]
    else:
        system_prompt = abuse_analyzer_prompts[language]["system"][severity]
        human_prompt = abuse_analyzer_prompts[language]["human"]["default"]

    return system_prompt, human_prompt


def get_abuse_analyzer_compare_llm_params(text: str, left: str, right: str, context: Optional[str] = None) -> Dict[str, str]:
    params = {
        "text": text,
        "left": left,
        "right": right,
    }

    if context:
        params["context"] = context

    return params


def get_abuse_analyzer_compare_llm_prompts(language: Language) -> Tuple[str, str]:
    return (
        abuse_analyzer_compare_prompts[language]["system"]["default"],
        abuse_analyzer_compare_prompts[language]["human"]["default"],
    )