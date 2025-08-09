from padai.llms.base import get_chat_model
from padai.datasets.psychological_abuse import get_communications_df, get_communications_sample
from padai.config.settings import settings
from padai.chains.abuse_analyzer import get_abuse_analyzer_chain, get_abuse_analyzer_params
from padai.utils.text import process_response
from padai.llms.disposable import make_disposable
import logging


logger = logging.getLogger(__name__)


def log_models(models) -> None:

    communications_df = get_communications_df()
    user_input, user_context = get_communications_sample(communications_df, language=settings.language)

    params = get_abuse_analyzer_params(user_input, user_context=user_context)

    for model in models:

        logger.info(f"Model: {model.full_name}")

        llm = get_chat_model(model.engine, model.params)
        disposable = make_disposable(llm)

        try:
            chain = get_abuse_analyzer_chain(llm, params)
            response: str = process_response(chain.invoke(params))

            logger.info(response)

        finally:
            # Important: drop chain reference so GC can break the link to llm
            del chain
            disposable.dispose()