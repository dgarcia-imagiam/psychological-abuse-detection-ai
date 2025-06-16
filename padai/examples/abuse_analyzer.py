import padai.config.bootstrap  # noqa: F401 always first import in main entry points

from padai.llms.base import get_default_chat_model
from padai.datasets.psychological_abuse import get_communications_df, get_communications_sample
from padai.config.settings import settings
from padai.chains.abuse_analyzer import get_abuse_analyzer_chain, get_abuse_analyzer_params
import logging


logger = logging.getLogger(__name__)


def main() -> None:

    communications_df = get_communications_df()
    user_input, user_context = get_communications_sample(communications_df, language=settings.language)

    params = get_abuse_analyzer_params(user_input, user_context=user_context)

    chain = get_abuse_analyzer_chain(get_default_chat_model(), params)
    response: str = chain.invoke(params)

    logger.info(response)


if __name__ == "__main__":
    main()
