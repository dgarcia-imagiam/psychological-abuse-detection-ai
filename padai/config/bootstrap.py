import threading
from padai.config.settings import settings

_IS_INITIALISED = False
_LOCK = threading.Lock()


def setup_logging() -> None:
    settings.init_logging()
    if settings.debug:
        import logging
        logger = logging.getLogger(__name__)
        logger.debug("Settings: %s", settings.model_dump_json(indent=4))


def initialise() -> None:
    global _IS_INITIALISED
    with _LOCK:
        if _IS_INITIALISED:
            return
        setup_logging()
        _IS_INITIALISED = True


initialise()
