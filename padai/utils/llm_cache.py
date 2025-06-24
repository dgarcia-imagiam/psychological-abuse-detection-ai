from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from padai.config.settings import settings
from pathlib import Path


def get_llm_cache_path() -> Path:
    return settings.path_in_cache("llm_runs/lc_cache.sqlite")


def set_llm_sqlite_cache():
    set_llm_cache(SQLiteCache(database_path=str(get_llm_cache_path())))

