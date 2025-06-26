import re
from pathlib import Path

_illegal = r'[<>:"/\\|?*]+'           # regex for the bad characters


def safe_file_name(name: str) -> Path:
    """
    Return *name* cleaned of characters that are illegal on Windows

    Example
    -------
    >>> safe_file_name("df.1.bedrock.us.amazon.nova-premier-v1:0.pkl")
    Path('df.1.bedrock.us.amazon.nova-premier-v1_0.pkl')
    """
    name = re.sub(_illegal, "_", name).strip(". ")   # no trailing dots/spaces
    return Path(name)
