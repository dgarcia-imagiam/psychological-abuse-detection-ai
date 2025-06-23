import padai.config.bootstrap  # noqa: F401 always first import in main entry points

from padai.datasets.psychological_abuse import get_communications_df, get_or_create_communication
import logging


logger = logging.getLogger(__name__)


def main() -> None:

    communications_df = get_communications_df()

    for id_ in communications_df.index:
        get_or_create_communication(id_, communications_df)


if __name__ == "__main__":
    main()
