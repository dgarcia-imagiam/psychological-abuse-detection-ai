from padai.config.settings import settings
from pathlib import Path
from matplotlib.figure import Figure
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class Experiments:

    def __init__(self, relative: str | Path):
        self.base_path: Path = settings.path_in_experiments(relative, is_file=False)
        self.indexes: Dict[str, int] = {}

    def _get_new_index(self, name: str) -> int:
        if name not in self.indexes:
            self.indexes[name] = 0

        self.indexes[name] += 1

        return self.indexes[name]

    def _get_figure_path(self, name: str, index: int, *, digits: int = 3, ext: str = "png") -> Path:
        folder = self.base_path / "figures" / name
        folder.mkdir(parents=True, exist_ok=True)

        filename = f"{index:0{digits}d}.{ext}"
        return folder / filename

    def add_figure(self, figure: Figure, name: str = "default") -> None:
        if "show" in settings.experiment.figure:
            if figure:
                figure.show()

        if "save" in settings.experiment.figure:
            index = self._get_new_index(name)
            path = self._get_figure_path(name, index)

            if figure:
                figure.savefig(path)

            logger.info(f"Saving figure to \"{path}\"")
