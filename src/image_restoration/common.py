from pathlib import Path


def parent(path: Path, n: int) -> Path:
    """ Get the n-th parent directory of a path
    """
    for _ in range(n):
        path = path.parent
    return path


CONFIGS_DIR = parent(Path(__file__), 3) / 'configs'
ASSETS_DIR  = parent(Path(__file__), 3) / 'assets'