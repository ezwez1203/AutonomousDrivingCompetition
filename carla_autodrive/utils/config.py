"""YAML configuration loader."""
from pathlib import Path
import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def load_config(name: str) -> dict:
    """Read a YAML file from config/ and return it as a dict.

    The name may include or omit the extension, for example "sim" or "sim.yaml".
    """
    if not name.endswith((".yaml", ".yml")):
        name += ".yaml"
    path = CONFIG_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
