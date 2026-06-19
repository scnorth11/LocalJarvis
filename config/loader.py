from pathlib import Path
from typing import Union
import yaml

from .config import Config
from .schema import AppConfig


def load_config(path: Union[str, Path] = "config/settings.yaml") -> Config:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError("config file must contain a mapping at top level")
    app_config = AppConfig.from_dict(raw)
    return Config(app_config)


if __name__ == "__main__":
    cfg = load_config()
    print("Loaded config:", cfg)
