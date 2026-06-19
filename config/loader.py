import logging
import logging.config
import logging.handlers
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
    config = Config(app_config)
    _setup_logging(config)
    return config


def _setup_logging(config: Config) -> None:
    """Configure the Python logging system from ``config/logging.yaml``.

    Creates the log directory specified in ``config.paths.log_dir`` and
    rewrites the file-handler ``filename`` entry to an absolute path so the
    app works regardless of the working directory.
    """
    log_dir = Path(config.paths.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logging_config_path = Path("config/logging.yaml")
    if not logging_config_path.exists():
        return

    with logging_config_path.open("r", encoding="utf-8") as f:
        log_cfg = yaml.safe_load(f)

    if not isinstance(log_cfg, dict):
        return

    # Resolve relative filenames in file handlers to be under log_dir.
    for handler in log_cfg.get("handlers", {}).values():
        if "filename" in handler:
            handler["filename"] = str(log_dir / Path(handler["filename"]).name)

    logging.config.dictConfig(log_cfg)


if __name__ == "__main__":
    cfg = load_config()
    print("Loaded config:", cfg)
