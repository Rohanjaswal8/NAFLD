from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]


def load_config(path: Path | None = None) -> dict:
    config_path = path or ROOT_DIR / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)
