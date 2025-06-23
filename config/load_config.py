import os
from pathlib import Path
import yaml


# locate config.yaml next to this file
_DEFAULT_CFG = Path(__file__).resolve().with_name("config.yaml")


def _pick_cfg_file(explicit: str | Path | None) -> Path:
    """
    Decide which YAML file to load, in priority order:
        1) caller-supplied path           → load_config("custom.yaml")
        2) environment variable CONFIG_FILE
        3) default config.yaml next to this file
    Return a *resolved* Path and raise if the file doesn’t exist.
    """
    cfg_file = (
        Path(explicit or os.getenv("CONFIG_FILE", _DEFAULT_CFG)).expanduser().resolve()
    )

    if not cfg_file.exists():
        raise FileNotFoundError(f"Configuration file not found at {cfg_file}")

    return cfg_file


def load_config(config_path: str | Path | None = None) -> dict:
    """ """
    config_file = _pick_cfg_file(config_path)

    with open(config_file, "r") as f:
        cfg = yaml.safe_load(f)

    # -- anchor every relative path to "base_dir" interpreted relative to *this* file
    project_root = (Path(__file__).resolve().parent / cfg["base_dir"]).resolve()
    cfg["ROOT"] = project_root

    def abs_path_(rel_path: str | Path) -> Path:
        return (project_root / rel_path).expanduser().resolve()

    for key in ["input_dir", "output_dir", "model_dir", "model_path", "logs_dir"]:
        cfg[key] = abs_path_(cfg[key])

    return cfg
