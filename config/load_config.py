import os
from pathlib import Path
import yaml

def load_config(config_path: str | None = None) -> dict:
    if config_path is None:
        env_path = os.getenv("CONFIG_FILE")
        if env_path:
            config_path = env_path
        else:
            config_path =  Path(__file__).resolve().with_name("config.yaml")
            
    config_file = Path(config_path).expanduser()
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
        
    return config