import yaml
from pathlib import Path

def loadYamlConfig(configPath: str) -> dict:
    """Load a YAML configuration file."""
    path = Path(configPath)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {configPath}")
        
    with open(path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
        
    if config is None:
        config = {}
        
    return config
