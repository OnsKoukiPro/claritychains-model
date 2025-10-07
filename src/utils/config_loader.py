import os
import yaml
from pathlib import Path

def load_config():
    """Load configuration with environment variable substitution"""
    config_path = Path(os.getenv('CONFIG_DIR', './config')) / 'config.yaml'

    with open(config_path, 'r') as f:
        config_str = f.read()

    # Replace environment variables
    config_str = os.path.expandvars(config_str)

    return yaml.safe_load(config_str)