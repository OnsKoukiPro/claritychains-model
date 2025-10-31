import yaml
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def load_config():
    """Load configuration with enhanced error handling"""
    try:
        # Try multiple possible config locations
        possible_paths = [
            Path(__file__).parent.parent.parent / 'config' / 'config.yaml',  # claritychainsv2/config/config.yaml
            Path(__file__).parent.parent / 'config' / 'config.yaml',         # app/config/config.yaml
            Path('config/config.yaml'),                                      # relative to current dir
        ]

        config_path = None
        for path in possible_paths:
            if path.exists():
                config_path = path
                break

        if config_path and config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                logger.info(f"✅ Configuration loaded from {config_path}")
                return config
        else:
            logger.warning("❌ Config file not found in any location, using defaults")

    except Exception as e:
        logger.error(f"❌ Config load error: {e}")

    # Default configuration
    default_config = {
        'paths': {
            'data_dir': './data',
            'raw_data': './data/raw',
            'processed_data': './data/processed'
        },
        'materials': {
            'lithium': {}, 'cobalt': {}, 'nickel': {}, 'copper': {}, 'rare_earths': {},
            'aluminum': {}, 'zinc': {}, 'lead': {}, 'tin': {}
        },
        'forecasting': {
            'use_fundamentals': True,
            'ev_adjustment_weight': 0.3,
            'risk_adjustment_weight': 0.2,
            'rolling_window': 12,
            'forecast_horizon': 6,
            'confidence_levels': [0.1, 0.5, 0.9]
        },
        'global_sources': {
            'ecb_enabled': True,
            'worldbank_enabled': True,
            'lme_enabled': True,
            'global_futures_enabled': True
        }
    }

    logger.info("✅ Using default configuration")
    return default_config