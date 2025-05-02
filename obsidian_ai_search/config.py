import tomllib # Requires Python 3.11+
# For older Python, use: import toml
import os
from pathlib import Path
import logging
from typing import Dict, Any
from dotenv import load_dotenv # Import load_dotenv

logger = logging.getLogger(__name__)

# --- Load .env file --- 
# Looks for .env in the current working directory or parent directories
# Loads variables into os.environ
dotenv_path = Path('.env')
if dotenv_path.exists():
    logger.info(f"Loading environment variables from: {dotenv_path.resolve()}")
    load_dotenv(dotenv_path=dotenv_path, override=True) # Override existing env vars if conflicts
else:
    logger.info("No .env file found, skipping environment variable loading from file.")
# --- End .env loading ---

# Default configuration values
DEFAULT_CONFIG = {
    "index_dir": ".obsidian_ai_search_index",
    "default_top_k": 5,
    "embedding_model": "all-MiniLM-L6-v2", # Default SentenceTransformer model
    # Add other potential config options here
}

# Potential config file locations (user config dir preferred)
CONFIG_DIR = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "obsidian-ai-search"
CONFIG_FILE = CONFIG_DIR / "config.toml"

def load_config() -> Dict[str, Any]:
    """Loads configuration from TOML file, merging with defaults."""
    config = DEFAULT_CONFIG.copy()
    
    if CONFIG_FILE.exists():
        logger.info(f"Loading configuration from: {CONFIG_FILE}")
        try:
            with open(CONFIG_FILE, "rb") as f:
                user_config = tomllib.load(f) # Use tomllib.load for Python 3.11+
                # For older Python + toml package: user_config = toml.load(f)
                
            # Merge user config over defaults (only update keys present in default)
            for key in DEFAULT_CONFIG:
                if key in user_config:
                    # TODO: Add type validation here if needed
                    config[key] = user_config[key]
                    logger.debug(f"Config: Loaded '{key}' = {config[key]} from file.")
                else:
                     logger.debug(f"Config: Using default for '{key}' = {config[key]}.")
            logger.info("Configuration loaded successfully.")
        except tomllib.TOMLDecodeError as e:
            logger.error(f"Error decoding TOML config file {CONFIG_FILE}: {e}. Using default settings.")
        except Exception as e:
            logger.error(f"Error reading config file {CONFIG_FILE}: {e}. Using default settings.")
    else:
        logger.info(f"Config file not found at {CONFIG_FILE}. Using default settings.")
        # Optionally, create a default config file on first run?
        # try:
        #     CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        #     with open(CONFIG_FILE, 'w') as f:
        #         # import toml # if using toml package
        #         # toml.dump(DEFAULT_CONFIG, f)
        #         # For tomllib, need manual writing or another lib
        #         logger.info(f"Created default config file at {CONFIG_FILE}")
        # except Exception as create_e:
        #     logger.error(f"Could not create default config file: {create_e}")
            
    return config

# Load config once when module is imported
APP_CONFIG = load_config() 