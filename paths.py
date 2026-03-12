from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

CONFIG_DIR = ROOT_DIR / "config"
APP_CONFIG_FPATH = CONFIG_DIR / "config.yaml"
PROMPT_CONFIG_FPATH = CONFIG_DIR / "prompt_config.yaml"

DATA_DIR = ROOT_DIR / "data"
RAWFILES_DIR = DATA_DIR / "raw_files"

LOGGING_DIR = ROOT_DIR / "logging"
LOGS_FPATH = LOGGING_DIR / "app.logs"

SRC_DIR = ROOT_DIR / "src"

ENV_FPATH = ROOT_DIR / ".env"

DB_DIR = ROOT_DIR / "DB"

print(ROOT_DIR)