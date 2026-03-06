"""
Minimal config_loader that supports environment variable overrides.
Replace or merge with your existing config_loader as needed.

Priority: ENV vars > config file > defaults
"""
import os
import json
from pathlib import Path


class Config:
    def __init__(self):
        self._config = {}
        self._load_file()

    def _load_file(self):
        """Load from config.json if it exists."""
        config_path = Path(__file__).parent / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                self._config = json.load(f)

    def get_service_config(self, service_name: str) -> dict:
        file_config = self._config.get(service_name, {})

        if service_name == "summary_generator":
            return {
                "url": os.getenv("TRITON_URL", file_config.get("url", "localhost:3701")),
                "model_name": os.getenv("MODEL_NAME", file_config.get("model_name", "mistral-nemo")),
            }

        return file_config


config = Config()