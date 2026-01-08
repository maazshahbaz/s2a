"""
Configuration loader for the audio processing pipeline.
Loads settings from config.json and provides easy access to all configuration parameters.
"""

import json
import os
from typing import Dict, Any


class Config:
    """Singleton configuration loader."""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self, config_path: str = "config.json"):
        """Load configuration from JSON file."""
        # Try to find config.json in multiple locations
        possible_paths = [
            config_path,
            os.path.join(os.path.dirname(__file__), config_path),
            os.path.join(os.getcwd(), config_path)
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    self._config = json.load(f)
                print(f"[Config] Loaded configuration from {path}")
                return
        
        raise FileNotFoundError(f"Could not find config.json in any of: {possible_paths}")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Path to config value using dots (e.g., 'services.transcription.url')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def get_service_config(self, service_name: str) -> Dict[str, Any]:
        """
        Get complete configuration for a service.
        
        Args:
            service_name: Name of the service (transcription, diarization, analysis, speaker_correction)
            
        Returns:
            Dictionary with 'url' and 'model_name' keys
        """
        return self.get(f'services.{service_name}', {})
    
    @property
    def raw_config(self) -> Dict[str, Any]:
        """Get the raw configuration dictionary."""
        return self._config


# Global config instance
config = Config()