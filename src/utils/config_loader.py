"""
Configuration management with YAML files
"""

import yaml
from typing import Dict, Any
import os

class Config:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.data = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not os.path.exists(self.config_path):
            self._create_default_config()
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _create_default_config(self):
        """Create default configuration file"""
        default_config = {
            'model': {
                'image_shape': [64, 64, 1],
                'latent_dim': 100,
                'learning_rate': 0.0001,
                'batch_size': 32
            },
            'detection': {
                'threshold': 0.1,
                'min_confidence': 0.7,
                'max_batch_size': 16
            },
            'api': {
                'host': '0.0.0.0',
                'port': 8000,
                'debug': True,
                'log_level': 'INFO'
            },
            'database': {
                'path': 'anomalies.db',
                'echo': False
            }
        }
        
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key"""
        keys = key.split('.')
        value = self.data
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

# Global configuration instance
config = Config()