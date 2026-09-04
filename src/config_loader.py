"""
config_loader.py
----------------
Loads configuration from config.yaml once and caches it.
"""
import yaml
import os
from pathlib import Path

_CONFIG_CACHE = None

def get_config():
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE
        
    config_path = Path(__file__).parent.parent / "config.yaml"
    if not config_path.exists():
        print(f"Warning: config.yaml not found at {config_path}. Using hardcoded defaults.")
        _CONFIG_CACHE = {
            "preprocessing": {"max_size": 1024, "clahe_clip_limit": 2.0},
            "matching": {"ratio_threshold": 0.75, "grid_size": 8, "max_per_cell": 5},
            "ransac": {"reproj_threshold": 3.0},
            "refinement": {"max_shift_pixels": 5}
        }
        return _CONFIG_CACHE

    with open(config_path, "r", encoding="utf-8") as f:
        _CONFIG_CACHE = yaml.safe_load(f)
        
    return _CONFIG_CACHE
