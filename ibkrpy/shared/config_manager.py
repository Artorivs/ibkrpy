# ibkrpy/shared/config_manager.py
# Minimal config reader (.env + YAML) and the shared AssetProfile type

import yaml
import os
from typing import Any, Dict, List
from dataclasses import dataclass

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
except ImportError:
    pass


@dataclass
class AssetProfile:
    """Basic properties of a tradable asset."""

    symbol: str
    secType: str = "STK"
    exchange: str = "SMART"
    currency: str = "USD"
    tags: List[str] = None


class ConfigManager:
    """Loads YAML and environment variables and serves config values."""

    def __init__(self, config_path: str = None):
        self.config_path = config_path or os.path.join(PROJECT_ROOT, "config.yaml")
        self._config_data: Dict[str, Any] = {}
        self.asset_profiles: List[AssetProfile] = []
        self.reload()

    def reload(self):
        """Reload the YAML config file."""
        if not os.path.exists(self.config_path):
            print(
                f"Warning: config file {self.config_path} not found; using an empty config."
            )
            return

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                self._config_data = yaml.safe_load(f) or {}

            self._parse_asset_profiles()
        except Exception as e:
            print(f"Failed to parse config file: {e}")

    def _parse_asset_profiles(self):
        """Convert the YAML asset dicts into AssetProfile objects."""
        raw_assets = self._config_data.get("assets", [])
        self.asset_profiles = []
        for asset in raw_assets:
            if isinstance(asset, dict) and "symbol" in asset:
                if "tags" in asset and isinstance(asset["tags"], str):
                    asset["tags"] = [
                        t.strip() for t in asset["tags"].split(",") if t.strip()
                    ]
                self.asset_profiles.append(AssetProfile(**asset))

    def get(self, dot_path: str, default: Any = None) -> Any:
        """
        Read a nested value with dot syntax, e.g. get("ib_settings.port").
        Precedence: 1. environment variable (.env) -> 2. config.yaml -> 3. default
        """
        env_key = dot_path.split(".")[-1].upper()
        if env_key in os.environ and os.environ[env_key].strip() != "":
            val = os.environ[env_key].strip()
            if val.isdigit():
                return int(val)
            if val.lower() == "true":
                return True
            if val.lower() == "false":
                return False
            return val

        keys = dot_path.split(".")
        value = self._config_data
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value if value != "" else default
