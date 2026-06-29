# ibkrpy/shared/config_manager.py
# 極簡化的配置讀取器，並兼管全域的核心資料結構 (支援 .env 與 YAML)

import yaml
import os
from typing import Any, Dict, List
from dataclasses import dataclass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    # 精確載入根目錄下的 .env 檔案
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
except ImportError:
    pass

@dataclass
class AssetProfile:
    """定義交易標的的基本屬性 (原 interfaces.py 職責)"""
    symbol: str
    secType: str = "STK"
    exchange: str = "SMART"
    currency: str = "USD"
    term: str = "long_term"
    tags: List[str] = None

class ConfigManager:
    """輕量級配置管理器，負責載入 YAML、環境變數與分發參數"""
    
    def __init__(self, config_path: str = None):
        # 若未指定，則強制使用根目錄的 config.yaml
        self.config_path = config_path or os.path.join(PROJECT_ROOT, "config.yaml")
        self._config_data: Dict[str, Any] = {}
        self.asset_profiles: List[AssetProfile] = []
        self.reload()

    def reload(self):
        """重新載入 YAML 配置檔案"""
        if not os.path.exists(self.config_path):
            print(f"警告: 找不到配置檔 {self.config_path}，將使用預設空配置。")
            return

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                self._config_data = yaml.safe_load(f) or {}
                
            self._parse_asset_profiles()
        except Exception as e:
            print(f"解析配置檔失敗: {e}")

    def _parse_asset_profiles(self):
        """將 YAML 中的 dict 轉換為 AssetProfile 資料類別"""
        raw_assets = self._config_data.get("assets", [])
        self.asset_profiles = []
        for asset in raw_assets:
            if isinstance(asset, dict) and "symbol" in asset:
                if "tags" in asset and isinstance(asset["tags"], str):
                    asset["tags"] = [t.strip() for t in asset["tags"].split(",") if t.strip()]
                self.asset_profiles.append(AssetProfile(**asset))

    def get(self, dot_path: str, default: Any = None) -> Any:
        """
        支援點綴語法獲取巢狀字典的值，例如: get("ib_settings.port")
        優先級別為 1. 環境變數 (.env) -> 2. config.yaml -> 3. default
        """
        env_key = dot_path.split('.')[-1].upper()
        if env_key in os.environ and os.environ[env_key].strip() != "":
            val = os.environ[env_key].strip()
            if val.isdigit(): return int(val)
            if val.lower() == 'true': return True
            if val.lower() == 'false': return False
            return val

        keys = dot_path.split('.')
        value = self._config_data
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
                
        return value if value != "" else default