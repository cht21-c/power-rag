"""
API Key 密钥存储模块
"""

import hashlib
import json
import os
import secrets
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Dict


@dataclass
class ApiKeyRecord:
    key_hash: str
    user_id: str
    role: str
    created_at: str
    enabled: bool = True


class KeyStore:
    def __init__(self, file_path: Optional[str] = None):
        if file_path is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            file_path = str(project_root / "framework" / "data" / "api_keys.json")
        self._path = Path(file_path)
        self._records: Dict[str, ApiKeyRecord] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                raw_text = open(self._path, "r", encoding="utf-8").read().strip()
                if not raw_text:
                    self._records = {}
                else:
                    data = json.loads(raw_text)
                    self._records = {
                        k: ApiKeyRecord(**v) for k, v in data.get("keys", {}).items()
                    }
            except (json.JSONDecodeError, TypeError) as e:
                raise RuntimeError(f"Failed to load API key store: {e}")
        else:
            self._records = {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {"keys": {k: asdict(v) for k, v in self._records.items()}}
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def hash_key(raw_key: str) -> str:
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    @staticmethod
    def generate_key() -> str:
        return "sk-" + secrets.token_hex(32)

    def add_key(self, raw_key: str, user_id: str, role: str = "operator",
                enabled: bool = True) -> ApiKeyRecord:
        if role not in ("admin", "operator"):
            raise ValueError(f"Invalid role: {role}")
        key_hash = self.hash_key(raw_key)
        record = ApiKeyRecord(
            key_hash=key_hash, user_id=user_id, role=role,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            enabled=enabled,
        )
        self._records[key_hash] = record
        self._save()
        return record

    def verify(self, raw_key: str) -> Optional[ApiKeyRecord]:
        if not raw_key:
            return None
        key_hash = self.hash_key(raw_key)
        record = self._records.get(key_hash)
        if record and record.enabled:
            return record
        return None

    def check_disabled(self, raw_key: str) -> bool:
        if not raw_key:
            return False
        key_hash = self.hash_key(raw_key)
        record = self._records.get(key_hash)
        return record is not None and not record.enabled

    def disable_key(self, raw_key: str) -> bool:
        key_hash = self.hash_key(raw_key)
        record = self._records.get(key_hash)
        if record:
            record.enabled = False
            self._save()
            return True
        return False

    def enable_key(self, raw_key: str) -> bool:
        key_hash = self.hash_key(raw_key)
        record = self._records.get(key_hash)
        if record:
            record.enabled = True
            self._save()
            return True
        return False

    def list_keys(self) -> List[ApiKeyRecord]:
        return list(self._records.values())

    def init_defaults(self) -> str:
        if self._records:
            return ""
        admin_key = self.generate_key()
        self.add_key(admin_key, "default_admin", role="admin")
        return admin_key
