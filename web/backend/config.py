"""Application configuration and secure key vault."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Server settings — overridable via env vars."""

    app_name: str = "PaperOrchestra"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]
    data_dir: Path = Path("./data")
    uploads_dir: Path = Path("./data/uploads")
    runs_dir: Path = Path("./data/runs")
    vault_path: Path = Path("./data/.vault.enc")
    vault_key_path: Path = Path("./data/.vault.key")

    model_config = {"env_prefix": "PO_"}


settings = Settings()


# ---------------------------------------------------------------------------
# Encrypted API-key vault
# ---------------------------------------------------------------------------

class KeyVault:
    """Encrypt API keys at rest with Fernet symmetric encryption."""

    def __init__(self, vault_path: Path | None = None, key_path: Path | None = None):
        self._vault_path = vault_path or settings.vault_path
        self._key_path = key_path or settings.vault_key_path
        self._fernet: Fernet | None = None

    # -- lifecycle --

    def _ensure_dirs(self) -> None:
        self._vault_path.parent.mkdir(parents=True, exist_ok=True)

    def _load_or_create_key(self) -> Fernet:
        if self._fernet:
            return self._fernet
        self._ensure_dirs()
        if self._key_path.exists():
            key = self._key_path.read_bytes()
        else:
            key = Fernet.generate_key()
            self._key_path.write_bytes(key)
            self._key_path.chmod(0o600)
        self._fernet = Fernet(key)
        return self._fernet

    def _read_store(self) -> dict[str, str]:
        f = self._load_or_create_key()
        if not self._vault_path.exists():
            return {}
        try:
            return json.loads(f.decrypt(self._vault_path.read_bytes()))
        except Exception:
            return {}

    def _write_store(self, store: dict[str, str]) -> None:
        f = self._load_or_create_key()
        self._ensure_dirs()
        self._vault_path.write_bytes(f.encrypt(json.dumps(store).encode()))
        self._vault_path.chmod(0o600)

    # -- public API --

    def set(self, name: str, value: str) -> None:
        store = self._read_store()
        store[name] = value
        self._write_store(store)

    def get(self, name: str) -> str | None:
        return self._read_store().get(name)

    def delete(self, name: str) -> bool:
        store = self._read_store()
        if name in store:
            del store[name]
            self._write_store(store)
            return True
        return False

    def list_keys(self) -> list[str]:
        return list(self._read_store().keys())

    def get_for_env(self) -> dict[str, str]:
        """Return all keys as a dict suitable for env var injection."""
        return dict(self._read_store())

    def has(self, name: str) -> bool:
        return name in self._read_store()


vault = KeyVault()
