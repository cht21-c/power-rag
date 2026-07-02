"""
认证中间件 - @require_auth 装饰器

用于 CLI 管理命令和 Chainlit Web UI 入口的轻量级认证包装。
"""

import os
import functools
from typing import Callable, Optional

from .key_store import KeyStore

_KEY_STORE_PATH = os.environ.get("API_KEY_STORE_PATH", "")

_store: Optional[KeyStore] = None
_current_user: Optional[dict] = None


def get_key_store() -> KeyStore:
    global _store
    _store_path = os.environ.get("API_KEY_STORE_PATH", "")
    if _store is None:
        _store = KeyStore(_store_path if _store_path else None)
    return _store


def set_current_user(user_id: str, role: str) -> None:
    global _current_user
    _current_user = {"user_id": user_id, "role": role}


def get_current_user() -> Optional[dict]:
    return _current_user


def clear_current_user() -> None:
    global _current_user
    _current_user = None


class AuthError(Exception):
    def __init__(self, message: str, error_code: str = "AUTH_FAILED"):
        super().__init__(message)
        self.error_code = error_code


def require_auth(role: Optional[str] = None):
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            store = get_key_store()
            api_key = os.environ.get("API_KEY", "").strip()

            if not api_key:
                if not store.list_keys():
                    set_current_user("guest", "operator")
                    result = func(*args, **kwargs)
                    clear_current_user()
                    return result
                raise AuthError("API_KEY is required. Set API_KEY env var.",
                                error_code="NO_API_KEY")

            record = store.verify(api_key)
            if record is None:
                if store.check_disabled(api_key):
                    raise AuthError("API Key disabled.", error_code="KEY_DISABLED")
                raise AuthError("Invalid API Key.", error_code="INVALID_KEY")

            if role is not None and record.role != role:
                if not (role == "operator" and record.role == "admin"):
                    raise AuthError(
                        f"Access denied. Required: {role}, current: {record.role}.",
                        error_code="INSUFFICIENT_ROLE")

            set_current_user(record.user_id, record.role)
            try:
                return func(*args, **kwargs)
            finally:
                clear_current_user()
        return wrapper
    return decorator


def authenticate_session(api_key: str) -> dict:
    store = get_key_store()
    if not api_key:
        if not store.list_keys():
            return {"user_id": "guest", "role": "operator"}
        raise AuthError("API Key is required.", error_code="NO_API_KEY")
    record = store.verify(api_key)
    if record is None:
        raise AuthError("Invalid API Key.", error_code="INVALID_KEY")
    if not record.enabled:
        raise AuthError("API Key disabled.", error_code="KEY_DISABLED")
    return {"user_id": record.user_id, "role": record.role}

def reset_key_store() -> None:
    """重置全局 KeyStore 单例（测试用）。"""
    global _store
    _store = None
