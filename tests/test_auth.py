"""
认证鉴权模块单元测试

覆盖：
- Key 生成、哈希存储、校验
- Key 启/禁用
- 角色校验（admin vs operator）
- require_auth 装饰器
- 开发模式 guest fallback
"""

import os
import json
import tempfile
import pytest

# 确保框架模块可导入
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from framework.auth.key_store import KeyStore, ApiKeyRecord
from framework.auth.middleware import (
    require_auth, authenticate_session, get_current_user,
    set_current_user, clear_current_user, AuthError, get_key_store, reset_key_store
)


class TestKeyStore:
    """KeyStore 的 CRUD 操作测试。"""

    def test_generate_and_verify(self):
        store = _temp_store()
        raw = store.generate_key()
        assert raw.startswith("sk-")
        assert len(raw) == 67  # "sk-" + 64 hex
        assert store.verify(raw) is None  # 尚未注册

    def test_add_and_verify(self):
        store = _temp_store()
        raw = store.generate_key()
        record = store.add_key(raw, "test_user", role="operator")
        assert record.user_id == "test_user"
        assert record.role == "operator"
        assert record.enabled is True

        verified = store.verify(raw)
        assert verified is not None
        assert verified.user_id == "test_user"

    def test_verify_wrong_key(self):
        store = _temp_store()
        store.add_key(store.generate_key(), "alice")
        assert store.verify("wrong-key") is None
        assert store.verify("") is None

    def test_disable_enable(self):
        store = _temp_store()
        raw = store.generate_key()
        store.add_key(raw, "bob")
        assert store.verify(raw) is not None

        store.disable_key(raw)
        assert store.verify(raw) is None  # 禁用后校验失败

        store.enable_key(raw)
        assert store.verify(raw) is not None

    def test_hash_not_stored_plaintext(self):
        store = _temp_store()
        raw = store.generate_key()
        store.add_key(raw, "test")
        store._load()  # 重新加载
        # 确认存储的是哈希不是明文
        for record in store.list_keys():
            assert raw not in record.key_hash

    def test_duplicate_key_overwrites(self):
        store = _temp_store()
        raw = store.generate_key()
        store.add_key(raw, "first", role="operator")
        store.add_key(raw, "second", role="admin")  # 同 key 覆盖
        record = store.verify(raw)
        assert record.user_id == "second"
        assert record.role == "admin"

    def test_init_defaults(self):
        store = _temp_store()
        key = store.init_defaults()
        assert key.startswith("sk-")
        assert store.verify(key) is not None
        # 再次调用不覆盖
        key2 = store.init_defaults()
        assert key2 == ""

    def test_invalid_role_raises(self):
        store = _temp_store()
        with pytest.raises(ValueError):
            store.add_key(store.generate_key(), "test", role="superadmin")


class TestMiddleware:
    """require_auth 认证中间件测试。"""

    def setup_method(self):
        clear_current_user()
        reset_key_store()
        # 清理环境变量
        for key in ("API_KEY", "API_KEY_STORE_PATH"):
            os.environ.pop(key, None)

    def test_guest_mode_no_keys(self):
        @require_auth()
        def test_func():
            user = get_current_user()
            return user

        result = test_func()
        assert result == {"user_id": "guest", "role": "operator"}

    def test_auth_with_valid_key(self):
        store = _temp_store()
        raw = store.generate_key()
        store.add_key(raw, "admin01", role="admin")

        os.environ["API_KEY"] = raw
        os.environ["API_KEY_STORE_PATH"] = str(store._path)

        @require_auth(role="admin")
        def test_func():
            return get_current_user()

        result = test_func()
        assert result["user_id"] == "admin01"
        assert result["role"] == "admin"

    def test_invalid_key_raises(self):
        store = _temp_store()
        store.add_key(store.generate_key(), "user01")
        os.environ["API_KEY"] = "sk-invalid-key-000000000000000000000000000000"
        os.environ["API_KEY_STORE_PATH"] = str(store._path)

        @require_auth()
        def test_func():
            pass

        with pytest.raises(AuthError) as exc:
            test_func()
        assert exc.value.error_code == "INVALID_KEY"

    def test_insufficient_role(self):
        store = _temp_store()
        raw = store.generate_key()
        store.add_key(raw, "operator01", role="operator")

        os.environ["API_KEY"] = raw
        os.environ["API_KEY_STORE_PATH"] = str(store._path)

        @require_auth(role="admin")
        def test_func():
            pass

        with pytest.raises(AuthError) as exc:
            test_func()
        assert exc.value.error_code == "INSUFFICIENT_ROLE"

    def test_admin_can_do_operator(self):
        store = _temp_store()
        raw = store.generate_key()
        store.add_key(raw, "admin01", role="admin")

        os.environ["API_KEY"] = raw
        os.environ["API_KEY_STORE_PATH"] = str(store._path)

        @require_auth(role="operator")
        def test_func():
            return get_current_user()

        result = test_func()
        assert result["user_id"] == "admin01"

    def test_disabled_key_raises(self):
        store = _temp_store()
        raw = store.generate_key()
        store.add_key(raw, "user01")
        store.disable_key(raw)

        os.environ["API_KEY"] = raw
        os.environ["API_KEY_STORE_PATH"] = str(store._path)

        @require_auth()
        def test_func():
            pass

        with pytest.raises(AuthError) as exc:
            test_func()
        assert exc.value.error_code == "KEY_DISABLED"

    def test_authenticate_session(self):
        store = _temp_store()
        raw = store.generate_key()
        store.add_key(raw, "testuser", role="operator")

        os.environ["API_KEY_STORE_PATH"] = str(store._path)

        result = authenticate_session(raw)
        assert result["user_id"] == "testuser"
        assert result["role"] == "operator"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _temp_store():
    """创建临时 KeyStore 实例。"""
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    tmp.close()
    store = KeyStore(tmp.name)
    # 清理引用以便删除
    import atexit
    atexit.register(lambda: os.unlink(tmp.name) if os.path.exists(tmp.name) else None)
    return store
