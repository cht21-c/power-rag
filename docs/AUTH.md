# 认证鉴权配置指南

## 概述

等保三级要求"身份鉴别"和"访问控制"。本模块提供基于 API Key 的轻量级认证方案，支持两级 RBAC。

## 角色定义

| 角色 | 权限 |
|------|------|
| `admin` | 全部权限：文档摄取（ingest）、清空集合、问答 |
| `operator` | 只读问答权限 |

## 快速开始

### 1. 安装依赖（已有）

框架层无需额外依赖，Python 标准库即可运行。

### 2. 生成 API Key

```bash
cd camera_sdk_agent
python -c "
from framework.auth.key_store import KeyStore
store = KeyStore()
admin_key = store.init_defaults()
if admin_key:
    print(f'Default admin key: {admin_key}')
    print('SAVE THIS KEY — it will not be shown again.')
else:
    store2 = KeyStore()
    new_key = store2.generate_key()
    store2.add_key(new_key, 'custom_user', role='operator')
    print(f'New key: {new_key}')
"
```

### 3. 添加更多用户

```python
from framework.auth.key_store import KeyStore

store = KeyStore()

# 新增 operator 用户
key = store.generate_key()
store.add_key(key, "operator01", role="operator")
print(f"Key for operator01: {key}")

# 新增 admin 用户
key2 = store.generate_key()
store.add_key(key2, "admin02", role="admin")
print(f"Key for admin02: {key2}")
```

### 4. 查看已有 Key（不显示明文）

```python
from framework.auth.key_store import KeyStore
store = KeyStore()
for record in store.list_keys():
    print(f"User: {record.user_id:15s}  Role: {record.role:10s}  "
          f"Enabled: {record.enabled}  Created: {record.created_at}")
```

### 5. 使用 Key

**CLI 模式：**

```bash
# 设置环境变量后运行
export API_KEY=sk-your-admin-key
python main.py --ingest  # admin 权限操作
python main.py            # 任何已认证用户都能问答
```

**Chainlit Web UI 模式：**

```bash
export API_KEY=sk-your-operator-key
chainlit run chat_ui.py
```

**开发模式（无 Key）：**

当 KeyStore 为空且未设置 API_KEY 环境变量时，系统自动以 `guest:operator` 身份运行，无需认证。

### 6. 禁用/启用 Key

```python
from framework.auth.key_store import KeyStore
store = KeyStore()
store.disable_key("sk-old-key-to-disable")
store.enable_key("sk-key-to-re-enable")
```

## 密钥存储

- Key 文件和路径：`framework/data/api_keys.json`
- 存储格式：JSON，Key 仅存 SHA-256 哈希，不存明文
- 文件权限：生产环境建议设 600（仅所有者可读写）

## 架构说明

```
framework/auth/
├── __init__.py       # 模块入口
├── key_store.py      # API Key CRUD + 哈希存储
└── middleware.py     # @require_auth 装饰器 + authenticate_session
```

## 认证流程

```
请求进入 → 读取 API_KEY 环境变量
  ├─ 未设置 & KeyStore 为空 → guest 模式
  ├─ 未设置 & KeyStore 有数据 → 拒绝 (401)
  ├─ 有效 Key → 校验成功，绑定 user_id + role
  └─ 无效/禁用 Key → 拒绝 (403)
```

## 等保合规对照

| 等保要求 | 实现方式 |
|---------|---------|
| 身份鉴别 | API Key 校验 |
| 访问控制 | admin / operator 两级 RBAC |
| 密钥安全 | SHA-256 哈希存储，不存明文 |
| 审计关联 | user_id 贯穿所有审计日志 |
