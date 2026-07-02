# 部署与安全加固指南

## HTTPS 配置（等保三级要求）

### 方案：Nginx 反向代理 + TLS

#### 1. 生成自签证书（开发/测试环境）

```bash
mkdir -p /etc/nginx/ssl
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout /etc/nginx/ssl/selfsigned.key \
  -out /etc/nginx/ssl/selfsigned.crt \
  -subj "/C=CN/ST=Jiangxi/L=Ganzhou/O=PowerPlant/CN=localhost"
```

> **生产环境**：替换为正式 CA 签发的证书（如 Let's Encrypt 或企业内部 CA）。

#### 2. Nginx 配置

创建 `/etc/nginx/conf.d/camera-sdk-agent.conf`：

```nginx
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;
    charset utf-8;

    ssl_certificate     /etc/nginx/ssl/selfsigned.crt;
    ssl_certificate_key /etc/nginx/ssl/selfsigned.key;
    ssl_protocols       TLSv1.2 TLSv1.3;
    ssl_ciphers         HIGH:!aNULL:!MD5;

    # 等保：禁止不安全的 HTTP 方法
    if ($request_method !~ ^(GET|POST|HEAD)$) {
        return 405;
    }

    # 等保：隐藏 Nginx 版本号
    server_tokens off;

    # Chainlit Web UI 反向代理
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300s;
    }

    # 图纸静态文件服务
    location /drawings/ {
        alias /usr/share/nginx/html/drawings/;
        autoindex on;
        charset utf-8;
    }
}
```

#### 3. 重启 Nginx

```bash
nginx -t && nginx -s reload
```

## 敏感配置加密

### 1. 安装依赖

```bash
pip install cryptography
```

### 2. 生成主密钥

```bash
export MASTER_KEY=$(python scripts/encrypt_config.py generate-key)
# 保存输出。主密钥必须通过环境变量注入，不落盘！
```

### 3. 加密敏感字段

```bash
# 加密 API Key
python scripts/encrypt_config.py encrypt "sk-your-deepseek-api-key"

# 加密数据库密码
python scripts/encrypt_config.py encrypt "your-db-password"
```

### 4. 配置 .env

将加密后的密文写入 `.env`（使用 _ENC 后缀）：

```bash
# .env
DEEPSEEK_API_KEY_ENC=gAAAAABl...   # 上面加密的输出
MYSQL_PASSWORD_ENC=gAAAAABm...      # 上面加密的输出
```

### 5. 启动时注入主密钥

```bash
export MASTER_KEY=<your-master-key>
python main.py
# 或
chainlit run chat_ui.py
```

### 6. 验证

```bash
# 确认 .env 中没有明文 Key
grep -i "sk-" .env && echo "WARNING: plaintext key found!" || echo "OK: no plaintext key"
```

## 等保三级部署清单

| 检查项 | 要求 | 状态 |
|--------|------|------|
| 传输加密 | HTTPS/TLS 1.2+ | 见上方 Nginx 配置 |
| 身份鉴别 | API Key 认证 | 见 docs/AUTH.md |
| 访问控制 | admin/operator RBAC | 见 docs/AUTH.md |
| 审计追踪 | 全链路日志 | 见 docs/AUDIT.md |
| 敏感信息保护 | Fernet 加密存储 | 见上方加密配置 |
| 密钥管理 | 主密钥不落盘（环境变量注入） | MASTER_KEY |
| Nginx 加固 | 隐藏版本号、限制 HTTP 方法 | 见上方配置 |

## 防火墙建议

```bash
# 仅开放必要端口
ufw allow 443/tcp   # HTTPS
ufw allow 22/tcp    # SSH（管理用）
ufw deny 8000/tcp   # Chainlit 直连端口（应通过 Nginx 代理）
ufw deny 3306/tcp   # MySQL（应仅本地访问）
ufw enable
```
