#!/usr/bin/env python3
"""
敏感配置加密工具

使用 Fernet 对称加密保护 .env 中的敏感字段。

用法:
    # 1. 生成主密钥（一次性，输出到控制台）
    python scripts/encrypt_config.py generate-key

    # 2. 将主密钥设为环境变量（不落盘！）
    export MASTER_KEY=<上一步的输出>
    # Windows: set MASTER_KEY=<上一步的输出>

    # 3. 加密单个值
    python scripts/encrypt_config.py encrypt "sk-your-api-key"

    # 4. 解密（验证用）
    python scripts/encrypt_config.py decrypt "<ciphered_text>"

配置方式:
    # .env 中敏感字段改用加密值：
    DEEPSEEK_API_KEY_ENC=gAAAAABl...  # 加密密文
    MYSQL_PASSWORD_ENC=gAAAAABm...
"""

import argparse
import os
import sys

from cryptography.fernet import Fernet


def get_master_key() -> bytes:
    key = os.environ.get("MASTER_KEY", "")
    if not key:
        print("ERROR: MASTER_KEY environment variable not set.", file=sys.stderr)
        print("Set it with: export MASTER_KEY=<key>", file=sys.stderr)
        sys.exit(1)
    try:
        return key.encode("utf-8")
    except Exception:
        return key


def cmd_generate_key():
    """生成一个新的 Fernet 主密钥。"""
    key = Fernet.generate_key()
    print(key.decode("utf-8"))
    print("\n# 请将此密钥设为环境变量（不落盘）:", file=sys.stderr)
    print(f"#   export MASTER_KEY={key.decode()}", file=sys.stderr)


def cmd_encrypt(plaintext: str):
    """加密一段明文。"""
    key = get_master_key()
    f = Fernet(key)
    token = f.encrypt(plaintext.encode("utf-8"))
    print(token.decode("utf-8"))


def cmd_decrypt(ciphertext: str):
    """解密一段密文。"""
    key = get_master_key()
    f = Fernet(key)
    plain = f.decrypt(ciphertext.encode("utf-8"))
    print(plain.decode("utf-8"))


def main():
    parser = argparse.ArgumentParser(description="敏感配置加密工具")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("generate-key", help="生成 Fernet 主密钥")
    p_enc = sub.add_parser("encrypt", help="加密明文")
    p_enc.add_argument("plaintext", help="要加密的明文")
    p_dec = sub.add_parser("decrypt", help="解密密文")
    p_dec.add_argument("ciphertext", help="要解密的密文")

    args = parser.parse_args()

    if args.command == "generate-key":
        cmd_generate_key()
    elif args.command == "encrypt":
        cmd_encrypt(args.plaintext)
    elif args.command == "decrypt":
        cmd_decrypt(args.ciphertext)


if __name__ == "__main__":
    main()
