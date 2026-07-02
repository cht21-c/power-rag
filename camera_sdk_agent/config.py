"""
Camera SDK RAG Agent - Configuration Module (等保三级改造)

变更：
- 新增 _decrypt_env() 辅助函数，支持 Fernet 加密字段
- 敏感字段（DEEPSEEK_API_KEY、MYSQL_PASSWORD）优先读取 _ENC 后缀加密值
- 解密使用 MASTER_KEY 环境变量（不落盘）
- print_config() 和 validate_config() 行为不变
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Encryption support (等保三级 - 敏感配置加密存储)
# ---------------------------------------------------------------------------

_fernet_cache = None


def _get_fernet():
    """延迟初始化 Fernet 实例（仅在需要解密时加载）。"""
    global _fernet_cache
    if _fernet_cache is not None:
        return _fernet_cache
    master_key = os.getenv("MASTER_KEY", "")
    if not master_key:
        # 未设置 MASTER_KEY 时返回 None，调用方回退到明文
        return None
    try:
        from cryptography.fernet import Fernet
        _fernet_cache = Fernet(master_key.encode("utf-8"))
        return _fernet_cache
    except ImportError:
        return None
    except Exception:
        return None


def _decrypt_env(env_var: str, default: str = "") -> str:
    """读取环境变量，自动处理加密值。

    优先级：
    1. 直接读环境变量（非加密明文）
    2. 如果为空，读 {env_var}_ENC 后缀的加密值，尝试解密
    3. 回退到 default

    Fernet 加密值的特征是 base64 格式（gAAAAAB 前缀）。
    """
    raw = os.getenv(env_var, "")
    if raw:
        return raw

    encrypted = os.getenv(f"{env_var}_ENC", "")
    if encrypted:
        f = _get_fernet()
        if f:
            try:
                return f.decrypt(encrypted.encode("utf-8")).decode("utf-8")
            except Exception:
                import warnings
                warnings.warn(f"Failed to decrypt {env_var}_ENC. Check MASTER_KEY.", RuntimeWarning)
                return ""
        else:
            import warnings
            warnings.warn(
                f"{env_var}_ENC is set but MASTER_KEY is not. "
                f"Install cryptography: pip install cryptography",
                RuntimeWarning,
            )
            return ""

    return default


# ---------------------------------------------------------------------------
# .env loading
# ---------------------------------------------------------------------------

try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Qdrant
# ---------------------------------------------------------------------------
QDRANT_HOST = os.getenv("QDRANT_HOST", "")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_LOCAL_PATH = os.getenv("QDRANT_LOCAL_PATH", str(Path(__file__).resolve().parent / "qdrant_data"))
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "camera_sdk_collection")
QDRANT_VECTOR_DIM = int(os.getenv("QDRANT_VECTOR_DIM", "1024"))
QDRANT_DISTANCE_METRIC = os.getenv("QDRANT_DISTANCE_METRIC", "Cosine")

# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-m3")
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "cpu")
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE_TOKENS = int(os.getenv("CHUNK_SIZE_TOKENS", "512"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "64"))

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", "20"))
BM25_TOP_K = int(os.getenv("BM25_TOP_K", "20"))
FINAL_TOP_K = int(os.getenv("FINAL_TOP_K", "5"))
FUSION_K = int(os.getenv("FUSION_K", "60"))

# ---------------------------------------------------------------------------
# P1 质量加固 —— 可配置阈值
# ---------------------------------------------------------------------------
# 幻觉护栏: RRF 融合分数阈值
# 最高分低于此值 → 直接拒答（不调用 LLM）
RAG_MIN_SCORE_STRICT = float(os.getenv("RAG_MIN_SCORE_STRICT", "0.02"))
# 最高分在此值和 STRICT 之间 → 生成但标注"置信度较低"
RAG_MIN_SCORE_WARN = float(os.getenv("RAG_MIN_SCORE_WARN", "0.05"))

# 意图置信度阈值: LLM 分类 confidence 低于此值 → 反问用户澄清
INTENT_CONFIDENCE_THRESHOLD = float(os.getenv("INTENT_CONFIDENCE_THRESHOLD", "0.55"))

# 图纸语义匹配: 图纸列表超过此数量时分批传入 LLM
DRAWING_MATCH_MAX_ITEMS = int(os.getenv("DRAWING_MATCH_MAX_ITEMS", "500"))

# 二次验证开关: drawing 路由高置信度时是否启用独立复核
VERIFY_ENABLED = os.getenv("VERIFY_ENABLED", "true").lower() in ("1", "true", "yes")

# ---------------------------------------------------------------------------
# P2 工程健壮性 —— 可配置参数
# ---------------------------------------------------------------------------
# 模型路由: 候选模型列表（优先级从高到低）
MODEL_CANDIDATES = os.getenv("MODEL_CANDIDATES", "deepseek-chat").split(",")

# 熔断器: 连续失败次数触发熔断
CIRCUIT_BREAKER_FAIL_THRESHOLD = int(os.getenv("CIRCUIT_BREAKER_FAIL_THRESHOLD", "5"))

# 熔断器: 冷却时间（秒），OPEN -> HALF_OPEN
CIRCUIT_BREAKER_COOLDOWN_SEC = int(os.getenv("CIRCUIT_BREAKER_COOLDOWN_SEC", "60"))

# 首包超时（秒），流式调用超过此时间未收到 token 视为失败
STREAM_FIRST_TOKEN_TIMEOUT_SEC = float(os.getenv("STREAM_FIRST_TOKEN_TIMEOUT_SEC", "10"))

# 记忆压缩: 保留最近 N 轮原文不压缩
MEMORY_WINDOW_TURNS = int(os.getenv("MEMORY_WINDOW_TURNS", "3"))

# 记忆压缩: 总消息超过此 token 估算值触发压缩
MEMORY_COMPRESS_TRIGGER_TOKENS = int(os.getenv("MEMORY_COMPRESS_TRIGGER_TOKENS", "3500"))

# 工具注册: 启用的工具名单（逗号分隔，空=全部启用）
ENABLED_TOOLS = os.getenv("ENABLED_TOOLS", "").split(",") if os.getenv("ENABLED_TOOLS") else None

# ---------------------------------------------------------------------------
# P2 结构化内容检测 —— 规则分块配置
# ---------------------------------------------------------------------------
# 结构化检测: 管道分隔字段数（如漏洞报告 4 字段: 名称|等级|IP|建议）
STRUCTURED_ROW_FIELD_COUNT = int(os.getenv("STRUCTURED_ROW_FIELD_COUNT", "4"))

# 结构化检测: 命中结构化模式的行数占比阈值（0.0-1.0，达到才判定为结构化块）
STRUCTURED_BLOCK_THRESHOLD = float(os.getenv("STRUCTURED_BLOCK_THRESHOLD", "0.6"))

# 规则分块: 规则分块时每个 chunk 最多容纳的记录行数
MAX_RECORDS_PER_CHUNK = int(os.getenv("MAX_RECORDS_PER_CHUNK", "8"))

# LLM 上下文: 消息+检索内容的最大 token 预算
LLM_MAX_CONTEXT_TOKENS = int(os.getenv("LLM_MAX_CONTEXT_TOKENS", "8000"))

# ---------------------------------------------------------------------------
# DeepSeek API Configuration (等保三级：支持加密存储)
# ---------------------------------------------------------------------------
DEEPSEEK_API_KEY = _decrypt_env("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# ---------------------------------------------------------------------------
# MySQL (等保三级：密码支持加密存储)
# ---------------------------------------------------------------------------
_MYSQL_PASSWORD = _decrypt_env("MYSQL_PASSWORD", "")

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
SDK_DOCS_DIR = PROJECT_ROOT / "sdk_docs"


# ---------------------------------------------------------------------------
# Validation / Printing
# ---------------------------------------------------------------------------

def validate_config() -> bool:
    if not DEEPSEEK_API_KEY:
        raise ValueError(
            "DEEPSEEK_API_KEY is not set. "
            "Set it via environment variable or in a .env file.\n"
            "For encrypted storage, set DEEPSEEK_API_KEY_ENC and MASTER_KEY."
        )
    return True


def print_config():
    key_masked = (
        DEEPSEEK_API_KEY[:8] + "..." + DEEPSEEK_API_KEY[-4:]
        if DEEPSEEK_API_KEY else "NOT SET"
    )
    print(f"Qdrant:          {QDRANT_HOST}:{QDRANT_PORT}")
    print(f"Collection:      {QDRANT_COLLECTION_NAME}")
    print(f"Embed Model:     {EMBED_MODEL_NAME}")
    print(f"Embed Device:    {EMBED_DEVICE}")
    print(f"Chunk Size:      {CHUNK_SIZE_TOKENS} tokens (overlap {CHUNK_OVERLAP_TOKENS})")
    print(f"Retrieval:       vector_top{VECTOR_TOP_K} + bm25_top{BM25_TOP_K} -> fusion_top{FINAL_TOP_K}")
    print(f"DeepSeek Model:  {DEEPSEEK_MODEL}")
    print(f"API Key:         {key_masked}")
    print(f"Docs Directory:  {SDK_DOCS_DIR}")
