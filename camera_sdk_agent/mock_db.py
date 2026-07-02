"""
Drawing database backed by MySQL (Docker).

Provides:
- drawings table with power plant engineering drawing records.
- query_drawing_by_name(name): fuzzy match drawing_name.
- query_all_drawings(): return all records.
"""

import os
import time

from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Session


# ---------------------------------------------------------------------------
# Database setup — reads config from environment / .env
# ---------------------------------------------------------------------------

MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = os.getenv("MYSQL_PORT", "3307")
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "plant2024")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "powerplant")

ENGINE_URL = (
    f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}"
    f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}"
    "?charset=utf8mb4"
)


class Base(DeclarativeBase):
    pass


class Drawing(Base):
    __tablename__ = "drawings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    drawing_name = Column(String(255), nullable=False)
    file_name = Column(String(255), nullable=False)
    url = Column(String(512), nullable=False)
    category = Column(String(50), default="其他")
    equipment_id = Column(String(100), default="")


# ---------------------------------------------------------------------------
# Seed data
# ---------------------------------------------------------------------------

_SEED_DATA = [
    {
        "drawing_name": "5号机组循环水泵出口管路图",
        "file_name": "5号机组循环水泵出口管路图.pdf",
        "url": "http://localhost:8080/drawings/5号机组循环水泵出口管路图.pdf",
        "category": "管路",
        "equipment_id": "5号机组",
    },
    {
        "drawing_name": "锅炉给水泵系统图",
        "file_name": "锅炉给水泵系统图.pdf",
        "url": "http://localhost:8080/drawings/锅炉给水泵系统图.pdf",
        "category": "管路",
        "equipment_id": "锅炉",
    },
    {
        "drawing_name": "汽轮机本体结构图",
        "file_name": "汽轮机本体结构图.pdf",
        "url": "http://localhost:8080/drawings/汽轮机本体结构图.pdf",
        "category": "设备",
        "equipment_id": "汽轮机",
    },
    {
        "drawing_name": "凝汽器管路布置图",
        "file_name": "凝汽器管路布置图.pdf",
        "url": "http://localhost:8080/drawings/凝汽器管路布置图.pdf",
        "category": "管路",
        "equipment_id": "凝汽器",
    },
    {
        "drawing_name": "电气主接线图",
        "file_name": "电气主接线图.pdf",
        "url": "http://localhost:8080/drawings/电气主接线图.pdf",
        "category": "电气",
        "equipment_id": "",
    },
    {
        "drawing_name": "6号机组风机系统图",
        "file_name": "6号机组风机系统图.pdf",
        "url": "http://localhost:8080/drawings/6号机组风机系统图.pdf",
        "category": "设备",
        "equipment_id": "6号机组",
    },
    {
        "drawing_name": "变压器接线图",
        "file_name": "变压器接线图.pdf",
        "url": "http://localhost:8080/drawings/变压器接线图.pdf",
        "category": "电气",
        "equipment_id": "",
    },
    {
        "drawing_name": "蒸汽管道平面图",
        "file_name": "蒸汽管道平面图.pdf",
        "url": "http://localhost:8080/drawings/蒸汽管道平面图.pdf",
        "category": "管路",
        "equipment_id": "",
    },
    {
        "drawing_name": "冷却水系统图",
        "file_name": "冷却水系统图.pdf",
        "url": "http://localhost:8080/drawings/冷却水系统图.pdf",
        "category": "管路",
        "equipment_id": "",
    },
    {
        "drawing_name": "继电保护配置图",
        "file_name": "继电保护配置图.pdf",
        "url": "http://localhost:8080/drawings/继电保护配置图.pdf",
        "category": "电气",
        "equipment_id": "",
    },
]


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------


def init_db():
    """Create tables and seed data. Idempotent. Retries on connection failure."""
    engine = create_engine(ENGINE_URL, echo=False, pool_pre_ping=True)

    # Retry loop: MySQL container may still be initializing
    max_retries = 10
    for attempt in range(1, max_retries + 1):
        try:
            Base.metadata.create_all(engine)
            with Session(engine) as session:
                count = session.query(Drawing).count()
                if count == 0:
                    for item in _SEED_DATA:
                        session.add(Drawing(**item))
                    session.commit()
            break  # success
        except Exception as e:
            if attempt == max_retries:
                raise
            print(f"  MySQL connection attempt {attempt}/{max_retries} failed: {e}, retrying ...")
            time.sleep(3)

    return engine


# Singleton engine
_engine = init_db()


# ---------------------------------------------------------------------------
# Query functions
# ---------------------------------------------------------------------------


def _query_by_name_session(session, name: str) -> list:
    """Internal: fuzzy match using an existing session. Returns raw Drawing objects."""
    return session.query(Drawing).filter(Drawing.drawing_name.like(f"%{name}%")).all()


def query_drawing_by_name(name: str) -> list[dict]:
    """Fuzzy-match drawing_name using LIKE.

    Args:
        name: Partial or full drawing name to search.

    Returns:
        List of dicts with all drawing fields.
    """
    with Session(_engine) as session:
        rows = _query_by_name_session(session, name)
        return [_row_to_dict(r) for r in rows]


def query_all_drawings() -> list[dict]:
    """Return all drawing records.

    Returns:
        List of dicts with all drawing fields.
    """
    with Session(_engine) as session:
        rows = session.query(Drawing).all()
        return [_row_to_dict(r) for r in rows]


def get_all_drawing_names() -> list[dict]:
    """Return all drawings as lightweight dicts for LLM semantic matching.

    Returns:
        List of dicts with keys: id, drawing_name, file_name, url, category, equipment_id.
    """
    return query_all_drawings()



def _row_to_dict(row: Drawing) -> dict:
    return {
        "id": row.id,
        "drawing_name": row.drawing_name,
        "file_name": row.file_name,
        "url": row.url,
        "category": row.category,
        "equipment_id": row.equipment_id,
    }


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== query_drawing_by_name('循环水泵') ===")
    for r in query_drawing_by_name("循环水泵"):
        print(r)

    print("\n=== query_drawing_by_name('锅炉') ===")
    for r in query_drawing_by_name("锅炉"):
        print(r)

    print("\n=== query_all_drawings() ===")
    all_drawings = query_all_drawings()
    assert len(all_drawings) >= 10, f"Expected >=10 records, got {len(all_drawings)}"
    for r in all_drawings:
        print(f"  [{r['category']}] {r['drawing_name']} — {r['url']}")
    print(f"\nTotal: {len(all_drawings)} records — OK")
