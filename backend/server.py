"""
Camera SDK RAG Agent — FastAPI 后端（懒加载版）

启动: uvicorn backend.server:app --host 0.0.0.0 --port 8765
"""

import asyncio, json, logging, os, sys, time, uuid
from pathlib import Path
from typing import Optional

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT / "camera_sdk_agent"))
sys.path.insert(0, str(_PROJECT))

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Camera SDK RAG Agent")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# 懒加载
_graph = _retriever = _store = None

def _init_agent():
    global _graph, _retriever, _store
    if _graph is not None: return

    from config import DEEPSEEK_API_KEY, DEEPSEEK_MODEL
    from ingestion.embedder import Embedder
    from store.qdrant_store import QdrantStore
    from retrieval.retriever import HybridRetriever
    from agent.graph import create_agent_graph

    _store = QdrantStore()
    embedder = Embedder()
    _retriever = HybridRetriever(_store, embedder)
    _graph = create_agent_graph(retriever=_retriever)
    pts = _store.count_points()
    logger.info("Agent ready, Qdrant points: %d", pts)

# Models
class LoginRequest(BaseModel): api_key: str
class ChatRequest(BaseModel): query: str; brand: Optional[str]=None; messages: list=[]

# ── Login ──
@app.post("/api/login")
async def login(req: LoginRequest):
    from framework.auth.middleware import authenticate_session, AuthError
    try:
        user = authenticate_session(req.api_key)
        return {"ok":True, "user_id":user["user_id"], "role":user["role"], "token":uuid.uuid4().hex[:16]}
    except AuthError as e:
        raise HTTPException(401, detail=str(e))

# ── Chat (SSE流式) ──
@app.post("/api/chat")
async def chat(req: ChatRequest):
    _init_agent()
    from framework.auth.middleware import set_current_user
    from agent.graph import AgentState
    set_current_user("guest", "operator")

    state: AgentState = {
        "query":req.query, "original_query":"", "rewritten_query":"",
        "camera_brand":req.brand, "keywords":[], "intent":"", "route":"rag",
        "confidence":0.0, "needs_clarification":False, "clarification_question":"",
        "parse_status":"success", "retrieved_context":"", "formatted_context":"",
        "max_retrieval_score":0.0, "messages":req.messages or [],
        "answer":"", "error":None, "_compressed":False,
    }
    thread = {"configurable":{"thread_id": f"api-{uuid.uuid4().hex[:8]}"}}

    async def stream():
        try:
            result = _graph.invoke(state, config=thread)
            answer = result.get("answer","")
            route = result.get("route","rag")
            conf = result.get("confidence",0.0)
            clarify = result.get("needs_clarification",False)

            yield f"data: {json.dumps({'type':'route','route':route,'confidence':conf}, ensure_ascii=False)}\n\n"
            if clarify:
                yield f"data: {json.dumps({'type':'clarify','text':answer}, ensure_ascii=False)}\n\n"
            else:
                for i in range(0, len(answer), 5):
                    yield f"data: {json.dumps({'type':'token','text':answer[i:i+5]}, ensure_ascii=False)}\n\n"
                    await asyncio.sleep(0.02)

                formatted = result.get("formatted_context","")
                # Parse File+Page pairs from formatted context
                sources = []
                cur_file = ""
                for l in formatted.split("\n"):
                    if l.startswith("File:"):
                        cur_file = l.replace("File:","").strip()
                    elif l.startswith("Page:") and cur_file:
                        pg = l.replace("Page:","").strip()
                        sources.append({"file": cur_file, "page": pg})
                        cur_file = ""
                if sources:
                    yield f"data: {json.dumps({'type':'sources','items':sources[:5]}, ensure_ascii=False)}\n\n"

            yield f"data: {json.dumps({'type':'done','trace_id':result.get('_trace_id','')}, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type':'error','text':str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")

# ── 审计查询 ──
@app.get("/api/audit/trace/{trace_id}")
async def audit_trace(trace_id: str):
    log_dir = _PROJECT / "logs" / "audit"
    if not log_dir.exists(): return {"records":[]}
    results = []
    for fp in sorted(log_dir.glob("*.jsonl"), reverse=True):
        for line in open(fp, encoding="utf-8"):
            try:
                r = json.loads(line)
                if r.get("trace_id")==trace_id: results.append(r)
            except: pass
        if results: break
    results.sort(key=lambda r: r.get("timestamp",""))
    return {"trace_id":trace_id, "records":results, "count":len(results)}

# ── 文档摄取 ──
@app.post("/api/ingest")
async def ingest(files: list[UploadFile]=File(...), brand: str=Form("unknown")):
    _init_agent()
    from config import SDK_DOCS_DIR
    from ingestion.parsers import parse_document
    from ingestion.embedder import Embedder
    from ingestion.chunker import chunk_document
    results = []
    for file in files:
        t0 = time.time()
        try:
            brand_dir = SDK_DOCS_DIR / brand
            brand_dir.mkdir(parents=True, exist_ok=True)
            dest = brand_dir / file.filename
            dest.write_bytes(await file.read())
            parsed = parse_document(dest)
            embedder = Embedder()
            chunks = chunk_document(parsed, embedder)
            texts = [c.text for c in chunks]
            vectors = embedder.encode(texts, show_progress=False)
            _store.create_collection(force_recreate=False)
            inserted = _store.insert_chunks(chunks, vectors)
            results.append({"file":file.filename,"ok":True,"chunks":len(chunks),"elapsed":round(time.time()-t0,1)})
        except Exception as e:
            results.append({"file":file.filename,"ok":False,"error":str(e)})
    if _retriever:
        try: _retriever.refresh_bm25()
        except: pass
    return {"results":results, "total":len(results), "success":sum(1 for r in results if r["ok"])}

# ── 前端页面 ──
@app.get("/")
async def serve_frontend():
    return FileResponse(str(_PROJECT / "backend" / "index.html"))

# ── 系统状态 ──
@app.get("/api/config")
async def get_config():
    _init_agent()
    from config import DEEPSEEK_MODEL
    return {"qdrant_points": _store.count_points() if _store else 0,
            "model": DEEPSEEK_MODEL,
            "brands": ["全部品牌","dahua","haikang","basler","hikrobot"], "ok":True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)
