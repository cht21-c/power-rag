import React, { useState, useRef, useEffect, useCallback } from "react";
import {
  Lock, User, Shield, Radio, Activity, FileText, Download,
  ChevronDown, ChevronRight, Send, AlertTriangle, CheckCircle2,
  CircleDashed, LogOut, Settings, Upload, Search, Gauge, Zap,
  Filter, Clock, ArrowRight, X, Terminal
} from "lucide-react";

/* ---------------------------------------------------------------
   设计说明（信号台 / Signal Console 方向）
   主题：电厂运维现场的巡检信号台 —— 深色控制台底色，
   用三色信号灯语义（青=正常/高置信、琥珀=告警/中置信、
   赤=故障/低置信）贯穿整个界面，呼应变电站/中控室仪表盘的
   视觉语言；数据类文本（trace_id、时间戳、分数）一律等宽字体，
   模拟仪表读数；卡片边框用细虚线，呼应工程图纸的图框线。
----------------------------------------------------------------- */

const FONT_IMPORT = `@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');`;

const BLUEPRINT_BG = {
  backgroundColor: "#0a0f1a",
  backgroundImage:
    "linear-gradient(rgba(94,234,212,0.045) 1px, transparent 1px), linear-gradient(90deg, rgba(94,234,212,0.045) 1px, transparent 1px)",
  backgroundSize: "28px 28px",
};

const mono = { fontFamily: "'JetBrains Mono', monospace" };
const display = { fontFamily: "'Space Grotesk', sans-serif" };
const body = { fontFamily: "'Inter', sans-serif" };

/* ---------------- 演示数据 ---------------- */

const DEMO_USERS = {
  "op-2201": { role: "operator", label: "巡检员 · op-2201" },
  "admin-01": { role: "admin", label: "系统管理员 · admin-01" },
};

const SOURCES_NORMAL = [
  { title: "循环水泵组维护规程 v3.2", score: 0.91, page: 14 },
  { title: "锅炉给水系统操作手册", score: 0.86, page: 7 },
];

const DRAWING_HIT = {
  name: "1号机组循环水泵接线图",
  file: "DWG-CWP-1103.pdf",
  category: "电气接线图",
  equipment: "循环水泵-01",
  url: "http://10.20.30.40/drawings/DWG-CWP-1103.pdf",
};

function classify(text) {
  const t = text.trim();
  if (!t) return "empty";
  if (/图纸|下载|调取|找图/.test(t)) {
    return /找不到|不存在|没有的设备/.test(t) ? "drawing_miss" : "drawing_hit";
  }
  if (/^(这个|那个|它|该设备)/.test(t) || /这个怎么|那怎么/.test(t)) return "rewrite";
  if (/^维护$|^保养$|怎么弄$/.test(t)) return "clarify";
  if (/XJ-9000|未知型号|没听过的参数|参数是多少/.test(t)) return "refuse";
  return "normal";
}

function buildResponse(kind, rawText) {
  const id = Math.random().toString(36).slice(2, 9);
  const base = { id, role: "assistant", ts: Date.now() };
  switch (kind) {
    case "drawing_hit":
      return {
        ...base,
        route: "mysql",
        text: "已在图纸库中定位到匹配记录，可直接下载：",
        drawing: DRAWING_HIT,
        confidence: 0.94,
        modelUsed: "deepseek-chat",
      };
    case "drawing_miss":
      return {
        ...base,
        route: "mysql",
        text: "未找到精确匹配的图纸。请核对设备编号后重试，或联系图档管理员补充录入。",
        confidence: 0.3,
        modelUsed: "deepseek-chat",
      };
    case "rewrite":
      return {
        ...base,
        route: "rag",
        rewrittenFrom: rawText,
        rewrittenTo: "循环水泵接线图的申请下载流程是什么",
        text: "该图纸可在「图纸检索」页签直接搜索设备编号获取下载链接；若权限受限，需由值班负责人在系统内审批。",
        confidence: 0.82,
        sources: SOURCES_NORMAL.slice(0, 1),
        modelUsed: "deepseek-chat",
      };
    case "clarify":
      return {
        ...base,
        route: "clarify",
        text: "这个问题可能指向两个方向，麻烦确认一下：",
        clarifyOptions: [
          "查询某设备的维护保养规程",
          "查询某设备的图纸下载方式",
        ],
        confidence: 0.47,
        modelUsed: "deepseek-chat",
      };
    case "refuse":
      return {
        ...base,
        route: "rag",
        text: "知识库中未检索到该型号 / 参数的相关记录，无法给出可靠数值，请以现场铭牌或厂家资料为准，避免使用估算值。",
        confidence: 0.12,
        refused: true,
        modelUsed: "deepseek-chat",
      };
    default:
      return {
        ...base,
        route: "rag",
        text: "循环水泵组启动前需完成润滑油位检查、密封水投用、盘车确认三项前置操作，确认无异常后方可合闸启动，启动后 10 分钟内需巡检轴承温度与振动值。",
        confidence: 0.88,
        sources: SOURCES_NORMAL,
        modelUsed: "deepseek-chat",
      };
  }
}

/* ---------------- 置信度信号条 ---------------- */

function ConfidenceStrip({ value }) {
  const zone = value >= 0.7 ? "high" : value >= 0.45 ? "mid" : "low";
  const color =
    zone === "high" ? "#5EEAD4" : zone === "mid" ? "#F5A524" : "#F0546E";
  const label =
    zone === "high" ? "高置信" : zone === "mid" ? "中置信 · 请核实" : "低置信 · 谨慎参考";
  return (
    <div className="flex items-center gap-2 mt-2">
      <Gauge size={12} style={{ color }} />
      <div className="flex-1 h-1 rounded-full bg-slate-800 relative overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-700"
          style={{ width: `${value * 100}%`, backgroundColor: color }}
        />
      </div>
      <span
        className="text-[10px] tracking-wide uppercase"
        style={{ ...mono, color }}
      >
        {label} · {value.toFixed(2)}
      </span>
    </div>
  );
}

/* ---------------- 来源引用 ---------------- */

function SourceList({ sources }) {
  const [open, setOpen] = useState(false);
  if (!sources || sources.length === 0) return null;
  return (
    <div className="mt-3 border-t border-dashed border-slate-700 pt-2">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1 text-[11px] text-slate-400 hover:text-cyan-300 transition-colors"
        style={mono}
      >
        {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        引用来源 ({sources.length})
      </button>
      {open && (
        <div className="mt-2 space-y-1.5">
          {sources.map((s, i) => (
            <div
              key={i}
              className="flex items-center justify-between text-[11px] text-slate-400 bg-slate-900/60 border border-slate-800 rounded px-2 py-1.5"
            >
              <span className="truncate">{s.title} · p.{s.page}</span>
              <span style={mono} className="text-cyan-400/80 ml-2 shrink-0">
                {s.score.toFixed(2)}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* ---------------- 图纸卡片 ---------------- */

function DrawingCard({ drawing }) {
  return (
    <div className="mt-3 border border-dashed border-cyan-500/30 rounded-md bg-cyan-950/20 p-3">
      <div className="flex items-start gap-2.5">
        <div className="mt-0.5 shrink-0 w-8 h-8 rounded bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center">
          <FileText size={15} className="text-cyan-300" />
        </div>
        <div className="min-w-0 flex-1">
          <div className="text-[13px] text-slate-100 font-medium truncate">
            {drawing.name}
          </div>
          <div style={mono} className="text-[10px] text-slate-500 mt-0.5">
            {drawing.category} · {drawing.equipment} · {drawing.file}
          </div>
        </div>
        <a
          href={drawing.url}
          className="shrink-0 flex items-center gap-1 text-[11px] px-2.5 py-1.5 rounded bg-cyan-500/15 border border-cyan-500/40 text-cyan-300 hover:bg-cyan-500/25 transition-colors"
        >
          <Download size={12} /> 下载
        </a>
      </div>
    </div>
  );
}

/* ---------------- 澄清反问 ---------------- */

function ClarifyChips({ options, onPick }) {
  return (
    <div className="mt-3 flex flex-col gap-1.5">
      {options.map((opt, i) => (
        <button
          key={i}
          onClick={() => onPick(opt)}
          className="text-left text-[12px] px-3 py-2 rounded border border-slate-700 bg-slate-900/70 text-slate-300 hover:border-amber-400/50 hover:text-amber-200 transition-colors flex items-center justify-between group"
        >
          {opt}
          <ArrowRight size={12} className="opacity-0 group-hover:opacity-100 transition-opacity" />
        </button>
      ))}
    </div>
  );
}

/* ---------------- 消息气泡 ---------------- */

function MessageBubble({ msg, onPickClarify }) {
  if (msg.role === "user") {
    return (
      <div className="flex justify-end mb-4">
        <div
          className="max-w-[72%] rounded-lg rounded-tr-sm bg-slate-800/80 border border-slate-700 px-3.5 py-2.5 text-[13px] text-slate-100"
          style={body}
        >
          {msg.text}
        </div>
      </div>
    );
  }

  const routeMeta = {
    mysql: { label: "图纸检索通道", color: "text-cyan-400" },
    rag: { label: "知识检索通道", color: "text-cyan-400" },
    clarify: { label: "待澄清", color: "text-amber-400" },
  }[msg.route];

  return (
    <div className="flex justify-start mb-4">
      <div className="max-w-[78%]">
        <div className="flex items-center gap-1.5 mb-1 px-0.5">
          <Radio size={10} className={routeMeta?.color} />
          <span
            className={`text-[10px] tracking-wider uppercase ${routeMeta?.color}`}
            style={mono}
          >
            {routeMeta?.label}
          </span>
          {msg.modelUsed && (
            <span className="text-[10px] text-slate-600" style={mono}>
              · {msg.modelUsed}
            </span>
          )}
        </div>

        <div
          className={`rounded-lg rounded-tl-sm border px-3.5 py-3 text-[13px] leading-relaxed ${
            msg.refused
              ? "bg-rose-950/20 border-rose-500/30 text-rose-100"
              : "bg-slate-900/80 border-slate-800 text-slate-200"
          }`}
          style={body}
        >
          {msg.rewrittenFrom && (
            <div
              className="text-[11px] text-slate-500 mb-2 pb-2 border-b border-dashed border-slate-700"
              style={mono}
            >
              我理解你在问：「{msg.rewrittenTo}」
            </div>
          )}

          {msg.refused && (
            <div className="flex items-center gap-1.5 mb-1.5 text-rose-300">
              <AlertTriangle size={13} />
              <span className="text-[11px] font-medium">未检索到可靠依据</span>
            </div>
          )}

          <div>{msg.text}</div>

          {msg.drawing && <DrawingCard drawing={msg.drawing} />}
          {msg.clarifyOptions && (
            <ClarifyChips options={msg.clarifyOptions} onPick={onPickClarify} />
          )}
          {typeof msg.confidence === "number" && !msg.clarifyOptions && (
            <ConfidenceStrip value={msg.confidence} />
          )}
          <SourceList sources={msg.sources} />
        </div>
      </div>
    </div>
  );
}

/* ---------------- 登录页 ---------------- */

function LoginScreen({ onLogin }) {
  const [key, setKey] = useState("");
  const [error, setError] = useState("");
  const [checking, setChecking] = useState(false);

  const submit = () => {
    setError("");
    setChecking(true);
    setTimeout(() => {
      const user = DEMO_USERS[key.trim()];
      if (!user) {
        setError("鉴权失败：Key 无效或已停用");
        setChecking(false);
        return;
      }
      setChecking(false);
      onLogin({ id: key.trim(), ...user });
    }, 500);
  };

  return (
    <div
      className="min-h-screen w-full flex items-center justify-center relative"
      style={BLUEPRINT_BG}
    >
      <style>{FONT_IMPORT}</style>
      <div className="absolute top-6 left-6 flex items-center gap-2 text-slate-600">
        <Terminal size={14} />
        <span className="text-[11px] tracking-widest uppercase" style={mono}>
          Plant Ops Signal Console
        </span>
      </div>

      <div className="w-[380px]">
        <div className="mb-8 text-center">
          <div className="inline-flex items-center justify-center w-12 h-12 rounded-md border border-cyan-500/30 bg-cyan-500/10 mb-4">
            <Shield size={20} className="text-cyan-300" />
          </div>
          <h1
            className="text-2xl text-slate-100 font-semibold"
            style={display}
          >
            厂务智答 · 信号台
          </h1>
          <p className="text-[12px] text-slate-500 mt-1.5" style={body}>
            电厂运维知识检索 Agent · 身份鉴别接入点
          </p>
        </div>

        <div className="border border-slate-800 rounded-lg bg-slate-900/60 p-5">
          <label
            className="text-[11px] uppercase tracking-wider text-slate-500 flex items-center gap-1.5 mb-2"
            style={mono}
          >
            <Lock size={11} /> Access Key
          </label>
          <input
            type="password"
            value={key}
            onChange={(e) => setKey(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && submit()}
            placeholder="输入你的 API Key"
            className="w-full bg-slate-950 border border-slate-700 rounded px-3 py-2.5 text-[13px] text-slate-100 placeholder-slate-600 outline-none focus:border-cyan-500/60 transition-colors"
            style={mono}
          />

          {error && (
            <div className="flex items-center gap-1.5 mt-2.5 text-rose-400 text-[11px]">
              <AlertTriangle size={12} /> {error}
            </div>
          )}

          <button
            onClick={submit}
            disabled={checking || !key.trim()}
            className="w-full mt-4 bg-cyan-500/15 hover:bg-cyan-500/25 disabled:opacity-40 disabled:cursor-not-allowed border border-cyan-500/40 text-cyan-300 rounded py-2.5 text-[13px] font-medium transition-colors flex items-center justify-center gap-2"
          >
            {checking ? (
              <>
                <CircleDashed size={13} className="animate-spin" /> 校验中
              </>
            ) : (
              <>
                <ArrowRight size={13} /> 进入控制台
              </>
            )}
          </button>

          <div
            className="mt-4 pt-4 border-t border-dashed border-slate-800 text-[10px] text-slate-600 leading-relaxed"
            style={mono}
          >
            演示 Key： op-2201（巡检员）/ admin-01（管理员）
          </div>
        </div>
      </div>
    </div>
  );
}

/* ---------------- 顶部状态条 ---------------- */

function TopBar({ user, modelStatus, onToggleModel, onLogout, onOpenTrace }) {
  const degraded = modelStatus === "degraded";
  return (
    <div className="h-14 border-b border-slate-800 bg-slate-950/80 backdrop-blur flex items-center justify-between px-5 shrink-0">
      <div className="flex items-center gap-3">
        <Terminal size={15} className="text-cyan-400" />
        <span className="text-[13px] text-slate-200 font-medium" style={display}>
          厂务智答 · 信号台
        </span>
        <div className="h-4 w-px bg-slate-800 mx-1" />
        <div className="flex items-center gap-1.5">
          <span
            className={`w-1.5 h-1.5 rounded-full ${
              degraded ? "bg-amber-400 animate-pulse" : "bg-cyan-400"
            }`}
          />
          <span className="text-[11px] text-slate-500" style={mono}>
            {degraded ? "已切换至备用模型通道" : "deepseek-chat · CLOSED"}
          </span>
        </div>
      </div>

      <div className="flex items-center gap-3">
        {user.role === "admin" && (
          <>
            <button
              onClick={onToggleModel}
              className="text-[11px] text-slate-500 hover:text-amber-300 border border-slate-800 hover:border-amber-500/40 rounded px-2.5 py-1.5 transition-colors flex items-center gap-1.5"
              style={mono}
            >
              <Zap size={11} /> 模拟熔断
            </button>
            <button
              onClick={onOpenTrace}
              className="text-[11px] text-slate-500 hover:text-cyan-300 border border-slate-800 hover:border-cyan-500/40 rounded px-2.5 py-1.5 transition-colors flex items-center gap-1.5"
              style={mono}
            >
              <Search size={11} /> 链路溯源
            </button>
          </>
        )}
        <div className="h-4 w-px bg-slate-800" />
        <div className="flex items-center gap-1.5 text-[11px] text-slate-400" style={mono}>
          <User size={11} />
          {DEMO_USERS[user.id]?.label}
        </div>
        <button
          onClick={onLogout}
          className="text-slate-600 hover:text-rose-400 transition-colors"
        >
          <LogOut size={14} />
        </button>
      </div>
    </div>
  );
}

/* ---------------- 侧栏 ---------------- */

function Sidebar({ user, brand, setBrand, onIngestClick }) {
  const brands = ["全部", "大华", "海康", "Basler", "HikRobot"];
  return (
    <div className="w-56 border-r border-slate-800 bg-slate-950/50 flex flex-col shrink-0">
      <div className="p-4">
        <div
          className="text-[10px] uppercase tracking-wider text-slate-600 mb-2 flex items-center gap-1.5"
          style={mono}
        >
          <Filter size={11} /> 品牌过滤
        </div>
        <div className="space-y-1">
          {brands.map((b) => (
            <button
              key={b}
              onClick={() => setBrand(b)}
              className={`w-full text-left text-[12px] px-2.5 py-1.5 rounded transition-colors ${
                brand === b
                  ? "bg-cyan-500/10 text-cyan-300 border border-cyan-500/30"
                  : "text-slate-500 hover:text-slate-300 border border-transparent"
              }`}
            >
              {b}
            </button>
          ))}
        </div>
      </div>

      <div className="p-4 border-t border-slate-800">
        <div
          className="text-[10px] uppercase tracking-wider text-slate-600 mb-2"
          style={mono}
        >
          会话
        </div>
        <div className="text-[12px] text-slate-300 bg-slate-900/60 border border-slate-800 rounded px-2.5 py-2 flex items-center gap-2">
          <Activity size={12} className="text-cyan-400" />
          当前巡检会话
        </div>
      </div>

      {user.role === "admin" && (
        <div className="p-4 border-t border-slate-800 mt-auto">
          <div
            className="text-[10px] uppercase tracking-wider text-slate-600 mb-2 flex items-center gap-1.5"
            style={mono}
          >
            <Shield size={11} /> 管理操作
          </div>
          <button
            onClick={onIngestClick}
            className="w-full flex items-center gap-2 text-[12px] text-slate-400 hover:text-cyan-300 border border-slate-800 hover:border-cyan-500/30 rounded px-2.5 py-2 transition-colors"
          >
            <Upload size={12} /> 文档入库
          </button>
        </div>
      )}
    </div>
  );
}

/* ---------------- 入库面板 ---------------- */

function IngestPanel({ onClose }) {
  const [running, setRunning] = useState(false);
  const [report, setReport] = useState(null);

  const runIngest = () => {
    setRunning(true);
    setReport(null);
    setTimeout(() => {
      setRunning(false);
      setReport({
        total: 12,
        ok: 11,
        fail: 1,
        failList: [{ file: "泵组图纸_扫描件.pdf", reason: "OCR 超时" }],
        elapsedMs: 4820,
      });
    }, 1400);
  };

  return (
    <div className="absolute inset-0 bg-slate-950/80 backdrop-blur-sm flex items-center justify-center z-20">
      <div className="w-[420px] border border-slate-800 rounded-lg bg-slate-900 p-5">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2 text-slate-200 text-[13px] font-medium">
            <Upload size={14} className="text-cyan-400" /> 文档入库流水线
          </div>
          <button onClick={onClose} className="text-slate-600 hover:text-slate-300">
            <X size={14} />
          </button>
        </div>

        <div className="border border-dashed border-slate-700 rounded p-4 text-center text-[11px] text-slate-500 mb-4" style={mono}>
          sdk_docs/ 目录下检测到 12 个待处理文档
        </div>

        <button
          onClick={runIngest}
          disabled={running}
          className="w-full bg-cyan-500/15 hover:bg-cyan-500/25 border border-cyan-500/40 text-cyan-300 rounded py-2 text-[12px] font-medium flex items-center justify-center gap-2 disabled:opacity-50"
        >
          {running ? (
            <>
              <CircleDashed size={12} className="animate-spin" /> 执行中 · 扫描→解析→分块→嵌入→入库
            </>
          ) : (
            "开始入库"
          )}
        </button>

        {report && (
          <div className="mt-4 border-t border-dashed border-slate-800 pt-3 text-[11px]" style={mono}>
            <div className="flex justify-between text-slate-400 mb-1.5">
              <span>共处理 {report.total} 个文档</span>
              <span>{report.elapsedMs}ms</span>
            </div>
            <div className="flex gap-3 mb-2">
              <span className="text-cyan-300 flex items-center gap-1">
                <CheckCircle2 size={11} /> 成功 {report.ok}
              </span>
              <span className="text-rose-400 flex items-center gap-1">
                <AlertTriangle size={11} /> 失败 {report.fail}
              </span>
            </div>
            {report.failList.map((f, i) => (
              <div key={i} className="text-slate-500 pl-3 border-l border-rose-500/30">
                {f.file} · {f.reason}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

/* ---------------- 链路溯源面板（管理端） ---------------- */

function TracePanel({ onClose, lastTrace }) {
  const nodes = [
    { name: "query_understand", latency: 210, status: "ok" },
    { name: "route_decision", latency: 4, status: "ok" },
    { name: lastTrace?.route === "mysql" ? "mysql_query" : "rag_retrieve", latency: 340, status: "ok" },
    { name: "llm_generate", latency: 1180, status: lastTrace?.refused ? "fallback" : "ok" },
  ];
  const statusColor = { ok: "#5EEAD4", fallback: "#F5A524", error: "#F0546E" };

  return (
    <div className="absolute inset-0 bg-slate-950/80 backdrop-blur-sm flex items-center justify-center z-20">
      <div className="w-[520px] border border-slate-800 rounded-lg bg-slate-900 p-5">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2 text-slate-200 text-[13px] font-medium">
            <Search size={14} className="text-cyan-400" /> 链路溯源
          </div>
          <button onClick={onClose} className="text-slate-600 hover:text-slate-300">
            <X size={14} />
          </button>
        </div>

        <div className="text-[11px] text-slate-500 mb-3" style={mono}>
          trace_id: {lastTrace?.id ?? "—"} · user: op-2201
        </div>

        <div className="space-y-1.5">
          {nodes.map((n, i) => (
            <div
              key={i}
              className="flex items-center justify-between border border-slate-800 rounded px-3 py-2 bg-slate-950/60"
            >
              <div className="flex items-center gap-2">
                <span
                  className="w-1.5 h-1.5 rounded-full"
                  style={{ backgroundColor: statusColor[n.status] }}
                />
                <span className="text-[12px] text-slate-300" style={mono}>
                  {n.name}
                </span>
              </div>
              <div className="flex items-center gap-1.5 text-[11px] text-slate-500" style={mono}>
                <Clock size={10} /> {n.latency}ms
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

/* ---------------- 主控制台 ---------------- */

function ConsoleApp({ user, onLogout }) {
  const [messages, setMessages] = useState([
    {
      id: "welcome",
      role: "assistant",
      route: "rag",
      text: "已接入电厂运维知识库与图纸检索通道。可以直接提问，也可以试试：「循环水泵组维护」「调取1号机组循环水泵接线图」。",
      confidence: 1,
      modelUsed: "system",
    },
  ]);
  const [input, setInput] = useState("");
  const [brand, setBrand] = useState("全部");
  const [modelStatus, setModelStatus] = useState("normal");
  const [showIngest, setShowIngest] = useState(false);
  const [showTrace, setShowTrace] = useState(false);
  const [lastTrace, setLastTrace] = useState(null);
  const scrollRef = useRef(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  const send = useCallback(
    (text) => {
      const trimmed = text.trim();
      if (!trimmed) return;
      const userMsg = { id: Math.random().toString(36).slice(2, 9), role: "user", text: trimmed };
      const kind = classify(trimmed);
      const reply = buildResponse(kind, trimmed);
      if (modelStatus === "degraded") reply.modelUsed = "deepseek-chat-backup";
      setMessages((m) => [...m, userMsg, reply]);
      setLastTrace(reply);
      setInput("");
    },
    [modelStatus]
  );

  const pickClarify = (option) => send(option);

  return (
    <div className="h-screen w-full flex flex-col relative" style={BLUEPRINT_BG}>
      <style>{FONT_IMPORT}</style>
      <TopBar
        user={user}
        modelStatus={modelStatus}
        onToggleModel={() => setModelStatus((s) => (s === "normal" ? "degraded" : "normal"))}
        onLogout={onLogout}
        onOpenTrace={() => setShowTrace(true)}
      />

      <div className="flex flex-1 min-h-0">
        <Sidebar user={user} brand={brand} setBrand={setBrand} onIngestClick={() => setShowIngest(true)} />

        <div className="flex-1 flex flex-col min-w-0">
          <div ref={scrollRef} className="flex-1 overflow-y-auto px-6 py-5">
            <div className="max-w-2xl mx-auto">
              {messages.map((m) => (
                <MessageBubble key={m.id} msg={m} onPickClarify={pickClarify} />
              ))}
            </div>
          </div>

          <div className="border-t border-slate-800 p-4">
            <div className="max-w-2xl mx-auto flex items-center gap-2">
              <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && send(input)}
                placeholder="输入问题，例如：循环水泵组启动前需要检查什么"
                className="flex-1 bg-slate-900 border border-slate-800 focus:border-cyan-500/50 rounded-md px-3.5 py-2.5 text-[13px] text-slate-100 placeholder-slate-600 outline-none transition-colors"
                style={body}
              />
              <button
                onClick={() => send(input)}
                className="w-10 h-10 shrink-0 rounded-md bg-cyan-500/15 hover:bg-cyan-500/25 border border-cyan-500/40 text-cyan-300 flex items-center justify-center transition-colors"
              >
                <Send size={15} />
              </button>
            </div>
            <div className="max-w-2xl mx-auto mt-2 text-[10px] text-slate-600" style={mono}>
              品牌过滤：{brand} · 试试「调取图纸」「那这个怎么下载」「XJ-9000参数是多少」
            </div>
          </div>
        </div>
      </div>

      {showIngest && <IngestPanel onClose={() => setShowIngest(false)} />}
      {showTrace && <TracePanel onClose={() => setShowTrace(false)} lastTrace={lastTrace} />}
    </div>
  );
}

/* ---------------- 根组件 ---------------- */

export default function App() {
  const [user, setUser] = useState(null);
  if (!user) return <LoginScreen onLogin={setUser} />;
  return <ConsoleApp user={user} onLogout={() => setUser(null)} />;
}
