"""ConversationCompressor — 滑动窗口 + LLM 摘要压缩"""
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_COMPRESS_PROMPT = """将以下对话历史压缩为结构化摘要。保留所有关键实体信息。

## 对话历史：
{history}

## 要求：
1. 用要点列表形式总结，每项格式：- [时间/轮次] 用户询问了/操作了：XXX
2. 保留：设备型号、参数值、图纸编号、规程名称、问题答案要点
3. 不要泛化为"用户问了几个问题"，要具体可追溯
4. 用中文输出，控制在 300 字以内

## 已有摘要（如有）：
{existing_summary}

输出只返回摘要文本，不要加其他内容。"""


class ConversationCompressor:
    """管理对话历史的滑动窗口 + 摘要压缩。

    用法:
        comp = ConversationCompressor(router, window_turns=3)
        compressed = comp.compress(messages)
    """

    def __init__(self, model_router, window_turns: int = 3,
                 trigger_tokens: int = 3500):
        self._router = model_router
        self.window_turns = window_turns
        self.trigger_tokens = trigger_tokens
        self._summary_cache = ""

    def needs_compression(self, messages: List[Dict[str, str]]) -> bool:
        """判断是否需要触发压缩。"""
        if len(messages) <= self.window_turns * 2:
            return False
        total_chars = sum(len(m.get("content", "")) for m in messages)
        estimated_tokens = int(total_chars * 0.3)
        return estimated_tokens > self.trigger_tokens

    def compress(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """执行压缩：保留最近 window_turns 轮原文，更早的压缩为摘要。

        Returns:
            压缩后的消息列表（window 原文 + 摘要 system 消息）
        """
        if not self.needs_compression(messages):
            return messages

        turn_size = 2  # user + assistant per turn
        window_start = max(0, len(messages) - self.window_turns * turn_size)
        window_msgs = messages[window_start:]
        old_msgs = messages[:window_start]

        # 构建历史文本
        history_lines = []
        for i, m in enumerate(old_msgs):
            role = "用户" if m.get("role") == "user" else "助手"
            content = m.get("content", "")[:200]
            history_lines.append(f"[{i//2 + 1}] {role}: {content}")

        # 尝试 LLM 摘要
        try:
            prompt = _COMPRESS_PROMPT.format(
                history="\n".join(history_lines),
                existing_summary=self._summary_cache or "(无)",
            )
            summary = self._router.generate([{"role": "user", "content": prompt}])
            self._summary_cache = summary.strip()
            logger.info("[Compressor] Summary generated (%d chars)", len(self._summary_cache))
        except Exception as e:
            # 降级：回退到简单拼接（截断逻辑不变，不影响主流程）
            logger.warning("[Compressor] LLM summary failed: %s, falling back to truncation", e)
            self._summary_cache = f"[前{window_start//2}轮对话摘要不可用]"

        # 构建压缩后的消息列表
        summary_msg = {"role": "system", "content": f"对话历史摘要：\n{self._summary_cache}"}
        return [summary_msg] + window_msgs

    def get_summary(self) -> str:
        return self._summary_cache

    def hard_truncate(self, messages: List[Dict[str, str]],
                      max_tokens: int = 3500) -> List[Dict[str, str]]:
        """硬截断兜底：保留最近消息使总 token 不超限。"""
        kept = []
        used = 0
        for msg in reversed(messages):
            t = int(len(msg.get("content", "")) * 0.3)
            if used + t > max_tokens:
                break
            kept.insert(0, msg)
            used += t
        return kept
