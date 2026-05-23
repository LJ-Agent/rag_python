"""LLM QA generator — prompt assembly, streaming, and fallback handling."""
from typing import AsyncIterator

from common.config_loader import get_config
from common.exception.exceptions import GenerationException
from common.util.logger import get_logger
from infrastructure.llm.llm_adapter import get_llm_adapter

logger = get_logger()

SYSTEM_PROMPT = """你是一个专业的知识库问答助手。请根据提供的参考文档内容回答用户问题。

要求：
1. 仅基于提供的参考文档内容回答，不要使用外部知识
2. 如果文档内容不足以回答问题，请明确说明"根据现有资料无法回答该问题"
3. 回答要准确、简洁、有条理
4. 引用具体文档内容时，可以标注来源

参考文档内容：
{contexts}

请回答用户问题：{query}"""

FALLBACK_ANSWER = "抱歉，根据现有资料无法回答该问题。请尝试重新描述您的问题，或补充相关资料。"


class QAGenerator:
    """Prompt assembly and LLM generation for QA, supporting both sync and streaming."""

    def __init__(self):
        self._llm = get_llm_adapter()
        cfg = get_config()["llm"]
        self._model = cfg["chat_model"]
        self._max_tokens = cfg["max_tokens"]
        self._temperature = cfg["temperature"]

    def generate(
        self,
        query: str,
        contexts: list[str],
        model: str | None = None,
        conversation_id: str | None = None,
        params: dict | None = None,
    ) -> dict:
        """Non-streaming generation. Returns {content, token_count, finish_reason}."""
        if not contexts:
            return {"content": self._fallback(), "token_count": 0, "finish_reason": "no_context"}

        messages = self._build_messages(query, contexts)
        try:
            result = self._llm.generate(messages, model=model or self._model)
            logger.info(
                f"QA generated: query='{query[:50]}...', tokens={result['token_count']}, "
                f"reason={result['finish_reason']}"
            )
            return result
        except GenerationException:
            return {"content": self._fallback(), "token_count": 0, "finish_reason": "error"}

    async def generate_stream(
        self,
        query: str,
        contexts: list[str],
        model: str | None = None,
        conversation_id: str | None = None,
        params: dict | None = None,
    ) -> AsyncIterator[dict]:
        """Streaming generation. Yields {content, is_end, token_count, finish_reason} chunks."""
        if not contexts:
            yield {"content": self._fallback(), "is_end": True, "token_count": 0, "finish_reason": "no_context"}
            return

        messages = self._build_messages(query, contexts)
        try:
            async for chunk in self._llm.generate_stream(messages, model=model or self._model):
                yield chunk
        except GenerationException:
            yield {
                "content": self._fallback(),
                "is_end": True,
                "token_count": 0,
                "finish_reason": "error",
            }

    def _build_messages(self, query: str, contexts: list[str]) -> list[dict]:
        """Assemble system + user messages with contexts."""
        context_text = "\n\n---\n\n".join(
            f"[文档片段 {i + 1}]\n{ctx}" for i, ctx in enumerate(contexts)
        )
        system_msg = SYSTEM_PROMPT.format(contexts=context_text, query=query)
        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": query},
        ]

    def _fallback(self) -> str:
        return FALLBACK_ANSWER
