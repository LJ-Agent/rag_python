"""LLM API adapter — unified interface for OpenAI-compatible models, chat + embedding."""
from typing import AsyncIterator

from openai import AsyncOpenAI, OpenAI

from common.config_loader import get_config
from common.exception.exceptions import AIComputeException, GenerationException
from common.util.logger import get_logger

logger = get_logger()


class LLMAdapter:
    """Unified LLM client supporting both embedding and chat/completion with retry."""

    def __init__(self):
        cfg = get_config()["llm"]
        self._api_key = cfg["api_key"]
        self._base_url = cfg["base_url"]
        self._embedding_model = cfg["embedding_model"]
        self._chat_model = cfg["chat_model"]
        self._local_embedding_model = cfg.get("local_embedding_model")
        self._max_tokens = cfg["max_tokens"]
        self._temperature = cfg["temperature"]
        self._timeout = cfg.get("request_timeout", 60)
        self._max_retries = cfg.get("max_retries", 3)

        self._client = OpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=self._timeout,
            max_retries=self._max_retries,
        )
        self._async_client = AsyncOpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=self._timeout,
            max_retries=self._max_retries,
        )
        self._local_embedder = None

    # ─── Embedding ─────────────────────────────────────────

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Batch compute embeddings for a list of texts."""
        try:
            response = self._client.embeddings.create(
                model=self._embedding_model,
                input=texts,
            )
            return [d.embedding for d in response.data]
        except Exception as e:
            raise AIComputeException(f"Embedding failed: {e}")

    def embed_single(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    # ─── Chat Completion (non-streaming) ───────────────────

    def generate(self, messages: list[dict], model: str | None = None) -> dict:
        """Non-streaming chat completion. Returns {content, token_count, finish_reason}."""
        try:
            response = self._client.chat.completions.create(
                model=model or self._chat_model,
                messages=messages,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
            choice = response.choices[0]
            return {
                "content": choice.message.content or "",
                "token_count": response.usage.total_tokens if response.usage else 0,
                "finish_reason": choice.finish_reason or "stop",
            }
        except Exception as e:
            raise GenerationException(f"LLM generation failed: {e}")

    # ─── Chat Completion (streaming) ────────────────────────

    async def generate_stream(
        self, messages: list[dict], model: str | None = None
    ) -> AsyncIterator[dict]:
        """Streaming chat completion. Yields {content, is_end, token_count, finish_reason} chunks."""
        try:
            stream = await self._async_client.chat.completions.create(
                model=model or self._chat_model,
                messages=messages,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                stream=True,
            )
            token_count = 0
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    token_count += 1
                    yield {
                        "content": content,
                        "is_end": False,
                        "token_count": token_count,
                        "finish_reason": "",
                    }
            yield {
                "content": "",
                "is_end": True,
                "token_count": token_count,
                "finish_reason": "stop",
            }
        except Exception as e:
            raise GenerationException(f"LLM streaming failed: {e}")

    # ─── Local Embedding (sentence-transformers) ────────────

    def _init_local_embedder(self):
        if self._local_embedder is None and self._local_embedding_model:
            from sentence_transformers import SentenceTransformer

            self._local_embedder = SentenceTransformer(self._local_embedding_model)
            logger.info(f"Local embedding model loaded: {self._local_embedding_model}")

    def embed_local(self, texts: list[str]) -> list[list[float]]:
        """Use local sentence-transformers model for embedding (no API cost)."""
        self._init_local_embedder()
        if self._local_embedder is None:
            raise AIComputeException("Local embedding model not configured")
        embeddings = self._local_embedder.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()


# Singleton
_llm_adapter: LLMAdapter | None = None


def get_llm_adapter() -> LLMAdapter:
    global _llm_adapter
    if _llm_adapter is None:
        _llm_adapter = LLMAdapter()
    return _llm_adapter
