"""ModelClient 抽象接口 + DeepSeekClient 实现"""
import abc, logging, time
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)

class ModelClient(abc.ABC):
    @abc.abstractmethod
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        ...

    @abc.abstractmethod
    def generate_stream(self, messages: List[Dict[str, str]], **kwargs) -> Iterator[str]:
        ...

class DeepSeekClient(ModelClient):
    def __init__(self, model: str = "deepseek-chat", api_key: str = "",
                 base_url: str = "https://api.deepseek.com", temperature: float = 0.2,
                 max_tokens: int = 2048):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

    def _get_client(self):
        if self._client is None:
            from langchain_openai import ChatOpenAI
            self._client = ChatOpenAI(
                model=self.model, api_key=self.api_key, base_url=self.base_url,
                temperature=self.temperature, max_tokens=self.max_tokens,
            )
        return self._client

    def generate(self, messages, **kwargs):
        client = self._get_client()
        response = client.invoke(messages, **kwargs)
        return response.content

    def generate_stream(self, messages, **kwargs):
        from langchain_core.callbacks import BaseCallbackHandler
        client = self._get_client()
        client.streaming = True
        chunks = []
        for chunk in client.stream(messages, **kwargs):
            if hasattr(chunk, 'content') and chunk.content:
                chunks.append(chunk.content)
                yield chunk.content
