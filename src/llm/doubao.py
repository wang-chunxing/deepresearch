import asyncio
import requests
from typing import List, Union, Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

class DoubaoLLM:
    def __init__(self, api_key, model, api_endpoint, max_tokens, reasoning_effort, temperature):
        self.api_key = api_key
        self.model = model
        self.api_endpoint = api_endpoint
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
    def _convert_messages_to_doubao_format(self, messages):
        doubao_messages = []
        for message in messages:
            if isinstance(message, HumanMessage):
                content = []
                if hasattr(message, 'content') and isinstance(message.content, str):
                    content.append({"text": message.content, "type": "text"})
                elif hasattr(message, 'content') and isinstance(message.content, list):
                    content = message.content
                doubao_messages.append({"role": "user", "content": content})
            elif isinstance(message, SystemMessage):
                doubao_messages.append({"role": "system", "content": [{"text": message.content, "type": "text"}]})
            elif isinstance(message, AIMessage):
                doubao_messages.append({"role": "assistant", "content": [{"text": message.content if hasattr(message, 'content') else str(message), "type": "text"}]})
        return doubao_messages
    async def ainvoke(self, messages):
        doubao_messages = self._convert_messages_to_doubao_format(messages)
        payload = {
            "model": self.model,
            "max_completion_tokens": self.max_tokens,
            "messages": doubao_messages,
            "reasoning_effort": self.reasoning_effort,
            "temperature": self.temperature
        }
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        max_retries = 3
        retry_delay = 2
        last_err = None
        for attempt in range(max_retries):
            try:
                response = requests.post(self.api_endpoint, json=payload, headers=headers, timeout=(10, 90))
                response.raise_for_status()
                result = response.json()
                if "choices" in result and result["choices"]:
                    content = result["choices"][0]["message"]["content"]
                    return AIMessage(content=content)
                raise Exception(f"invalid doubao response: {result}")
            except requests.exceptions.Timeout as e:
                last_err = e
                if attempt == max_retries - 1:
                    raise
                delay = retry_delay * (2 ** attempt)
                await asyncio.sleep(delay)
            except requests.exceptions.RequestException as e:
                last_err = e
                raise
            except Exception as e:
                last_err = e
                raise

class DoubaoEmbeddings:
    def __init__(self, api_key: str, endpoint: str, model: str):
        self.api_key = api_key
        self.endpoint = endpoint
        self.model = model
    def _parse_embeddings(self, data: Any) -> List[List[float]]:
        if isinstance(data, list):
            return [item.get("embedding", []) for item in data]
        if isinstance(data, dict) and "embedding" in data:
            return [data.get("embedding", [])]
        return []
    def _embed(self, inputs: Union[List[str], List[Dict[str, Any]]]) -> List[List[float]]:
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        is_multimodal = "multimodal" in (self.endpoint or "")
        if is_multimodal:
            if inputs and isinstance(inputs[0], dict):
                payload_input = inputs
            else:
                payload_input = [{"type": "text", "text": t} for t in inputs]  # type: ignore[arg-type]
            payload = {"model": self.model, "encoding_format": "float", "input": payload_input}
        else:
            payload = {"model": self.model, "encoding_format": "float", "input": inputs}
        resp = requests.post(self.endpoint, json=payload, headers=headers, timeout=(15, 60))
        if resp.status_code == 401:
            raise RuntimeError("authentication failed for doubao embeddings")
        resp.raise_for_status()
        body = resp.json()
        return self._parse_embeddings(body.get("data"))
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)
    def embed_query(self, text: str) -> List[float]:
        vecs = self._embed([text])
        return vecs[0] if vecs else []
    def embed_multimodal(self, inputs: List[Dict[str, Any]]) -> List[List[float]]:
        return self._embed(inputs)
