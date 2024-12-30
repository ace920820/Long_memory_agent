import os
import requests
from typing import List, Dict, Union, Optional
from pydantic import BaseModel, ValidationError

import logging

class LLMAPIWrapper:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.model = model or os.getenv("DASHSCOPE_MODEL")
        self.base_url = base_url

        # 初始化日志记录器
        self.logger = logging.getLogger("LLMAPIWrapper")
        if not self.api_key:
            raise ValueError("API Key is required. Set it via parameter or environment variable `DASHSCOPE_API_KEY`.")

    def call_model(
            self,
            model: str,
            messages: List[Dict[str, str]],
            temperature: Optional[float] = None,
            top_p: Optional[float] = None,
            max_tokens: Optional[int] = None,
            stream: bool = False,
            stop: Optional[Union[str, List[str]]] = None,
    ) -> Dict:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "parameters": {
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "stop": stop,
            },
        }
        payload["parameters"] = {k: v for k, v in payload["parameters"].items() if v is not None}

        # 打印日志信息
        # self.logger.debug(f"Calling LLM API with payload: {payload}")

        response = requests.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
        if response.status_code != 200:
            self.logger.error(f"API call failed with status {response.status_code}: {response.text}")
            raise RuntimeError(f"API call failed with status {response.status_code}: {response.text}")

        # 打印响应日志
        self.logger.debug(f"LLM API Response: {response.json()}")
        return response.json()

    def call_llm(self, messages: List[Dict[str, str]], model: str = "qwen-plus") -> Dict:
        """调用 LLM，传递消息列表并确保格式正确。"""
        # 验证消息格式
        for msg in messages:
            if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                raise ValueError("Each message must be a dictionary with 'role' and 'content' keys.")
            if not isinstance(msg['content'], str):
                raise ValueError("The 'content' field in each message must be a string.")

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "parameters": {}
        }

        # 打印日志
        self.logger.debug(f"Calling LLM API with payload: {payload}")

        response = self.call_model(model=model, messages=messages)
        self.logger.debug(f"LLM API Response: {response}")
        return response


# Example usage
if __name__ == "__main__":
    try:
        api_wrapper = LLMAPIWrapper(api_key="sk-6423f724e06d488ca246f2425f6707f4")
        response = api_wrapper.call_model(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "你好，你是谁？"}
            ],
            temperature=0.7,
            max_tokens=200,
            stream=False,
        )
        print(response)
    except ValueError as e:
        print(f"Initialization Error: {e}")
    except RuntimeError as e:
        print(f"API Error: {e}")
