from models.llm_api_wrapper import LLMAPIWrapper

class LLMModel:
    def __init__(self, config):
        self.api_wrapper = LLMAPIWrapper(api_key=config["api_key"], model=config["llm_model"])
        self.model = config["llm_model"]

    def generate_response(self, prompt, context):
        """根据输入的 prompt 和上下文生成响应。"""
        # context 是消息列表，直接传递给 API
        if not isinstance(context, list):
            raise ValueError("Context must be a list of message dictionaries.")
        return self.api_wrapper.call_llm(context + [{"role": "user", "content": prompt}])


