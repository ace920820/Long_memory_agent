from models.llm_api_wrapper import LLMAPIWrapper
import logging

class LLMModel:
    def __init__(self, config):
        self.api_wrapper = LLMAPIWrapper(api_key=config["api_key"], model=config["llm_model"])
        self.model = config["llm_model"]

    def generate_response(self, prompt: str, context: list = None) -> str:
        """生成响应
        :param prompt: 提示文本
        :param context: 对话上下文列表，默认为空列表
        :return: 生成的响应文本
        """
        if context is None:
            context = []
        
        try:
            # 使用 api_wrapper 而不是直接使用 model
            response = self.api_wrapper.call_llm(context + [{"role": "user", "content": prompt}])
            
            # 根据响应类型处理返回值
            if isinstance(response, dict) and 'choices' in response:
                return response['choices'][0]['message']['content']
            elif isinstance(response, str):
                return response
            else:
                logging.warning(f"Unexpected response format: {response}")
                return str(response)
                
        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            return "抱歉，生成回答时出现错误。"


