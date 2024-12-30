from models.llm_api_wrapper import LLMAPIWrapper
import logging
import os
from datetime import datetime

class LLMModel:
    def __init__(self, config=None):
        self.config = config
        self.api_wrapper = LLMAPIWrapper(api_key=config["api_key"], model=config["llm_model"])
        self.model = config["llm_model"]
        # 设置专门的 LLM 调用日志记录器
        self.llm_logger = logging.getLogger('llm_calls')
        self.setup_llm_logger()
    
    def setup_llm_logger(self):
        """设置专门的 LLM 调用日志"""
        # 创建 logs 目录（如果不存在）
        if not os.path.exists('logs'):
            os.makedirs('logs')
            
        # 设置文件处理器
        fh = logging.FileHandler('logs/llm_api.log', encoding='utf-8')
        fh.setLevel(logging.INFO)
        
        # 设置格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s\n'
            'Request ID: %(request_id)s\n'
            'Prompt:\n%(prompt)s\n'
            'Response:\n%(response)s\n'
            'Tokens: %(tokens)d\n'
            'Latency: %(latency).2f ms\n'
            '-' * 80
        )
        fh.setFormatter(formatter)
        
        # 添加处理器到记录器
        self.llm_logger.addHandler(fh)
        self.llm_logger.setLevel(logging.INFO)

    def generate_response(self, prompt: str, context: list = None) -> str:
        """生成响应并记录详细日志
        :param prompt: 提示文本
        :param context: 对话上下文列表，默认为空列表
        :return: 生成的响应文本
        """
        if context is None:
            context = []
        
        try:
            # 生成请求ID
            request_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            
            # 记录请求开始
            start_time = datetime.now()
            self.llm_logger.info("开始 LLM 请求", extra={
                'request_id': request_id,
                'prompt': prompt,
                'response': '',
                'tokens': 0,
                'latency': 0
            })
            
            # 使用 api_wrapper 调用 LLM
            full_prompt = context + [{"role": "user", "content": prompt}]
            response = self.api_wrapper.call_llm(full_prompt)
            
            # 处理响应
            if isinstance(response, dict) and 'choices' in response:
                result = response['choices'][0]['message']['content']
            elif isinstance(response, str):
                result = response
            else:
                logging.warning(f"Unexpected response format: {response}")
                result = str(response)
            
            # 计算延迟和token数
            end_time = datetime.now()
            latency = (end_time - start_time).total_seconds() * 1000
            tokens = len(str(full_prompt).split()) + len(result.split())  # 简单估算
            
            # 记录完整的请求信息
            self.llm_logger.info("LLM 请求完成", extra={
                'request_id': request_id,
                'prompt': str(full_prompt),
                'response': result,
                'tokens': tokens,
                'latency': latency
            })
            
            return result
            
        except Exception as e:
            # 记录错误信息
            self.llm_logger.error(f"LLM 请求失败: {str(e)}", extra={
                'request_id': request_id if 'request_id' in locals() else 'unknown',
                'prompt': str(full_prompt) if 'full_prompt' in locals() else prompt,
                'response': str(e),
                'tokens': 0,
                'latency': 0
            })
            logging.error(f"Error generating response: {str(e)}")
            return "抱歉，生成回答时出现错误。"

    def generate(self, prompt: str, **kwargs) -> str:
        """兼容性方法，调用 generate_response"""
        return self.generate_response(prompt, kwargs.get('context'))


