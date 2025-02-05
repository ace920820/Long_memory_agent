from models.llm_api_wrapper import LLMAPIWrapper
from services.prompt_manager import PromptManager
import logging
import os
from datetime import datetime

class LLMModel:
    def __init__(self, config=None):
        config["api_key"] =  os.getenv("LLM_API_KEY")
        self.config = config
        self.model=  os.getenv("LLM_MODEL")
        if self.model[:3] == "gpt":
            self.base_url = "https://api.gptsapi.net/v1"
        elif self.model[:8] == "deepseek":
            self.base_url = 'https://api.deepseek.com'
        elif self.model[:4] == "qwen":
            self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.api_wrapper = LLMAPIWrapper(api_key=config["api_key"], model=self.model,base_url=self.base_url)

        
        # 初始化提示词管理器
        self.prompt_manager = PromptManager()
        self.current_role_prompt = None  # 当前角色的提示词
        
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

    def set_role(self, role_name: str) -> bool:
        """设置当前角色"""
        try:
            self.current_role_prompt = self.prompt_manager.get_role_prompt(role_name)
            return True if self.current_role_prompt else False
        except Exception as e:
            logging.error(f"Error setting role: {str(e)}")
            return False

    def _process_response(self, response: dict) -> str:
        """处理 API 响应"""
        try:
            if isinstance(response, dict) and 'choices' in response:
                return response['choices'][0]['message']['content']
            elif isinstance(response, str):
                return response
            else:
                logging.warning(f"Unexpected response format: {response}")
                return str(response)
        except Exception as e:
            logging.error(f"Error processing response: {str(e)}")
            return "抱歉，处理响应时出现错误。"

    def generate_response(self, prompt: str, context: dict = None) -> str:
        """生成响应并记录详细日志"""
        if context is None:
            context = {}
        
        try:
            # 生成请求ID用于日志追踪
            request_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            start_time = datetime.now()
            
            # 获取系统指令
            system_instruction = self.prompt_manager.get_system_instruction(
                template_type='base',
                role_prompt=self.current_role_prompt,
                memories=context.get('memories', []),
                user_input=prompt
            )
            
            # 构建消息列表
            messages = [
                {"role": "system", "content": system_instruction}
            ]
            
            # 添加对话历史（如果有）
            if 'chat_history' in context:
                messages.extend(context['chat_history'])
            
            # 添加当前用户输入
            messages.append({"role": "user", "content": prompt})
            
            # 记录请求开始
            self.llm_logger.info("开始 LLM 请求", extra={
                'request_id': request_id,
                'prompt': str(messages),
                'response': '',
                'tokens': 0,
                'latency': 0
            })
            
            # 调用 API
            response = self.api_wrapper.call_llm(messages,model=self.model)
            result = self._process_response(response)
            
            # 计算延迟和token数
            end_time = datetime.now()
            latency = (end_time - start_time).total_seconds() * 1000
            tokens = len(str(messages).split()) + len(result.split())  # 简单估算
            
            # 记录完整的请求信息
            self.llm_logger.info("LLM 请求完成", extra={
                'request_id': request_id,
                'prompt': str(messages),
                'response': result,
                'tokens': tokens,
                'latency': latency
            })
            
            return result
            
        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            return "抱歉，生成回答时出现错误。"

    def generate(self, prompt: str, **kwargs) -> str:
        """兼容性方法，调用 generate_response"""
        return self.generate_response(prompt, kwargs.get('context'))


