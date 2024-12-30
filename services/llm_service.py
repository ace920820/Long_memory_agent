import logging
from models.agent import ChatAgent
from services.context_manager import ContextManager
import yaml

class LLMService:
    def __init__(self, chat_agent):
        self.chat_agent = chat_agent

    def handle_query(self, user_id: str, query: str) -> dict:
        """处理用户查询
        :param user_id: 用户ID
        :param query: 用户输入
        :return: 包含响应和记忆状态的字典
        """
        try:
            result = self.chat_agent.chat(user_id, query)
            
            # 如果返回的是字典（新格式）
            if isinstance(result, dict):
                return result
            
            # 如果返回的是字符串（旧格式），转换为新格式
            return {
                "response": result,
                "memory_status": None
            }
            
        except Exception as e:
            logging.error(f"Error in handle_query: {str(e)}")
            return {
                "error": "处理请求时出现错误",
                "memory_status": None
            }

    def set_user_role(self, user_id, role):
        """Set the role for a user."""
        roles_config = self.load_roles_config()
        self.chat_agent.set_role(user_id, role, roles_config)

    def load_roles_config(self):
        """Load roles configuration."""
        with open("config/config.yaml", "r") as f:
            config = yaml.safe_load(f)
        return config.get("roles", {})
