import logging
from models.agent import ChatAgent
from services.context_manager import ContextManager
import yaml

class LLMService:
    def __init__(self, chat_agent):
        self.chat_agent = chat_agent

    def handle_query(self, user_id: str, query: str) -> dict:
        """处理用户查询
        
        Args:
            user_id: 用户ID
            query: 用户输入的查询
            
        Returns:
            dict: 包含响应的字典
        """
        try:
            # 使用 chat_agent 处理查询
            result = self.chat_agent.chat(user_id, query)
            
            # 确保返回格式正确
            if isinstance(result, dict):
                return result
            else:
                return {"response": str(result)}
                
        except Exception as e:
            logging.error(f"Error in handle_query: {str(e)}")
            return {
                "error": "处理查询时出现错误",
                "details": str(e)
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
