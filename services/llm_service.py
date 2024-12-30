import logging
from models.agent import ChatAgent
from services.context_manager import ContextManager
import yaml

class LLMService:
    def __init__(self, agent):
        self.agent = agent
        self.context_manager = ContextManager()

    def handle_query(self, user_id, user_input):
        """Handle user query, format response, and manage context."""
        # Retrieve user-specific context
        raw_response = self.agent.chat(user_id, user_input)

        # Log raw response for debugging
        logger = logging.getLogger("my_app")
        logger.debug(f"Raw API response: {raw_response}")

        # Handle different response formats
        if isinstance(raw_response, dict) and 'choices' in raw_response:
            # Extract the assistant's response content
            response_content = raw_response['choices'][0]['message']['content']
        elif isinstance(raw_response, str):
            # If the response is a plain string
            response_content = raw_response
        else:
            # Fallback for unexpected response formats
            logger.error(f"Unexpected response format: {raw_response}")
            response_content = "对不起，我无法理解当前的响应格式。"

        # Update context with the latest interaction
        self.context_manager.update_context(user_id, user_input, response_content)

        return response_content
    def set_user_role(self, user_id, role):
        """Set the role for a user."""
        roles_config = self.load_roles_config()
        self.agent.set_role(user_id, role, roles_config)

    def load_roles_config(self):
        """Load roles configuration."""
        with open("config/config.yaml", "r") as f:
            config = yaml.safe_load(f)
        return config.get("roles", {})
