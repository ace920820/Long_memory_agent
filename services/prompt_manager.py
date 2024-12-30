import yaml
import os
from typing import Dict, Optional

class PromptManager:
    def __init__(self, prompts_dir: str = "prompts"):
        self.prompts_dir = prompts_dir
        self.config = self._load_config()
        self.roles = self._load_roles()
        self.templates = self._load_templates()
    
    def _load_config(self) -> Dict:
        """加载主配置文件"""
        with open(os.path.join(self.prompts_dir, "config.yaml"), "r", encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _load_roles(self) -> Dict:
        """加载所有角色提示词"""
        roles = {}
        roles_path = os.path.join(self.prompts_dir, "roles")
        for filename in os.listdir(roles_path):
            if filename.endswith(".yaml"):
                with open(os.path.join(roles_path, filename), "r", encoding='utf-8') as f:
                    role_data = yaml.safe_load(f)
                    roles[role_data["name"]] = role_data
        return roles
    
    def _load_templates(self) -> Dict:
        """加载所有模板"""
        templates = {}
        templates_path = os.path.join(self.prompts_dir, "templates")
        for filename in os.listdir(templates_path):
            if filename.endswith(".yaml"):
                with open(os.path.join(templates_path, filename), "r", encoding='utf-8') as f:
                    template_data = yaml.safe_load(f)
                    templates.update(template_data)
        return templates
    
    def get_role_prompt(self, role_name: str) -> Optional[str]:
        """获取角色的基础提示词"""
        role = self.roles.get(role_name)
        return role["base_prompt"] if role else None
    
    def format_prompt(self, template_name: str, **kwargs) -> str:
        """格式化提示词模板"""
        template = self.templates.get(template_name)
        if not template:
            raise ValueError(f"Template {template_name} not found")
        return template.format(**kwargs)
    
    def get_system_prompt(self, prompt_name: str) -> str:
        """获取系统提示词"""
        return self.config["system_prompts"].get(prompt_name, "") 