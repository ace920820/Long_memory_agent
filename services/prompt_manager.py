import yaml
import os
from typing import Dict, Optional, List
from jinja2 import Environment, BaseLoader
import re

class PromptManager:
    def __init__(self, prompts_dir: str = "prompts"):
        self.prompts_dir = prompts_dir
        self.config = self._load_config()
        self.roles = self._load_roles()
        self.templates = self._load_templates()
        self.jinja_env = self._setup_jinja_environment()
        self.system_templates = self._load_system_templates()
    
    def _setup_jinja_environment(self):
        """设置 Jinja2 环境和自定义过滤器"""
        env = Environment(loader=BaseLoader())
        
        # 添加相关性评分过滤器
        env.filters['relevance_score'] = self._calculate_relevance
        env.filters['any_relevant'] = self._any_relevant
        env.filters['filter_relevant'] = self._filter_relevant
        env.filters['recent_relevant'] = self._recent_relevant
        env.filters['get_relevant_history'] = self._get_relevant_history
        
        return env
    
    def _calculate_relevance(self, text: str, query: str) -> float:
        """计算文本与查询的相关性分数"""
        # 这里可以使用更复杂的相关性计算方法
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())
        overlap = query_words.intersection(text_words)
        return len(overlap) / max(len(query_words), 1)
    
    def _any_relevant(self, memories: List[str], query: str, threshold: float = 0.3) -> bool:
        """检查是否有任何相关记忆"""
        return any(self._calculate_relevance(m, query) > threshold for m in memories)
    
    def _filter_relevant(self, memories: List[str], query: str, threshold: float = 0.3) -> List[str]:
        """过滤出相关记忆"""
        return [m for m in memories if self._calculate_relevance(m, query) > threshold]
    
    def _recent_relevant(self, history: str, query: str, threshold: float = 0.3) -> bool:
        """检查最近的对话是否相关"""
        recent_turns = history.split('\n')[-6:]  # 最近3轮对话
        return any(self._calculate_relevance(turn, query) > threshold for turn in recent_turns)
    
    def _get_relevant_history(self, history: str, query: str, threshold: float = 0.3) -> str:
        """获取相关的对话历史"""
        turns = history.split('\n')
        relevant_turns = [turn for turn in turns if self._calculate_relevance(turn, query) > threshold]
        return '\n'.join(relevant_turns[-6:])  # 最多返回3轮相关对话
    
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
    
    def _load_system_templates(self) -> Dict:
        """加载系统指令模板"""
        system_template_path = os.path.join(self.prompts_dir, "templates", "system.yaml")
        with open(system_template_path, "r", encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def get_role_prompt(self, role_name: str) -> Optional[str]:
        """获取角色的基础提示词"""
        role = self.roles.get(role_name)
        return role["base_prompt"] if role else None
    
    def format_prompt(self, template_name: str, **kwargs) -> str:
        """使用 Jinja2 格式化提示词模板"""
        template_data = self.templates.get(template_name)
        if not template_data:
            raise ValueError(f"Template {template_name} not found")
            
        # 选择合适的模板变体
        template_str = template_data.get('base')
        if not kwargs.get('context') and not kwargs.get('memories') and not kwargs.get('chat_history'):
            template_str = template_data.get('simple', template_str)
            
        template = self.jinja_env.from_string(template_str)
        return template.render(**kwargs)
    
    def get_system_prompt(self, prompt_name: str) -> str:
        """获取系统提示词"""
        return self.config["system_prompts"].get(prompt_name, "") 
    
    def get_system_instruction(self, template_type: str = 'base', **kwargs) -> str:
        """获取格式化的系统指令
        
        Args:
            template_type: 系统指令类型 ('base', 'memory_manager', 'conversation_manager', 'rag_enhancement')
            **kwargs: 模板变量
        """
        if template_type not in self.system_templates['system']:
            raise ValueError(f"Unknown system template type: {template_type}")
            
        template_str = self.system_templates['system'][template_type]
        template = self.jinja_env.from_string(template_str)
        
        # 添加自定义过滤器
        def sort_by_relevance(memories, query=kwargs.get('user_input', '')):
            return sorted(
                memories,
                key=lambda x: self._calculate_relevance(x, query),
                reverse=True
            )
        
        self.jinja_env.filters['sort_by_relevance'] = sort_by_relevance
        
        return template.render(**kwargs) 