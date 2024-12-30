from typing import List, Dict

class ContextManager:
    def __init__(self, max_context_length=2000):
        self.max_context_length = max_context_length
        self.context_cache = {}
        
    def build_context(self, user_id: str, query: str, memories: List[Dict], 
                     recent_messages: List[Dict]) -> str:
        """构建优化的上下文"""
        # 1. 动态上下文长度控制
        available_length = self.max_context_length
        
        # 2. 上下文分层
        context_parts = {
            'critical': [],    # 关键信息（高相关性记忆）
            'relevant': [],    # 相关信息（中等相关性记忆）
            'background': [],  # 背景信息（低相关性记忆）
            'recent': []       # 最近对话
        }
        
        # 3. 智能分配空间
        for memory in memories:
            score = memory.get('relevance_score', 0)
            content = memory.get('content', '')
            
            if score > 0.8:
                context_parts['critical'].append(content)
            elif score > 0.6:
                context_parts['relevant'].append(content)
            else:
                context_parts['background'].append(content)
        
        # 4. 构建最终上下文
        final_context = []
        priority_order = ['critical', 'relevant', 'recent', 'background']
        
        for part in priority_order:
            items = context_parts[part]
            for item in items:
                if len('\n'.join(final_context)) + len(item) < available_length:
                    final_context.append(item)
                else:
                    break
                    
        return '\n'.join(final_context)