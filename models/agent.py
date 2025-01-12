import logging

class ChatAgent:
    def __init__(self, llm_model, roles_config, default_roles, rag_module=None, memory_manager=None):
        """初始化聊天代理。
        
        Args:
            llm_model: LLM模型实例
            roles_config: 角色配置
            default_roles: 默认角色设置
            rag_module: RAG模块实例（可选）
            memory_manager: 记忆管理器实例（可选）
        """
        # 修改属性名，确保与其他地方一致
        self.llm_model = llm_model  # 将 llm 改为 llm_model
        self.roles_config = roles_config
        self.user_roles = default_roles.copy()
        self.user_contexts = {}
        self.rag_module = rag_module
        self.memory_manager = memory_manager

    def set_role(self, user_id, role, roles_config):
        """Set the role for a user."""
        if role in roles_config:
            self.user_roles[user_id] = roles_config[role]
        else:
            raise ValueError(f"Role {role} is not defined in the configuration.")

    def determine_strategy(self, user_input):
        """Analyze user input and decide response strategy."""
        if "帮助" in user_input or "help" in user_input.lower():
            return "guidance"
        elif "天气" in user_input:
            return "weather"
        else:
            return "general"

    def classify_intent(self, user_input):
        """Classify the user's intent using the classification model."""
        if self.classification_model:
            return self.classification_model.predict(user_input)
        return "general"

    def update_context(self, user_id, user_input, assistant_response):
        """更新上下文，并避免重复角色信息。"""
        if user_id not in self.user_contexts:
            self.user_contexts[user_id] = []

        new_entry = f"User: {user_input}\nAssistant: {assistant_response}"
        # 避免重复角色上下文
        if new_entry not in self.user_contexts[user_id]:
            self.user_contexts[user_id].append(new_entry)

        # 保留最近5轮对话
        if len(self.user_contexts[user_id]) > 5:
            self.user_contexts[user_id] = self.user_contexts[user_id][-5:]

    def handle_task(self, user_input):
        """Process task-related inputs."""
        if self.task_manager:
            if "设置提醒" in user_input:
                return self.task_manager.create_reminder(user_input)
            elif "查找" in user_input:
                return self.task_manager.search(user_input)
        return None

    def summarize_context(self, user_id):
        """Summarize the context when it's too long."""
        context = "\n".join(self.user_contexts.get(user_id, []))
        if len(context) > 1000:  # Example threshold
            prompt = f"Summarize the following conversation:\n{context}"
            summary = self.model.generate_response(prompt)
            self.user_contexts[user_id] = [summary]

    def chat(self, user_id: str, user_input: str) -> dict:
        """处理用户输入并返回响应"""
        try:
            # 确保用户有角色设置
            if user_id not in self.user_roles:
                raise ValueError(f"No role set for user {user_id}")

            # 1. 获取知识库相关文档
            relevant_docs = []
            if self.rag_module:
                relevant_docs = self.rag_module.search(user_input, top_k=5)
                logging.debug(f"Retrieved documents: {relevant_docs}")

            # 2. 获取相关记忆
            memories = []
            if self.memory_manager:
                memories = self.memory_manager.retrieve_memories(user_id, user_input)
                logging.debug(f"Retrieved memories: {memories}")

            # 3. 构建上下文
            context = {
                'chat_history': self.user_contexts.get(user_id, []),
                'memories': [memory['content'] for memory in memories] if memories else [],
                'context': [doc['matched_chunks'] for doc in relevant_docs] if relevant_docs else []
            }

            # 4. 生成回答
            assistant_message = self.rag_module.generate_response(user_input,self.llm_model, context)

            # 5. 更新对话历史
            if user_id not in self.user_contexts:
                self.user_contexts[user_id] = []
            
            self.user_contexts[user_id].extend([
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": assistant_message}
            ])

            # 保持对话历史在合理长度
            self.user_contexts[user_id] = self.user_contexts[user_id][-10:]  # 保留最近10轮对话

            # 6. 存储新的记忆
            memory_status = None
            if self.memory_manager:
                memory_status = self.memory_manager.add_memory(
                    user_id, 
                    f"用户说: {user_input}\n助手回答: {assistant_message}",
                    "dialogue"
                )

            # 7. 记录调试信息
            logging.debug(f"""
            Chat details:
            User: {user_id}
            Input: {user_input}
            Retrieved docs: {len(relevant_docs) if relevant_docs else 0}
            Retrieved memories: {len(memories) if memories else 0}
            Context length: {len(context['chat_history'])}
            Response length: {len(assistant_message)}
            """)

            return {
                "response": assistant_message,
                "memory_status": memory_status
            }

        except Exception as e:
            logging.error(f"Error in chat: {str(e)}")
            return {
                "response": "抱歉，处理您的请求时出现错误。",
                "memory_status": None
            }

    def set_user_role(self, user_id: str, role: str) -> bool:
        """Set the role for a user."""
        try:
            if role not in self.roles_config:
                raise ValueError(f"Invalid role: {role}")
            
            # 设置新角色
            self.user_roles[user_id] = role
            
            # 更新 LLM 模型的角色设置
            if not self.llm_model.set_role(role):
                raise ValueError(f"Failed to set role {role} in LLM model")
            
            # 清空该用户的对话上下文
            if user_id in self.user_contexts:
                self.user_contexts[user_id] = []
                logging.info(f"Cleared conversation history for user {user_id} after role change to {role}")
            
            return True
            
        except Exception as e:
            logging.error(f"Error in set_user_role: {str(e)}")
            raise
