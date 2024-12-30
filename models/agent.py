import logging

class ChatAgent:
    def __init__(self, model, roles_config, default_roles, task_manager=None, classification_model=None, rag_module=None):
        self.model = model
        self.roles_config = roles_config
        self.task_manager = task_manager
        self.classification_model = classification_model
        self.user_contexts = {}
        self.user_roles = {}
        self.rag_module = rag_module
        
        # 初始化默认角色
        for user_id, role_info in default_roles.items():
            if isinstance(role_info, dict) and 'role' in role_info:
                role_name = role_info['role']
                if role_name in roles_config:
                    self.user_roles[user_id] = role_name
                else:
                    raise ValueError(f"Default role '{role_name}' is not defined in roles configuration.")
            else:
                logging.warning(f"Invalid role_info format for user {user_id}")

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

    def chat(self, user_id, user_input):
        """处理对话并通过上下文管理角色。"""
        user_context = self.user_contexts.get(user_id, [])

        # 获取当前角色的提示词
        role_name = self.user_roles.get(user_id)
        role_prompt = None
        if role_name:
            role_config = self.roles_config.get(role_name, {})
            role_prompt = role_config.get("prompt", "")

        try:
            # 使用RAG模块生成响应（如果可用）
            if self.rag_module:
                assistant_message = self.rag_module.generate_response(
                    user_input, 
                    self.model,
                    role_prompt=role_prompt
                )
            else:
                # 使用原有的响应生成逻辑
                if role_prompt:
                    if not user_context:
                        user_context.append({"role": "system", "content": role_prompt})
                user_context.append({"role": "user", "content": user_input})
                response = self.model.generate_response(user_input, user_context)
                if isinstance(response, dict) and 'choices' in response:
                    assistant_message = response['choices'][0]['message']['content']
                elif isinstance(response, str):
                    assistant_message = response
                else:
                    raise ValueError("Unexpected response format from LLM.")

            user_context.append({"role": "assistant", "content": assistant_message})
            self.user_contexts[user_id] = user_context[-10:]  # 保留最近10轮对话
            return assistant_message

        except Exception as e:
            logging.error(f"Error in chat: {str(e)}")
            return "抱歉，处理您的请求时出现错误。"

    def set_user_role(self, user_id, role):
        """Set the role for a user and clear their conversation history."""
        if role not in self.roles_config:
            raise ValueError(f"Invalid role: {role}")
        
        # 设置新角色
        self.user_roles[user_id] = role
        
        # 清空该用户的对话上下文
        if user_id in self.user_contexts:
            self.user_contexts[user_id] = []
            logging.info(f"Cleared conversation history for user {user_id} after role change to {role}")
        
        return True
