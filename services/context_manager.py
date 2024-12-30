class ContextManager:
    def __init__(self, max_context_length=1000):
        """Manage conversation context with length control."""
        self.user_contexts = {}
        self.max_context_length = max_context_length

    def update_context(self, user_id, user_input, assistant_response):
        """Update user-specific context, truncating if necessary."""
        if user_id not in self.user_contexts:
            self.user_contexts[user_id] = ""
        self.user_contexts[user_id] += f"User: {user_input}\nAssistant: {assistant_response}\n"
        if len(self.user_contexts[user_id]) > self.max_context_length:
            self.user_contexts[user_id] = self.user_contexts[user_id][-self.max_context_length:]

    def get_context(self, user_id):
        """Retrieve the context for a specific user."""
        return self.user_contexts.get(user_id, "")