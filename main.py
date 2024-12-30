from flask import Flask, render_template, request, jsonify
from models.llm_model import LLMModel
from models.agent import ChatAgent
from services.llm_service import LLMService
import yaml
import logging.config
import sys
import logging

# 设置控制台输出为 UTF-8 编码
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

app = Flask(__name__)

# Load configuration
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

roles_config = config.get("roles", {})
default_roles = config.get("default_roles", {})

# Initialize components
llm_model = LLMModel(config)
chat_agent = ChatAgent(llm_model, roles_config, default_roles)
llm_service = LLMService(chat_agent)


# Initialize logging
with open("config/logger_config.yaml", "r") as f:
    log_config = yaml.safe_load(f)
logging.config.dictConfig(log_config)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('user_input', '')
    user_id = request.json.get('user_id', 'default_user')

    if user_input.lower() == 'exit':
        return jsonify({"response": "Goodbye!"})

    # 确保用户的角色已设置（默认为启动时的角色）
    if user_id not in chat_agent.user_roles:
        return jsonify({"response": "Error: User role is not configured."})

    response = llm_service.handle_query(user_id, user_input)
    return jsonify({"response": response})



if __name__ == "__main__":
    app.run(debug=True)
