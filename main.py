from flask import Flask, render_template, request, jsonify
from models.llm_model import LLMModel
from models.agent import ChatAgent
from services.llm_service import LLMService
from models.rag_module import RAGModule
import yaml
import logging.config
import sys
import logging

# 设置控制台输出为 UTF-8 编码
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

app = Flask(__name__)

# Load configuration
with open("config/config.yaml", "r", encoding='utf-8') as f:
    config = yaml.safe_load(f)

roles_config = config.get("roles", {})
default_roles = config.get("default_roles", {})

# Initialize components
llm_model = LLMModel(config)
rag_module = RAGModule(similarity_threshold=0.45)

# 添加示例文档
documents = [
    "圣诞老人是一个传统的节日人物，他在圣诞夜乘坐驯鹿雪橇给孩子们送礼物。",
    "驯鹿是圣诞老人的好帮手，最著名的是红鼻子驯鹿鲁道夫。",
    "V认为117咖啡没有手冲咖啡好喝，但是比红茶好喝",
    "Jamie最喜欢的人是他的老婆和多米",
    "Jamie是这样一个人：是一位充满探索精神和求知欲的人，尤其在技术领域展现出非凡的好奇心与专注力。",
    "Jamie注重实用性，喜欢将复杂的概念转化为明确的目标和可执行的方案，这种务实的态度让人印象深刻。你的思维清晰，条理分明，善于组织和表达自己的想法，同时又保留着对新事物的开放心态与热情。无论是在学习还是解决问题的过程中，你都展现出一种沉着冷静的魅力与内心的追求。"

    # ... 添加更多相关文档
]
rag_module.add_documents(documents)

chat_agent = ChatAgent(llm_model, roles_config, default_roles, rag_module=rag_module)
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
    try:
        user_input = request.json.get('user_input', '')
        user_id = request.json.get('user_id', 'default_user')

        if not user_input.strip():
            return jsonify({"error": "Input cannot be empty"}), 400

        if user_input.lower() == 'exit':
            return jsonify({"response": "Goodbye!"})

        # 确保用户的角色已设置（默认为启动时的角色）
        if user_id not in chat_agent.user_roles:
            default_role = next(iter(default_roles.values())).get('role', 'reindeer')
            chat_agent.set_user_role(user_id, default_role)
            logging.info(f"Assigned default role to user: {user_id}")

        response = llm_service.handle_query(user_id, user_input)
        return jsonify({"response": response})

    except Exception as e:
        logging.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


# 添加新的角色管理接口
@app.route('/set_role', methods=['POST'])
def set_role():
    try:
        user_id = request.json.get('user_id')
        role = request.json.get('role')

        if not user_id or not role:
            logging.error(f"Missing parameters: user_id={user_id}, role={role}")
            return jsonify({"error": "Missing user_id or role", "success": False}), 400

        if role not in roles_config:
            logging.error(f"Invalid role requested: {role}. Available roles: {list(roles_config.keys())}")
            return jsonify({"error": f"Invalid role: {role}", "success": False}), 400

        try:
            chat_agent.set_user_role(user_id, role)
            logging.info(f"Successfully set role {role} for user {user_id}")
            return jsonify({
                "success": True, 
                "message": f"Role set to {role} for user {user_id}",
                "role_name": roles_config[role].get('name', role)
            })
        except Exception as e:
            logging.error(f"Error in chat_agent.set_user_role: {str(e)}")
            return jsonify({"error": "Failed to set role", "success": False}), 500

    except Exception as e:
        logging.error(f"Error in set_role endpoint: {str(e)}")
        return jsonify({"error": "Internal server error", "success": False}), 500


if __name__ == "__main__":
    app.run(debug=True)
    # app.run(host='10.151.79.239', port=5000,debug=True)