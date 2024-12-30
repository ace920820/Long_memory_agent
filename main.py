from flask import Flask, render_template, request, jsonify
from models.llm_model import LLMModel
from models.agent import ChatAgent
from services.llm_service import LLMService
from models.rag_module import RAGModule
from models.memory_manager import MemoryManager
import yaml
import logging.config
import sys
import logging
import json

# 设置控制台输出为 UTF-8 编码
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

def create_memory_manager():
    """创建并初始化 MemoryManager 实例"""
    try:
        manager = MemoryManager(
            model_name="all-MiniLM-L6-v2",
            memory_file="config/user_memories.json",
            similarity_threshold=0.55
        )
        
        # 验证 get_all_memories 方法是否存在
        if not hasattr(manager, 'get_all_memories'):
            raise AttributeError("MemoryManager missing required method: get_all_memories")
        
        # 打印调试信息
        print("Memory manager initialized successfully")
        print("Available methods:", [m for m in dir(manager) if not m.startswith('_')])
        
        return manager
        
    except Exception as e:
        print(f"Error initializing memory manager: {str(e)}")
        raise

def create_app():
    app = Flask(__name__)
    
    # Load configuration
    with open("config/config.yaml", "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)

    roles_config = config.get("roles", {})
    default_roles = config.get("default_roles", {})

    # Initialize components
    llm_model = LLMModel(config)
    rag_module = RAGModule(similarity_threshold=0.55)
    memory_manager = create_memory_manager()

    # 添加示例文档
    documents = [
        "圣诞老人是一个传统的节日人物，他在圣诞夜乘坐驯鹿雪橇给孩子们送礼物。",
        "驯鹿是圣诞老人的好帮手，最著名的是红鼻子驯鹿鲁道夫。",
        "V认为117咖啡没有手冲咖啡好喝，但是比红茶好喝",
        "Jamie最喜欢的人是他的老婆和多米",
        "Jamie是这样一个人：是一位充满探索精神和求知欲的人，尤其在技术领域展现出非凡的好奇心与专注力。"
    ]
    rag_module.add_documents(documents)

    chat_agent = ChatAgent(
        llm_model, 
        roles_config, 
        default_roles, 
        rag_module=rag_module,
        memory_manager=memory_manager
    )
    llm_service = LLMService(chat_agent)

    # Initialize logging
    with open("config/logger_config.yaml", "r") as f:
        log_config = yaml.safe_load(f)
    logging.config.dictConfig(log_config)

    # 路由定义
    @app.route('/')
    def home():
        return render_template('index.html')

    @app.route('/memory-manager')
    def memory_manager_page():
        return render_template('memory_manager.html')

    @app.route('/chat', methods=['POST'])
    def chat():
        try:
            user_input = request.json.get('user_input', '')
            user_id = request.json.get('user_id', 'default_user')

            if not user_input.strip():
                return jsonify({"error": "Input cannot be empty"}), 400

            if user_input.lower() == 'exit':
                return jsonify({"response": "Goodbye!"})

            if user_id not in chat_agent.user_roles:
                default_role = next(iter(default_roles.values())).get('role', 'reindeer')
                chat_agent.set_user_role(user_id, default_role)

            result = llm_service.handle_query(user_id, user_input)
            return jsonify(result if isinstance(result, dict) else {"response": result})

        except Exception as e:
            logging.error(f"Error in chat endpoint: {str(e)}")
            return jsonify({"error": "Internal server error"}), 500

    @app.route('/set_role', methods=['POST'])
    def set_role():
        try:
            user_id = request.json.get('user_id')
            role = request.json.get('role')

            if not user_id or not role:
                return jsonify({"error": "Missing user_id or role", "success": False}), 400

            if role not in roles_config:
                return jsonify({"error": f"Invalid role: {role}", "success": False}), 400

            chat_agent.set_user_role(user_id, role)
            return jsonify({
                "success": True,
                "message": f"Role set to {role} for user {user_id}",
                "role_name": roles_config[role].get('name', role)
            })

        except Exception as e:
            logging.error(f"Error in set_role endpoint: {str(e)}")
            return jsonify({"error": "Internal server error", "success": False}), 500

    @app.route('/api/memories', methods=['GET'])
    def get_memories():
        try:
            user_id = request.args.get('user_id', 'default_user')
            memories = memory_manager.get_all_memories(user_id)
            return jsonify({"memories": memories})
        except Exception as e:
            logging.error(f"Error getting memories: {str(e)}")
            return jsonify({"error": "Failed to get memories"}), 500

    @app.route('/api/memories', methods=['POST'])
    def add_memory():
        try:
            data = request.json
            user_id = data.get('user_id', 'default_user')
            content = data.get('content')
            memory_type = data.get('type', 'general')
            
            result = memory_manager.add_memory(user_id, content, memory_type)
            return jsonify(result)
        except Exception as e:
            logging.error(f"Error adding memory: {str(e)}")
            return jsonify({"error": "Failed to add memory"}), 500

    @app.route('/api/memories/restructure-preview', methods=['POST'])
    def preview_restructure():
        try:
            data = request.json
            memory_ids = data.get('memoryIds', [])
            template = data.get('template')
            user_id = data.get('user_id', 'default_user')
            
            preview = memory_manager.preview_restructure(user_id, memory_ids, template)
            return jsonify({"preview": preview})
        except Exception as e:
            logging.error(f"Error previewing restructure: {str(e)}")
            return jsonify({"error": "Failed to preview restructure"}), 500

    @app.route('/api/memories/restructure', methods=['POST'])
    def restructure_memories():
        try:
            data = request.json
            memory_ids = data.get('memoryIds', [])
            template = data.get('template')
            user_id = data.get('user_id', 'default_user')
            
            result = memory_manager.restructure_memories(user_id, memory_ids, template)
            return jsonify(result)
        except Exception as e:
            logging.error(f"Error restructuring memories: {str(e)}")
            return jsonify({"error": "Failed to restructure memories"}), 500

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)