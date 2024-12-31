from flask import Flask, render_template, request, jsonify
from models.llm_model import LLMModel
from models.agent import ChatAgent
from services.llm_service import LLMService
from models.rag_module import RAGModule
from models.memory_manager import MemoryManager
from services.prompt_manager import PromptManager
import yaml
import logging.config
import sys
import logging
import json
import os

# 设置控制台输出为 UTF-8 编码
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

def create_memory_manager():
    """创建并初始化 MemoryManager 实例"""
    try:
        manager = MemoryManager(
            model_name="all-MiniLM-L6-v2",
            memory_file="config/user_memories.json",
            similarity_threshold=0.45
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
    # 确保必要的目录存在
    os.makedirs('logs', exist_ok=True)
    os.makedirs('data/vector_store', exist_ok=True)
    os.makedirs('config', exist_ok=True)
    
    app = Flask(__name__)
    
    # Load configuration
    with open("config/config.yaml", "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 从 prompts 配置加载角色信息
    prompt_manager = PromptManager()
    roles_config = {
        role: prompt_manager.roles[role]
        for role in prompt_manager.config.get('available_roles', [])
    }
    
    # 设置默认角色
    default_roles = {
        'default_user': {
            'role': prompt_manager.config.get('default_role', 'reindeer')
        }
    }

    # Initialize components
    try:
        # 初始化记忆管理器
        memory_manager = MemoryManager(
            model_name="all-MiniLM-L6-v2",  # 使用默认模型
            memory_file="config/user_memories.json",
            similarity_threshold=0.6
        )
        
        # 初始化 RAG 模块
        rag_module = RAGModule(
            model_name="all-MiniLM-L6-v2",
            similarity_threshold=0.6,
            index_path="data/vector_store"
        )
        
        # 初始化 LLM 模型
        llm_model = LLMModel({
            "api_key": config.get("llm", {}).get("api_key", "your_api_key_here"),
            "llm_model": config.get("llm", {}).get("llm_model", "qwen-plus"),
            "temperature": config.get("llm", {}).get("temperature", 0.7),
            "max_tokens": config.get("llm", {}).get("max_tokens", 2000)
        })
        
        # 初始化聊天代理
        chat_agent = ChatAgent(
            llm_model=llm_model,
            roles_config=roles_config,
            default_roles=default_roles,
            rag_module=rag_module,
            memory_manager=memory_manager
        )
        
        # 确保默认用户有默认角色
        default_role = next(iter(default_roles.values())).get('role', 'reindeer')
        chat_agent.set_user_role('default_user', default_role)

        # 初始化 LLM 服务并添加到应用上下文
        app.llm_service = LLMService(chat_agent)
        
    except Exception as e:
        logging.error(f"Error initializing components: {str(e)}")
        raise

    # 设置日志配置
    with open('config/logging_config.yaml', 'r') as f:
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

            # 确保用户有角色设置
            if user_id not in chat_agent.user_roles:
                default_role = next(iter(default_roles.values())).get('role', 'reindeer')
                chat_agent.set_user_role(user_id, default_role)
                logging.info(f"Assigned default role '{default_role}' to user {user_id}")

            # 使用应用上下文中的 llm_service
            result = app.llm_service.handle_query(user_id, user_input)
            return jsonify(result if isinstance(result, dict) else {"response": result})

        except Exception as e:
            logging.error(f"Error in chat endpoint: {str(e)}")
            return jsonify({"error": "Internal server error"}), 500

    @app.route('/set_role', methods=['POST'])
    def set_role():
        try:
            data = request.json
            user_id = data.get('user_id', 'default_user')
            role = data.get('role')
            
            # 获取角色提示词
            role_prompt = prompt_manager.get_role_prompt(role)
            if not role_prompt:
                return jsonify({"error": "Invalid role"}), 400
            
            # 设置用户角色
            chat_agent.set_user_role(user_id, role)
            
            return jsonify({
                "success": True,
                "message": prompt_manager.get_system_prompt("role_switch").format(role=role)
            })
        except Exception as e:
            logging.error(f"Error setting role: {str(e)}")
            return jsonify({"error": "Failed to set role"}), 500

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

    @app.route('/api/memories/delete', methods=['POST'])
    def delete_memories():
        try:
            data = request.json
            memory_ids = data.get('memoryIds', [])
            user_id = data.get('user_id', 'default_user')
            
            result = memory_manager.delete_memories(user_id, memory_ids)
            return jsonify(result)
            
        except Exception as e:
            logging.error(f"Error deleting memories: {str(e)}")
            return jsonify({
                "success": False,
                "error": "Failed to delete memories",
                "details": str(e)
            }), 500

    @app.route('/api/memories/split', methods=['POST'])
    def split_memory():
        try:
            data = request.json
            memory_id = data.get('memoryId')
            user_id = data.get('user_id', 'default_user')
            
            if memory_id is None:
                return jsonify({"success": False, "error": "Memory ID is required"}), 400
            
            result = memory_manager.split_memory(user_id, memory_id)
            return jsonify(result)
            
        except Exception as e:
            logging.error(f"Error splitting memory: {str(e)}")
            return jsonify({
                "success": False,
                "error": "Failed to split memory",
                "details": str(e)
            }), 500

    @app.route('/api/memories/clean', methods=['POST'])
    def clean_memories():
        try:
            data = request.json
            user_id = data.get('user_id', 'default_user')
            
            result = memory_manager.clean_memories(user_id)
            return jsonify(result)
        except Exception as e:
            logging.error(f"Error in clean_memories endpoint: {str(e)}")
            return jsonify({"error": "Failed to clean memories"}), 500

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)