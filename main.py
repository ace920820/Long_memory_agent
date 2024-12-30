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
    llm_model = LLMModel(config)
    rag_module = RAGModule(similarity_threshold=0.55)
    memory_manager = MemoryManager(
        model_config=config.get('embedding_model'),
        memory_file="config/user_memories.json",
        similarity_threshold=0.45
    )

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

    # 初始化提示词管理器
    prompt_manager = PromptManager()

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