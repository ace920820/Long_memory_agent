from flask import Flask, render_template, request, jsonify
from models.llm_model import LLMModel
from models.agent import ChatAgent
from services.llm_service import LLMService
from models.rag_module import RAGModule
from models.memory_manager import MemoryManager
from services.prompt_manager import PromptManager
from routes.document_routes import init_document_routes
import yaml
import logging.config
import sys
import logging
import json
import os
from datetime import datetime

# 设置控制台输出为 UTF-8 编码
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

def create_rag_module():
    """创建并初始化 RAGModule 实例"""
    try:
        rag_module = RAGModule()
        print("RAG module initialized successfully")
        return rag_module
    except Exception as e:
        print(f"Error initializing RAG module: {str(e)}")
        raise

def create_app(rag_module):
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
        for role in prompt_manager.role_config.get('available_roles', [])
    }
    
    # 设置默认角色
    default_roles = {
        'default_user': {
            'role': prompt_manager.role_config.get('default_role', 'reindeer')
        }
    }

    # Initialize components
    try:
        # 初始化记忆管理器
        memory_manager = MemoryManager(
            model_name="all-MiniLM-L6-v2",  # 使用默认模型
            memory_file="config/user_memories.json",
            similarity_threshold=0.5
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

    # 注册文档管理路由
    app.register_blueprint(init_document_routes(rag_module))

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
            logging.info(f"正在获取用户 {user_id} 的记忆列表")
            
            # 获取记忆列表
            memories = memory_manager.get_all_memories(user_id)
            
            # 验证返回的数据
            if memories is None:
                return jsonify({
                    "success": False,
                    "error": "记忆管理器返回了空数据"
                }), 500
                
            if not isinstance(memories, list):
                logging.error(f"记忆数据类型错误: {type(memories)}")
                return jsonify({
                    "success": False,
                    "error": "记忆数据格式错误"
                }), 500
                
            # 验证每个记忆的数据完整性
            valid_memories = []
            for memory in memories:
                if isinstance(memory, dict) and 'content' in memory:
                    # 确保必要的字段存在
                    memory['type'] = memory.get('type', 'general')
                    memory['timestamp'] = memory.get('timestamp', datetime.now().isoformat())
                    memory['access_stats'] = memory.get('access_stats', {
                        'count': 0,
                        'first_access': memory.get('timestamp', datetime.now().isoformat()),
                        'last_access': memory.get('timestamp', datetime.now().isoformat()),
                        'access_history': []
                    })
                    valid_memories.append(memory)
                else:
                    logging.warning(f"跳过无效的记忆数据: {memory}")
            
            logging.info(f"成功获取 {len(valid_memories)} 条记忆")
            
            return jsonify({
                "success": True,
                "memories": valid_memories,
                "count": len(valid_memories)
            })
            
        except Exception as e:
            logging.error(f"获取记忆列表时出错: {str(e)}")
            return jsonify({
                "success": False,
                "error": "获取记忆列表失败",
                "details": str(e)
            }), 500

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

    @app.route('/api/memories/delete', methods=['POST'])
    def delete_memories():
        try:
            data = request.json
            print(f"接收到delete_memories请求： {data}")
            memory_ids = data.get('memoryIds', [])
            user_id = data.get('user_id', 'default_user')
            
            # 验证 memory_ids
            if not memory_ids:
                return jsonify({
                    "success": False,
                    "error": "记忆ID列表不能为空"
                }), 400
                
            # 过滤掉无效的 ID（None 或空值）
            valid_memory_ids = [mid for mid in memory_ids if mid is not None and str(mid).strip()]
            
            if not valid_memory_ids:
                return jsonify({
                    "success": False,
                    "error": "没有提供有效的记忆ID"
                }), 400
            
            result = memory_manager.delete_memories(user_id, valid_memory_ids)
            return jsonify(result)
            
        except Exception as e:
            logging.error(f"Error deleting memories: {str(e)}")
            return jsonify({
                "success": False,
                "error": "删除记忆失败",
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
            
            result = memory_manager.split_memory(user_id, memory_id,llm_model)
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

    @app.route('/api/memories/priority', methods=['POST'])
    def update_memory_priority():
        try:
            data = request.json
            user_id = data.get('user_id', 'default_user')
            memory_id = data.get('memory_id')
            
            if memory_id is None:
                return jsonify({"success": False, "error": "Memory ID is required"}), 400
            
            result = memory_manager.update_memory_priority(user_id, memory_id)
            return jsonify(result)
            
        except Exception as e:
            logging.error(f"Error updating memory priority: {str(e)}")
            return jsonify({"error": "Failed to update memory priority"}), 500

    @app.route('/api/memories/clean-low-priority', methods=['POST'])
    def clean_low_priority():
        try:
            data = request.json
            user_id = data.get('user_id', 'default_user')
            
            result = memory_manager.clean_low_priority_memories(user_id)
            return jsonify(result)
            
        except Exception as e:
            logging.error(f"Error cleaning low priority memories: {str(e)}")
            return jsonify({"error": "Failed to clean low priority memories"}), 500

    @app.route('/api/memories/access', methods=['POST'])
    def access_memory():
        try:
            data = request.json
            user_id = data.get('user_id', 'default_user')
            memory_id = data.get('memory_id')
            access_type = data.get('access_type', 'read')
            
            if memory_id is None:
                return jsonify({"success": False, "error": "Memory ID is required"}), 400
            
            result = memory_manager.update_memory_access(user_id, memory_id, access_type)
            return jsonify(result)
            
        except Exception as e:
            logging.error(f"Error accessing memory: {str(e)}")
            return jsonify({"error": "Failed to access memory"}), 500

    @app.route('/get_default_role', methods=['GET'])
    def get_default_role():
        try:
            default_role = prompt_manager.role_config.get('default_role', 'assistant')
            return jsonify({
                "success": True,
                "default_role": default_role
            })
        except Exception as e:
            logging.error(f"Error getting default role: {str(e)}")
            return jsonify({
                "success": False,
                "error": "Failed to get default role",
                "default_role": "assistant"  # 返回一个安全的默认值
            }), 500

    @app.route('/get_available_roles', methods=['GET'])
    def get_available_roles():
        try:
            available_roles = prompt_manager.role_config.get('available_roles', [])
            role_descriptions = prompt_manager.role_config.get('role_descriptions', {})
            role_avatars = prompt_manager.role_config.get('role_avatars', {})
            
            roles_info = {
                role: {
                    'name': role_descriptions.get(role, role),
                    'description': role_descriptions.get(role, ''),
                    'avatar': role_avatars.get(role, '')
                }
                for role in available_roles
            }
            
            return jsonify({
                "success": True,
                "roles": roles_info
            })
        except Exception as e:
            logging.error(f"Error getting available roles: {str(e)}")
            return jsonify({
                "success": False,
                "error": "Failed to get available roles"
            }), 500

    @app.route('/knowledge-base')
    def knowledge_base():
        return render_template('knowledge_base.html')

    return app

if __name__ == "__main__":
    # 初始化 RAG 模块
    rag_module = create_rag_module()
    app = create_app(rag_module)
    app.run(debug=True)