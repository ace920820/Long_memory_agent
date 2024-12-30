import json
import logging
from datetime import datetime
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

class MemoryManager:
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 memory_file: str = "config/user_memories.json",
                 similarity_threshold: float = 0.6):
        """初始化记忆管理器
        :param model_name: 嵌入模型名称
        :param memory_file: 记忆存储文件路径
        :param similarity_threshold: 相似度阈值
        """
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.memory_file = memory_file
        self.similarity_threshold = similarity_threshold
        self.memories = {}  # 用户记忆字典
        self.indices = {}  # 每个用户的FAISS索引
        
        # 加载已存在的记忆
        self.load_memories()
        
        # 验证方法是否正确绑定
        assert hasattr(self, 'get_all_memories'), "get_all_memories method not properly bound"

    def load_memories(self):
        """从文件加载记忆"""
        try:
            if not os.path.exists(self.memory_file):
                logging.info(f"Memory file not found at {self.memory_file}, creating new one")
                self.memories = {"default_user": []}
                self.indices = {"default_user": faiss.IndexFlatL2(self.dimension)}  # 创建空索引
                self.save_memories()
                return

            with open(self.memory_file, 'r', encoding='utf-8') as f:
                content = f.read()
                logging.debug(f"Raw file content: {content}")
                self.memories = json.loads(content)

            # 验证记忆格式并重建索引
            self.indices = {}
            for user_id, user_memories in self.memories.items():
                # 创建用户的索引
                self.indices[user_id] = faiss.IndexFlatL2(self.dimension)
                
                if user_memories:  # 只在有记忆的情况下添加向量
                    try:
                        memory_texts = [m['content'] for m in user_memories]
                        vectors = self.model.encode(memory_texts)
                        self.indices[user_id].add(vectors.astype('float32'))
                    except Exception as e:
                        logging.error(f"Error building index for user {user_id}: {str(e)}")
                        # 如果构建索引失败，创建空索引
                        self.indices[user_id] = faiss.IndexFlatL2(self.dimension)

            logging.info(f"Successfully loaded memories for {len(self.memories)} users")
            logging.debug(f"Loaded memories: {json.dumps(self.memories, ensure_ascii=False, indent=2)}")
            
        except Exception as e:
            logging.error(f"Error loading memories: {str(e)}")
            # 出错时初始化空记忆和索引
            self.memories = {"default_user": []}
            self.indices = {"default_user": faiss.IndexFlatL2(self.dimension)}

    def save_memories(self):
        """保存记忆到文件"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
            
            # 只保存必要的字段
            simplified_memories = {}
            for user_id, memories in self.memories.items():
                simplified_memories[user_id] = []
                for memory in memories:
                    simplified_memory = {
                        'content': memory['content'],
                        'type': memory.get('type', 'general'),  # 添加默认类型
                        'timestamp': memory.get('timestamp', datetime.now().isoformat())  # 添加默认时间戳
                    }
                    simplified_memories[user_id].append(simplified_memory)
            
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(simplified_memories, f, ensure_ascii=False, indent=2)
            
            logging.info("Successfully saved memories to file")
            logging.debug(f"Saved memories: {json.dumps(simplified_memories, ensure_ascii=False, indent=2)}")
        except Exception as e:
            logging.error(f"Error saving memories: {str(e)}")

    def add_memory(self, user_id: str, content: str, memory_type: str = "general") -> Dict:
        """添加新的记忆"""
        try:
            # 确保用户ID存在于memories字典中
            if user_id not in self.memories:
                self.memories[user_id] = []
                self.indices[user_id] = faiss.IndexFlatL2(self.dimension)
                logging.info(f"Created new memory space for user {user_id}")

            # 生成向量
            try:
                vector = self.model.encode([content])[0]
            except Exception as e:
                logging.error(f"Error encoding content: {str(e)}")
                raise

            # 创建记忆条目
            memory_entry = {
                "content": content,
                "type": memory_type,
                "timestamp": datetime.now().isoformat()
            }

            # 添加记忆和向量
            try:
                # 确保向量是正确的形状和类型
                vector_array = np.array([vector]).astype('float32')
                if vector_array.shape[1] != self.dimension:
                    raise ValueError(f"Vector dimension mismatch. Expected {self.dimension}, got {vector_array.shape[1]}")
                
                # 添加记忆
                self.memories[user_id].append(memory_entry)
                
                # 添加向量到索引
                if user_id not in self.indices:
                    self.indices[user_id] = faiss.IndexFlatL2(self.dimension)
                self.indices[user_id].add(vector_array)
                
                logging.debug(f"Added memory for user {user_id}: {memory_entry}")
                
            except Exception as e:
                logging.error(f"Error adding memory to storage: {str(e)}")
                # 如果添加向量失败，需要回滚记忆添加
                if user_id in self.memories and self.memories[user_id]:
                    self.memories[user_id].pop()
                raise

            # 保存到文件
            try:
                self.save_memories()
            except Exception as e:
                logging.error(f"Error saving memories to file: {str(e)}")
                # 回滚内存中的更改
                if user_id in self.memories and self.memories[user_id]:
                    self.memories[user_id].pop()
                if user_id in self.indices:
                    # 重建索引
                    old_vectors = np.array([self.model.encode([m['content']]) for m in self.memories[user_id]])
                    self.indices[user_id] = faiss.IndexFlatL2(self.dimension)
                    if old_vectors.size > 0:
                        self.indices[user_id].add(old_vectors.reshape(-1, self.dimension).astype('float32'))
                raise

            logging.info(f"Successfully added new memory for user {user_id}")
            return {
                "success": True,
                "content": content,
                "type": memory_type,
                "message": "记忆已更新"
            }

        except Exception as e:
            logging.error(f"Error in add_memory: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "message": "记忆更新失败"
            }

    def retrieve_memories(self, user_id: str, query: str, top_k: int = 5) -> List[Dict]:
        """检索相关记忆"""
        try:
            # 确保用户存在且有记忆
            if user_id not in self.memories:
                logging.info(f"No memories found for user {user_id}")
                return []
            
            if not self.memories[user_id]:
                logging.info(f"Memory list is empty for user {user_id}")
                return []

            if user_id not in self.indices:
                logging.error(f"No index found for user {user_id}")
                return []

            # 生成查询向量
            try:
                query_vector = self.model.encode([query])[0]
            except Exception as e:
                logging.error(f"Error encoding query: {str(e)}")
                return []

            # 搜索相似记忆
            try:
                distances, indices = self.indices[user_id].search(
                    np.array([query_vector]).astype('float32'), 
                    min(top_k, len(self.memories[user_id]))
                )
            except Exception as e:
                logging.error(f"Error searching memories: {str(e)}")
                return []

            # 处理结果
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.memories[user_id]):
                    similarity_score = 1 / (1 + float(distances[0][i]))
                    memory = self.memories[user_id][idx]
                    
                    if similarity_score >= self.similarity_threshold:
                        memory_copy = memory.copy()
                        memory_copy['score'] = similarity_score
                        results.append(memory_copy)
                        logging.info(f"记忆匹配 (得分: {similarity_score:.4f}):\n内容: {memory['content']}")
                    else:
                        logging.info(f"记忆因相似度过低被过滤 (得分: {similarity_score:.4f}):\n内容: {memory['content']}")

            # 按相似度排序
            sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
            
            # 记录结果
            if sorted_results:
                logging.info("\n最终使用的记忆:")
                for idx, memory in enumerate(sorted_results, 1):
                    logging.info(f"{idx}. 得分: {memory['score']:.4f}\n内容: {memory['content']}\n")
            else:
                logging.info("没有找到相关记忆")

            return sorted_results

        except Exception as e:
            logging.error(f"Error in retrieve_memories: {str(e)}")
            return []

    def get_all_memories(self, user_id: str) -> List[Dict]:
        """获取用户的所有记忆，并为每条记忆添加ID"""
        try:
            # 检查用户是否存在
            if user_id not in self.memories:
                logging.info(f"Creating new memory list for user {user_id}")
                self.memories[user_id] = []
                self.indices[user_id] = faiss.IndexFlatL2(self.dimension)
                return []

            memories = self.memories[user_id]
            formatted_memories = []
            
            # 为每条记忆添加ID和格式化时间
            for i, memory in enumerate(memories):
                try:
                    formatted_memory = memory.copy()
                    formatted_memory['id'] = i
                    formatted_memory.setdefault('type', 'general')
                    formatted_memory.setdefault('timestamp', datetime.now().isoformat())
                    
                    if 'content' not in formatted_memory:
                        logging.warning(f"Memory at index {i} missing content field")
                        continue
                    
                    formatted_memories.append(formatted_memory)
                    
                except Exception as e:
                    logging.error(f"Error formatting memory at index {i}: {str(e)}")
                    continue

            logging.info(f"Successfully retrieved {len(formatted_memories)} memories for user {user_id}")
            return formatted_memories

        except Exception as e:
            logging.error(f"Error in get_all_memories: {str(e)}")
            return []

    def preview_restructure(self, user_id: str, memory_ids: List[int], template: str) -> str:
        """预览记忆重构结果"""
        try:
            # 获取选中的记忆
            memories = self.memories.get(user_id, [])
            selected_memories = [m for i, m in enumerate(memories) if i in memory_ids]
            
            # 根据模板重构记忆
            if template == "fact":
                return self._restructure_as_fact(selected_memories)
            elif template == "preference":
                return self._restructure_as_preference(selected_memories)
            elif template == "personality":
                return self._restructure_as_personality(selected_memories)
            else:
                return "不支持的重构模板"
            
        except Exception as e:
            logging.error(f"Error previewing restructure: {str(e)}")
            return "预览生成失败"

    def restructure_memories(self, user_id: str, memory_ids: List[int], template: str) -> Dict:
        """重构并保存记忆"""
        try:
            preview = self.preview_restructure(user_id, memory_ids, template)
            if preview == "预览生成失败" or preview == "不支持的重构模板":
                return {"success": False, "error": preview}
            
            # 添加重构后的新记忆
            result = self.add_memory(
                user_id=user_id,
                content=preview,
                memory_type=template
            )
            
            # 可选：删除原始记忆
            # self._delete_memories(user_id, memory_ids)
            
            return {
                "success": True,
                "message": "记忆重构成功",
                "new_memory": result
            }
            
        except Exception as e:
            logging.error(f"Error restructuring memories: {str(e)}")
            return {"success": False, "error": str(e)}

    def _restructure_as_fact(self, memories: List[Dict]) -> str:
        """将记忆重构为事实陈述"""
        facts = []
        for memory in memories:
            content = memory['content']
            if "用户说:" in content and "助手回答:" in content:
                # 从对话中提取事实
                user_part = content.split("助手回答:")[0].replace("用户说:", "").strip()
                facts.append(f"用户表示{user_part}")
            else:
                facts.append(content)
        
        return "。".join(facts) + "。"

    def _restructure_as_preference(self, memories: List[Dict]) -> str:
        """将记忆重构为偏好描述"""
        preferences = []
        for memory in memories:
            content = memory['content']
            if "最喜欢" in content or "喜欢" in content or "讨厌" in content or "不喜欢" in content:
                if "用户说:" in content:
                    content = content.split("助手回答:")[0].replace("用户说:", "").strip()
                preferences.append(content)
        
        return "；".join(preferences) + "。"

    def _restructure_as_personality(self, memories: List[Dict]) -> str:
        """将记忆重构为性格特征描述"""
        traits = []
        keywords = ["是一个", "性格", "特点", "表现出", "倾向于"]
        
        for memory in memories:
            content = memory['content']
            if any(keyword in content for keyword in keywords):
                if "用户说:" in content:
                    content = content.split("助手回答:")[0].replace("用户说:", "").strip()
                traits.append(content)
        
        if traits:
            return "这个人" + "；".join(traits) + "。"
        return "没有足够的信息来描述性格特征。" 