import json
import logging
from datetime import datetime
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

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

    def load_memories(self):
        """从文件加载记忆"""
        try:
            with open(self.memory_file, 'r', encoding='utf-8') as f:
                self.memories = json.load(f)
                # 为每个用户重建索引
                for user_id, user_memories in self.memories.items():
                    memory_texts = [m['content'] for m in user_memories]
                    if memory_texts:
                        vectors = self.model.encode(memory_texts)
                        index = faiss.IndexFlatL2(self.dimension)
                        index.add(vectors)
                        self.indices[user_id] = index
            logging.info(f"Successfully loaded memories for {len(self.memories)} users")
        except FileNotFoundError:
            logging.info("No existing memory file found. Starting fresh.")
        except Exception as e:
            logging.error(f"Error loading memories: {str(e)}")

    def save_memories(self):
        """保存记忆到文件"""
        try:
            # 只保存必要的字段
            simplified_memories = {}
            for user_id, memories in self.memories.items():
                simplified_memories[user_id] = []
                for memory in memories:
                    simplified_memory = {
                        'content': memory['content'],
                        'type': memory['type'],
                        'timestamp': memory['timestamp']
                    }
                    simplified_memories[user_id].append(simplified_memory)
            
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(simplified_memories, f, ensure_ascii=False, indent=2)
            logging.info("Successfully saved memories to file")
        except Exception as e:
            logging.error(f"Error saving memories: {str(e)}")

    def add_memory(self, user_id: str, content: str, memory_type: str = "general") -> Dict:
        """添加新的记忆
        :return: 包含记忆内容和状态的字典
        """
        try:
            vector = self.model.encode([content])[0]
            
            # 创建记忆条目
            memory_entry = {
                "content": content,
                "type": memory_type,
                "timestamp": datetime.now().isoformat()
            }

            # 更新用户记忆
            if user_id not in self.memories:
                self.memories[user_id] = []
                self.indices[user_id] = faiss.IndexFlatL2(self.dimension)

            self.memories[user_id].append(memory_entry)
            self.indices[user_id].add(np.array([vector]).astype('float32'))

            # 保存到文件
            self.save_memories()
            logging.info(f"Added new memory for user {user_id}")
            
            return {
                "success": True,
                "content": content,
                "type": memory_type,
                "message": "记忆已更新"
            }

        except Exception as e:
            logging.error(f"Error adding memory: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "message": "记忆更新失败"
            }

    def retrieve_memories(self, user_id: str, query: str, top_k: int = 5) -> List[Dict]:
        """检索相关记忆"""
        try:
            if user_id not in self.memories or not self.memories[user_id]:
                return []

            query_vector = self.model.encode([query])[0]
            distances, indices = self.indices[user_id].search(
                np.array([query_vector]), top_k
            )

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

            # 按相似度得分排序
            sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
            
            # 记录最终使用的记忆
            if sorted_results:
                logging.info("\n最终使用的记忆:")
                for idx, memory in enumerate(sorted_results, 1):
                    logging.info(f"{idx}. 得分: {memory['score']:.4f}\n内容: {memory['content']}\n")
            else:
                logging.info("没有找到相关记忆")

            return sorted_results

        except Exception as e:
            logging.error(f"Error retrieving memories: {str(e)}")
            return [] 