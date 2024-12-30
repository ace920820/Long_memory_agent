import json
import logging
from datetime import datetime
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import torch
from transformers import BertTokenizer, BertModel
from sentence_transformers import util
import re

class MemoryManager:
    def __init__(self, 
                 model_config: dict = None,  # 新增模型配置参数
                 memory_file: str = "config/user_memories.json",
                 similarity_threshold: float = 0.6):
        """初始化记忆管理器
        :param model_config: 模型配置字典，包含类型和路径
        :param memory_file: 记忆存储文件路径
        :param similarity_threshold: 相似度阈值
        """
        if model_config is None:
            # 默认使用 all-MiniLM-L6-v2
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
        else:
            model_type = model_config.get('type', 'all-MiniLM-L6-v2')
            model_path = model_config.get('paths', {}).get(model_type)
            
            if not model_path:
                logging.warning(f"Model path not found for {model_type}, using default model")
                self.model = SentenceTransformer("all-MiniLM-L6-v2")
            else:
                try:
                    if model_type == 'bert-chinese-local':
                        # 使用本地 BERT 模型
                        from transformers import BertTokenizer, BertModel
                        import torch
                        
                        class BertEmbedding:
                            def __init__(self, model_path):
                                self.tokenizer = BertTokenizer.from_pretrained(model_path)
                                self.model = BertModel.from_pretrained(model_path)
                                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                                self.model.to(self.device)
                                self.model.eval()

                            def encode(self, sentences, batch_size=32):
                                all_embeddings = []
                                
                                for i in range(0, len(sentences), batch_size):
                                    batch = sentences[i:i + batch_size]
                                    encoded = self.tokenizer(batch, 
                                                           padding=True, 
                                                           truncation=True, 
                                                           max_length=512, 
                                                           return_tensors='pt')
                                    
                                    with torch.no_grad():
                                        encoded = {k: v.to(self.device) for k, v in encoded.items()}
                                        outputs = self.model(**encoded)
                                        # 使用 [CLS] token 的输出作为句子表示
                                        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                                        all_embeddings.append(embeddings)
                                
                                return np.vstack(all_embeddings)

                            def get_sentence_embedding_dimension(self):
                                return 768  # BERT base 的输出维度

                        self.model = BertEmbedding(model_path)
                        logging.info(f"Using local BERT model from {model_path}")
                    else:
                        # 使用 sentence-transformers 模型
                        self.model = SentenceTransformer(model_path)
                        logging.info(f"Using sentence-transformers model: {model_path}")
                except Exception as e:
                    logging.error(f"Error loading model {model_type}: {str(e)}")
                    logging.warning("Falling back to default model")
                    self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.dimension = self.model.get_sentence_embedding_dimension()
        self.memory_file = memory_file
        self.similarity_threshold = similarity_threshold
        self.memories = {}
        self.indices = {}
        
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
            # logging.debug(f"Saved memories: {json.dumps(simplified_memories, ensure_ascii=False, indent=2)}")
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

    def retrieve_memories(self, user_id: str, query: str, top_k: int = 50) -> List[Dict]:
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

    def preview_restructure(self, user_id: str, memory_ids: List[int], template: str) -> Dict:
        """预览记忆重构结果"""
        try:
            # 获取选中的记忆
            memories = self.memories.get(user_id, [])
            previews = []
            
            # 对每条记忆单独进行重构预览
            for idx in memory_ids:
                if 0 <= idx < len(memories):
                    single_memory = [memories[idx]]
                    if template == "fact":
                        preview = self._restructure_as_fact(single_memory)
                    elif template == "preference":
                        preview = self._restructure_as_preference(single_memory)
                    elif template == "personality":
                        preview = self._restructure_as_personality(single_memory)
                    else:
                        preview = "不支持的重构模板"
                    previews.append(preview)
            
            return {
                "success": True,
                "previews": previews
            }
            
        except Exception as e:
            logging.error(f"Error previewing restructure: {str(e)}")
            return {
                "success": False,
                "error": "预览生成失败",
                "details": str(e)
            }

    def restructure_memories(self, user_id: str, memory_ids: List[int], template: str) -> Dict:
        """重构记忆，每条记忆单独重构"""
        try:
            if user_id not in self.memories:
                return {"success": False, "error": "User not found"}
            
            if not memory_ids:
                return {"success": False, "error": "No memories selected"}
            
            # 获取所有选中的记忆
            memories = self.memories.get(user_id, [])
            restructured_memories = []
            updated_indices = []
            
            # 对每条记忆单独进行重构
            for idx in memory_ids:
                if 0 <= idx < len(memories):
                    # 对单条记忆进行重构
                    single_memory = [memories[idx]]
                    if template == "fact":
                        new_content = self._restructure_as_fact(single_memory)
                    elif template == "preference":
                        new_content = self._restructure_as_preference(single_memory)
                    elif template == "personality":
                        new_content = self._restructure_as_personality(single_memory)
                    else:
                        continue
                    
                    # 创建新的记忆条目
                    new_memory = {
                        "content": new_content,
                        "type": template,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # 更新记忆
                    self.memories[user_id][idx] = new_memory
                    restructured_memories.append(new_memory)
                    updated_indices.append(idx)
            
            if restructured_memories:
                try:
                    # 重建索引
                    memory_texts = [m['content'] for m in self.memories[user_id]]
                    vectors = self.model.encode(memory_texts)
                    self.indices[user_id] = faiss.IndexFlatL2(self.dimension)
                    self.indices[user_id].add(vectors.astype('float32'))
                    
                    # 保存更改
                    self.save_memories()
                    
                    return {
                        "success": True,
                        "message": f"成功重构 {len(restructured_memories)} 条记忆",
                        "updated_memories": restructured_memories,
                        "updated_indices": updated_indices
                    }
                    
                except Exception as e:
                    logging.error(f"Error updating memories: {str(e)}")
                    return {"success": False, "error": str(e)}
            else:
                return {"success": False, "error": "No memories were restructured"}
            
        except Exception as e:
            logging.error(f"Error restructuring memories: {str(e)}")
            return {"success": False, "error": str(e)}

    def _restructure_as_fact(self, memories: List[Dict]) -> str:
        """将记忆重构为事实陈述"""
        if not memories:
            return ""
        
        content = memories[0]['content']
        if "用户说:" in content and "助手回答:" in content:
            # 从对话中提取事实
            user_part = content.split("助手回答:")[0].replace("用户说:", "").strip()
            return f"用户表示{user_part}"
        return content

    def _restructure_as_preference(self, memories: List[Dict]) -> str:
        """将记忆重构为偏好描述"""
        if not memories:
            return ""
        
        content = memories[0]['content']
        if "用户说:" in content:
            content = content.split("助手回答:")[0].replace("用户说:", "").strip()
        
        # 如果内容中没有偏好相关的关键词，尝试添加适当的前缀
        if not any(keyword in content for keyword in ["最喜欢", "喜欢", "讨厌", "不喜欢"]):
            if "想" in content or "要" in content:
                content = f"用户喜欢{content}"
        
        return content

    def _restructure_as_personality(self, memories: List[Dict]) -> str:
        """将记忆重构为性格特征描述"""
        if not memories:
            return ""
        
        content = memories[0]['content']
        if "用户说:" in content:
            content = content.split("助手回答:")[0].replace("用户说:", "").strip()
        
        keywords = ["是一个", "性格", "特点", "表现出", "倾向于"]
        if not any(keyword in content for keyword in keywords):
            return f"用户在交谈中表现出{content}的特点"
        return content

    def delete_memories(self, user_id: str, memory_ids: List[int]) -> Dict:
        """删除指定的记忆"""
        try:
            if user_id not in self.memories:
                return {"success": False, "error": "User not found"}

            logging.info(f"Attempting to delete memories {memory_ids} for user {user_id}")
            
            # 按照索引从大到小排序，这样删除时不会影响其他记忆的索引
            sorted_ids = sorted(memory_ids, reverse=True)
            deleted_count = 0

            # 删除记忆
            for idx in sorted_ids:
                if 0 <= idx < len(self.memories[user_id]):
                    logging.info(f"Deleting memory at index {idx}")
                    self.memories[user_id].pop(idx)
                    deleted_count += 1
                else:
                    logging.warning(f"Invalid memory index: {idx}")

            # 重建索引
            if deleted_count > 0:
                if self.memories[user_id]:
                    memory_texts = [m['content'] for m in self.memories[user_id]]
                    vectors = self.model.encode(memory_texts)
                    self.indices[user_id] = faiss.IndexFlatL2(self.dimension)
                    self.indices[user_id].add(vectors.astype('float32'))
                else:
                    # 如果没有记忆了，创建空索引
                    self.indices[user_id] = faiss.IndexFlatL2(self.dimension)

                # 保存更改
                self.save_memories()
                logging.info(f"Successfully deleted {deleted_count} memories")

            return {
                "success": True,
                "deleted_count": deleted_count,
                "message": f"Successfully deleted {deleted_count} memories"
            }

        except Exception as e:
            logging.error(f"Error deleting memories: {str(e)}")
            return {"success": False, "error": str(e)}

    def split_memory(self, user_id: str, memory_id: int) -> Dict:
        """拆分记忆为多个较短的记忆"""
        try:
            if user_id not in self.memories:
                return {"success": False, "error": "User not found"}
            
            memories = self.memories[user_id]
            if not (0 <= memory_id < len(memories)):
                return {"success": False, "error": "Memory not found"}
            
            content = memories[memory_id]['content']
            memory_type = memories[memory_id]['type']
            
            # 1. 首先尝试基于规则的拆分
            segments = self._rule_based_split(content)
            
            # 2. 如果规则拆分得到的片段为1条，使用LLM进行拆分
            if len(segments) <= 1:
                logging.info("Rule-based split failed, attempting LLM-based split")
                llm_segments = self._llm_based_split(content)
                if llm_segments and len(llm_segments) > 1:  # 确保LLM拆分成功且产生多个片段
                    segments = llm_segments
                    logging.info(f"LLM split successful, generated {len(segments)} segments")
                else:
                    logging.warning("LLM split failed or produced single segment")
                    return {
                        "success": False, 
                        "error": "无法进一步拆分该记忆",
                        "message": "记忆内容过于简短或已经是最小单位"
                    }
            
            # 3. 添加新的记忆
            new_memories = []
            for segment in segments:
                if segment.strip():  # 忽略空字符串
                    result = self.add_memory(user_id, segment, memory_type)
                    if result.get("success"):
                        new_memories.append(result)
            
            # 4. 如果成功添加了新记忆，删除原始记忆
            if len(new_memories) > 1:  # 只有在成功拆分为多条记忆时才删除原记忆
                self.delete_memories(user_id, [memory_id])
                return {
                    "success": True,
                    "message": f"记忆已拆分为 {len(new_memories)} 条",
                    "new_memories": new_memories
                }
            else:
                # 清理已添加的新记忆（如果有的话）
                for new_memory in new_memories:
                    if 'id' in new_memory:
                        self.delete_memories(user_id, [new_memory['id']])
                return {
                    "success": False,
                    "error": "拆分失败",
                    "message": "无法将记忆拆分为多个有意义的片段"
                }
            
        except Exception as e:
            logging.error(f"Error splitting memory: {str(e)}")
            return {"success": False, "error": str(e)}

    def _rule_based_split(self, content: str) -> List[str]:
        """基于规则的记忆拆分"""
        segments = []
        
        # 1. 处理对话格式
        if "用户说:" in content and "助手回答:" in content:
            parts = content.split("助手回答:")
            if len(parts) == 2:
                user_part = parts[0].replace("用户说:", "").strip()
                assistant_part = parts[1].strip()
                if user_part:
                    segments.append(f"用户说: {user_part}")
                if assistant_part:
                    segments.append(f"助手回答: {assistant_part}")
                return segments
        
        # 2. 基于分隔符拆分
        separators = ["。", "！", "？", "\n"]
        temp_segments = [content]
        
        for sep in separators:
            new_segments = []
            for seg in temp_segments:
                parts = seg.split(sep)
                parts = [p.strip() + sep for p in parts[:-1]] + [parts[-1].strip()]
                new_segments.extend([p for p in parts if p.strip()])
            temp_segments = new_segments
        
        # 3. 合并过短的片段
        min_length = 10  # 最小长度阈值
        current_segment = ""
        
        for seg in temp_segments:
            if len(current_segment) + len(seg) <= 100:  # 最大长度阈值
                current_segment += seg
            else:
                if current_segment:
                    segments.append(current_segment)
                current_segment = seg
        
        if current_segment:
            segments.append(current_segment)
        
        return segments or [content]

    def _llm_based_split(self, content: str) -> Optional[List[str]]:
        """使用LLM进行记忆拆分"""
        try:
            # 构建更详细的提示词
            prompt = f"""请将以下内容拆分成2-5条独立的、有意义的短句。每条短句都应该：
1. 包含完整的信息
2. 保持原始语境
3. 避免重复信息
4. 长度适中（10-50个字符）

原始内容：
{content}

请直接返回拆分后的句子，每句一行，不要添加任何其他内容。如果内容无法合理拆分，请返回空行。"""

            # 调用LLM
            from models.llm_model import LLMModel
            llm = LLMModel()  # 假设你有一个LLM模型实例
            
            try:
                response = llm.generate(prompt)
                
                # 处理响应
                segments = []
                if response:
                    lines = response.strip().split('\n')
                    for line in lines:
                        line = line.strip()
                        if line and not line.startswith(('原始内容', '请', '拆分')):
                            segments.append(line)
                    
                    # 验证拆分结果
                    if len(segments) > 1:
                        # 检查每个片段的有效性
                        valid_segments = []
                        for seg in segments:
                            if 10 <= len(seg) <= 100 and not any(seg in s for s in valid_segments):
                                valid_segments.append(seg)
                        
                        if len(valid_segments) > 1:
                            return valid_segments
                
                return None
                
            except Exception as e:
                logging.error(f"Error calling LLM: {str(e)}")
                return None
                
        except Exception as e:
            logging.error(f"Error in LLM-based split: {str(e)}")
            return None 

    def get_relevant_memories(self, user_id, query, top_k=5):
        """优化的记忆检索方法"""
        try:
            # 1. 动态相似度阈值
            base_threshold = self.similarity_threshold
            query_length = len(query)
            adjusted_threshold = base_threshold * (1 - 0.1 * (query_length > 50))  # 长查询降低阈值
            
            # 2. 时间衰减因子
            current_time = datetime.now()
            def get_time_weight(memory_time):
                time_diff = (current_time - datetime.fromisoformat(memory_time)).days
                return 1 / (1 + 0.1 * time_diff)  # 时间权重衰减
            
            # 3. 记忆类型权重
            type_weights = {
                'fact': 1.2,
                'preference': 1.1,
                'personality': 1.3,
                'experience': 1.0,
                'general': 0.9
            }
            
            # 获取并排序记忆
            relevant_memories = []
            for memory in self.get_all_memories(user_id):
                similarity = self.calculate_similarity(query, memory['content'])
                time_weight = get_time_weight(memory.get('timestamp', ''))
                type_weight = type_weights.get(memory.get('type', 'general'), 1.0)
                
                final_score = similarity * time_weight * type_weight
                if final_score >= adjusted_threshold:
                    memory['relevance_score'] = final_score
                    relevant_memories.append(memory)
            
            return sorted(relevant_memories, key=lambda x: x['relevance_score'], reverse=True)
            
        except Exception as e:
            logging.error(f"Error in get_relevant_memories: {str(e)}")
            return []

    def get_memories_for_context(self, user_id, query):
        """获取用于上下文的记忆"""
        relevant_memories = self.get_relevant_memories(user_id, query)
        
        # 格式化记忆为上下文字符串
        context_memories = []
        for memory in relevant_memories:
            formatted_memory = f"记忆 ({memory.get('type', 'general')}): {memory.get('content', '')}"
            context_memories.append(formatted_memory)
        
        return "\n".join(context_memories) if context_memories else "" 

    def clean_memories(self, user_id: str) -> Dict:
        """清理重复和无意义的记忆，并自动更新记忆类型"""
        try:
            if user_id not in self.memories:
                return {"success": False, "error": "用户不存在"}

            all_memories = self.get_all_memories(user_id)
            if not all_memories:
                return {"success": False, "error": "没有找到需要整理的记忆"}

            # 创建记忆内容的嵌入向量
            memory_texts = [m['content'] for m in all_memories]
            memory_embeddings = self.model.encode(memory_texts, convert_to_tensor=True)

            # 计算记忆间的相似度矩阵
            similarities = util.pytorch_cos_sim(memory_embeddings, memory_embeddings)

            # 要删除的记忆索引
            memories_to_delete = []
            # 要更新类型的记忆
            memories_to_update = []

            # 定义无意义记忆的模式
            meaningless_patterns = [
                r"用户(说|表示|问).*(吗|呢)\??$",  # 问句模式
                r"用户(说|表示).*什么.*",  # 泛泛而谈的模式
                r"^(嗯|好的|明白|知道了|谢谢).*",  # 简单应答模式
                r"用户想知道.*",  # 询问模式
                r"用户在问.*"  # 询问模式
            ]

            # 定义记忆类型的模式
            type_patterns = {
                'fact': [
                    r"用户(说|表示).*是.*",  # 事实陈述
                    r".*的特点是.*",
                    r".*位于.*",
                    r".*包含.*",
                    r".*由.*组成"
                ],
                'preference': [
                    r"用户(喜欢|讨厌|热爱|偏好|不喜欢).*",  # 偏好表达
                    r".*觉得.*很(好|差|棒|糟)",
                    r".*认为.*比.*更.*",
                    r".*更倾向于.*"
                ],
                'personality': [
                    r".*的性格.*",  # 性格特征
                    r".*的个性.*",
                    r".*这个人.*",
                    r".*的习惯是.*"
                ],
                'experience': [
                    r".*曾经.*",  # 经历
                    r".*经历过.*",
                    r".*发生过.*",
                    r".*做过.*"
                ]
            }

            # 检测重复、无意义的记忆，并更新记忆类型
            for i in range(len(all_memories)):
                if i in memories_to_delete:
                    continue

                content = all_memories[i]['content']
                current_type = all_memories[i].get('type', 'general')
                
                # 检查是否是无意义的记忆
                if any(re.search(pattern, content) for pattern in meaningless_patterns):
                    memories_to_delete.append(i)
                    logging.info(f"标记无意义记忆: {content}")
                    continue

                # 检查重复记忆
                for j in range(i + 1, len(all_memories)):
                    if j in memories_to_delete:
                        continue
                    
                    similarity = similarities[i][j]
                    if similarity >= 0.85:  # 高相似度阈值
                        content_j = all_memories[j]['content']
                        logging.info(f"发现相似记忆 ({similarity:.2f}):\n{content}\n{content_j}")
                        memories_to_delete.append(j)

                # 自动判断并更新记忆类型
                detected_type = None
                for memory_type, patterns in type_patterns.items():
                    if any(re.search(pattern, content) for pattern in patterns):
                        detected_type = memory_type
                        break

                if detected_type and detected_type != current_type:
                    memories_to_update.append({
                        'index': i,
                        'new_type': detected_type,
                        'old_type': current_type
                    })

            # 更新记忆类型
            updated_count = 0
            for update in memories_to_update:
                idx = update['index']
                if idx < len(self.memories[user_id]):
                    self.memories[user_id][idx]['type'] = update['new_type']
                    updated_count += 1
                    logging.info(f"更新记忆类型: {update['old_type']} -> {update['new_type']}\n内容: {self.memories[user_id][idx]['content']}")

            # 删除标记的记忆
            deleted_count = 0
            if memories_to_delete:
                delete_result = self.delete_memories(user_id, memories_to_delete)
                if delete_result.get("success"):
                    deleted_count = delete_result.get('deleted_count', 0)

            # 保存更改
            if updated_count > 0:
                self.save_memories()

            return {
                "success": True,
                "message": f"清理完成：删除了 {deleted_count} 条重复或无意义记忆，更新了 {updated_count} 条记忆的类型",
                "deleted_count": deleted_count,
                "updated_count": updated_count,
                "total_memories": len(all_memories)
            }

        except Exception as e:
            logging.error(f"Error in clean_memories: {str(e)}")
            return {"success": False, "error": str(e)} 