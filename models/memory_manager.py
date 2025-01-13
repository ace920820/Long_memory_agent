import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
import faiss
import numpy as np
import os
import torch
import re
import math
import time
import random

class MemoryManager:
    def __init__(self, model_name: str = "bge-small-zh-v1.5", 
                 model_path: str = "D:/models/bge-small-zh-v1.5",
                 memory_file: str = "config/user_memories.json",
                 similarity_threshold: float = 0.5,
                 new_memory_similarity_threshold: float = 0.9):
        """初始化记忆管理器
        
        Args:
            model_name: 使用的嵌入模型名称
            model_path: 模型本地路径
            memory_file: 记忆存储文件路径
            similarity_threshold: 相似度阈值
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.memory_file = memory_file
        self.similarity_threshold = similarity_threshold
        self.new_memory_similarity_threshold = new_memory_similarity_threshold
        self.memories = self.load_memories()
        
        # 初始化 BGE-Rerank 模型
        self.reranker_tokenizer = AutoTokenizer.from_pretrained(r"D:\models\bge-reranker-base")
        self.reranker = AutoModelForSequenceClassification.from_pretrained(r"D:\models\bge-reranker-base")
        self.reranker.eval()
        logging.info("Successfully loaded BGE-Rerank model")
        
        # 先获取实际的向量维度
        if self.memories:
            sample_text = ["测试文本"]
            sample_vector = self.encode(sample_text)
            self.dimension = sample_vector.shape[1]
            logging.info(f"Actual vector dimension: {self.dimension}")
        else:
            self.dimension = 768  # 默认维度
        
        self.indices = {}  # 用户ID到索引的映射
        
        # 为每个用户创建索引
        for user_id, memories in self.memories.items():
            self.create_user_index(user_id)
        
        # 确保记忆文件存在
        if not os.path.exists(memory_file):
            self.save_memories()
        
        # 优先级相关配置
        self.priority_config = {
            'weights': {
                'time': 0.3,
                'importance': 0.4,
                'access': 0.3
            },
            'type_weights': {
                'fact': 0.8,
                'preference': 0.6,
                'dialogue': 0.4,
                'general': 0.5
            },
            'max_memories': 1000,
            'cleanup_threshold': 0.85
        }
        
        # 初始化或迁移现有记忆的访问统计
        self._initialize_memory_stats()

    def load_memories(self) -> Dict:
        """从文件加载记忆"""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    memories = json.load(f)
                    
                # 确保所有记忆都有完整的访问统计结构
                for user_id, user_memories in memories.items():
                    for memory in user_memories:
                        if 'access_stats' not in memory:
                            memory['access_stats'] = {
                                'count': memory.get('access_count', 0),  # 兼容旧数据
                                'first_access': memory.get('timestamp', datetime.now().isoformat()),
                                'last_access': memory.get('last_access', memory.get('timestamp')),
                                'access_history': []
                            }
                        # 删除旧的统计字段
                        memory.pop('access_count', None)
                        
                        # 确保有优先级分数
                        if 'priority_score' not in memory:
                            memory['priority_score'] = 0.0
                            
                return memories
            return {}
            
        except Exception as e:
            logging.error(f"Error loading memories: {str(e)}")
            return {}

    def save_memories(self):
        """保存记忆到文件"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
            
            # 保存包含访问统计的完整记忆数据
            simplified_memories = {}
            for user_id, memories in self.memories.items():
                simplified_memories[user_id] = []
                for memory in memories:
                    simplified_memory = {
                        'id': memory.get('id', self._generate_memory_id()),  # 保存ID,如果没有则生成新的
                        'content': memory['content'],
                        'type': memory.get('type', 'general'),
                        'timestamp': memory.get('timestamp', datetime.now().isoformat()),
                        'access_stats': {
                            'count': memory.get('access_stats', {}).get('count', 0),
                            'first_access': memory.get('access_stats', {}).get(
                                'first_access', memory.get('timestamp', datetime.now().isoformat())
                            ),
                            'last_access': memory.get('access_stats', {}).get(
                                'last_access', memory.get('timestamp', datetime.now().isoformat())
                            ),
                            'access_history': memory.get('access_stats', {}).get('access_history', [])
                        },
                        'priority_score': memory.get('priority_score', 0.0)
                    }
                    simplified_memories[user_id].append(simplified_memory)
            
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(simplified_memories, f, ensure_ascii=False, indent=2)
            
            logging.info("Successfully saved memories to file")
            
        except Exception as e:
            logging.error(f"Error saving memories: {str(e)}")

    def create_user_index(self, user_id: str):
        """为用户创建向量索引"""
        if user_id not in self.indices:
            self.indices[user_id] = faiss.IndexFlatIP(self.dimension)

        if user_id in self.memories and self.memories[user_id]:
            memory_texts = [m['content'] for m in self.memories[user_id]]
            vectors = self.encode(memory_texts)
            logging.info(f"Vector dimension: {vectors.shape}")  # 添加日志
            if vectors.shape[1] != self.dimension:
                logging.error(f"Dimension mismatch: expected {self.dimension}, got {vectors.shape[1]}")
                self.dimension = vectors.shape[1]  # 更新维度
                self.indices[user_id] = faiss.IndexFlatIP(self.dimension)  # 重新创建索引
            self.indices[user_id].add(vectors.astype('float32'))

    def _generate_memory_id(self) -> str:
        """生成唯一的记忆ID"""
        timestamp = int(time.time() * 1000)  # 毫秒级时间戳
        random_num = random.randint(1000, 9999)  # 4位随机数
        return f"m_{timestamp}_{random_num}"

    def add_memory(self, user_id: str, content: str, memory_type: str = "general") -> Dict:
        """添加新的记忆
        
        Args:
            user_id: 用户ID
            content: 记忆内容
            memory_type: 记忆类型
            
        Returns:
            Dict: 包含操作结果的字典
        """
        try:
            # 确保用户存在
            if user_id not in self.memories:
                self.memories[user_id] = []
                self.indices[user_id] = faiss.IndexFlatIP(self.dimension)

            # 如果内容包含对话格式，只保留用户的输入部分
            if "用户说:" in content and "助手回答:" in content:
                content = content.split("助手回答:")[0].replace("用户说:", "").strip()
            elif "用户说:" in content:
                content = content.replace("用户说:", "").strip()

            # 如果内容为空，不添加记忆
            if not content:
                return {
                    "success": False,
                    "message": "记忆内容为空，未添加记忆"
                }

            # 检查是否存在相似记忆
            if self.memories[user_id]:
                # 编码新内容
                new_vector = self.encode([content], convert_to_tensor=True)
                
                # 获取现有记忆的内容
                existing_contents = [m['content'] for m in self.memories[user_id]]
                existing_vectors = self.encode(existing_contents, convert_to_tensor=True)
                
                # 计算相似度
                similarities = self.calculate_similarity(new_vector, existing_vectors)
                max_similarity = torch.max(similarities).item()
                
                # 如果存在高相似度的记忆，不添加新记忆
                if max_similarity >= self.new_memory_similarity_threshold:
                    logging.info(f"发现相似记忆 (相似度: {max_similarity:.4f})")
                    return {
                        "success": False,
                        "message": "已存在相似记忆，未添加新记忆",
                        "similarity": max_similarity
                    }

            # 创建新记忆
            new_memory = {
                'id': self._generate_memory_id(),  # 添加唯一ID
                'content': content,
                'type': memory_type,
                'timestamp': datetime.now().isoformat(),
                'access_stats': {
                    'count': 0,
                    'first_access': datetime.now().isoformat(),
                    'last_access': datetime.now().isoformat(),
                    'access_history': []
                }
            }

            # 添加到记忆列表
            self.memories[user_id].append(new_memory)

            # 更新向量索引
            try:
                vector = self.encode([content]).astype('float32')
                self.indices[user_id].add(vector)
            except Exception as e:
                logging.error(f"Error updating index: {str(e)}")
                # 回滚记忆添加
                self.memories[user_id].pop()
                raise

            # 保存到文件
            try:
                self.save_memories()
            except Exception as e:
                logging.error(f"Error saving memories: {str(e)}")
                # 回滚所有更改
                self.memories[user_id].pop()
                if user_id in self.indices:
                    self.create_user_index(user_id)  # 重建索引
                raise

            logging.info(f"成功添加新记忆: {content}")
            return {
                "success": True,
                "content": content,
                "type": memory_type,
                "message": "记忆已添加"
            }

        except Exception as e:
            logging.error(f"Error in add_memory: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "message": "记忆添加失败"
            }

    def retrieve_memories(self, user_id: str, query: str, top_k: int = 5) -> List[Dict]:
        """检索相关记忆"""
        try:
            if user_id not in self.indices or user_id not in self.memories:
                return []

            # 编码查询文本
            query_vector = self.encode([query])[0]
            
            # 使用FAISS进行向量检索
            similarities, indices = self.indices[user_id].search(
                query_vector.reshape(1, -1).astype('float32'), 
                min(top_k * 2, len(self.memories[user_id]))  # 检索更多候选项用于重排序
            )
            
            # 获取检索到的记忆
            retrieved_memories = []
            for sim, idx in zip(similarities[0], indices[0]):
                if idx < len(self.memories[user_id]):
                    memory = self.memories[user_id][idx].copy()
                    memory['similarity'] = float(sim)  # 添加相似度分数
                    if sim >= self.similarity_threshold:
                        retrieved_memories.append(memory)
            
            # 使用 rerank 进行重排序
            reranked_memories = self._rerank_results(query, retrieved_memories)
            
            # 只返回 top_k 个最相关的记忆
            return reranked_memories[:top_k]
            
        except Exception as e:
            logging.error(f"Error retrieving memories: {str(e)}")
            return []

    def get_all_memories(self, user_id: str) -> List[Dict]:
        """获取用户的所有记忆
        
        Args:
            user_id: 用户ID
            
        Returns:
            List[Dict]: 记忆列表
        """
        try:
            if user_id not in self.memories:
                logging.info(f"No memories found for user {user_id}")
                return []
            
            memories = self.memories[user_id]
            logging.info(f"Successfully retrieved {len(memories)} memories for user {user_id}")
            return memories
            
        except Exception as e:
            logging.error(f"Error retrieving memories: {str(e)}")
            return []

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

    def delete_memories(self, user_id: str, memory_ids: List[str]) -> Dict:
        """删除指定的记忆
        
        Args:
            user_id (str): 用户ID
            memory_ids (List[str]): 要删除的记忆ID列表
            
        Returns:
            Dict: 包含操作结果的字典
        """
        try:
            if user_id not in self.memories:
                return {"success": False, "error": "User not found"}

            logging.info(f"Attempting to delete memories {memory_ids} for user {user_id}")
            
            deleted_count = 0
            # 创建一个新的记忆列表，保留未被删除的记忆
            new_memories = []
            for memory in self.memories[user_id]:
                if memory.get('id') not in memory_ids:
                    new_memories.append(memory)
                else:
                    deleted_count += 1
                    logging.info(f"Deleting memory with id {memory.get('id')}")
            
            # 更新记忆列表
            self.memories[user_id] = new_memories

            # 重建索引
            if deleted_count > 0:
                if self.memories[user_id]:
                    memory_texts = [m['content'] for m in self.memories[user_id]]
                    vectors = self.encode(memory_texts)
                    self.indices[user_id] = faiss.IndexFlatIP(self.dimension)
                    self.indices[user_id].add(vectors.astype('float32'))
                else:
                    # 如果没有记忆了，创建空索引
                    self.indices[user_id] = faiss.IndexFlatIP(self.dimension)

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

    def split_memory(self, user_id: str, memory_id: int,llm_model) -> Dict:
        """拆分记忆为多个较短的记忆"""
        try:
            if user_id not in self.memories:
                return {"success": False, "error": "User not found"}
            
            memories = self.memories[user_id]
            if not memories or memory_id < 0 or memory_id >= len(memories):
                return {"success": False, "error": "Memory not found"}
            
            content = memories[memory_id]['content']
            memory_type = memories[memory_id]['type']
            
            # 1. 首先尝试基于规则的拆分
            segments = self._rule_based_split(content)
            
            # 2. 如果规则拆分得到的片段为1条，使用LLM进行拆分
            if len(segments) <= 1:
                logging.info("Rule-based split failed, attempting LLM-based split")
                llm_segments = self._llm_based_split(content,llm_model)
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
                    # 创建新记忆
                    new_memory = {
                        'id': self._generate_memory_id(),
                        'content': segment,
                        'type': memory_type,
                        'timestamp': datetime.now().isoformat(),
                        'access_stats': {
                            'count': 0,
                            'first_access': datetime.now().isoformat(),
                            'last_access': datetime.now().isoformat(),
                            'access_history': []
                        },
                        'priority_score': 0.0
                    }
                    self.memories[user_id].append(new_memory)
                    new_memories.append(new_memory)
            
            # 4. 如果成功添加了新记忆，删除原始记忆并更新索引
            if len(new_memories) > 1:  # 只有在成功拆分为多条记忆时才删除原记忆
                # 删除原始记忆
                self.delete_memories(user_id, [memories[memory_id]['id']])
                
                # 重建索引
                memory_texts = [m['content'] for m in self.memories[user_id]]
                vectors = self.encode(memory_texts)
                self.indices[user_id] = faiss.IndexFlatIP(self.dimension)
                self.indices[user_id].add(vectors.astype('float32'))
                
                # 保存更改
                self.save_memories()
                
                return {
                    "success": True,
                    "message": f"记忆已拆分为 {len(new_memories)} 条",
                    "new_memories": new_memories
                }
            else:
                # 清理已添加的新记忆（如果有的话）
                indices_to_delete = []
                for i, memory in enumerate(self.memories[user_id]):
                    if memory['id'] in [m['id'] for m in new_memories]:
                        indices_to_delete.append(i)
                if indices_to_delete:
                    self.delete_memories(user_id, [self.memories[user_id][i]['id'] for i in indices_to_delete])
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

    def _llm_based_split(self, content: str,llm_model) -> Optional[List[str]]:
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
            
            try:
                response = llm_model.generate(prompt)
                
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
            memory_embeddings = self.encode(memory_texts, convert_to_tensor=True)

            # 计算记忆间的相似度矩阵
            similarities = self.calculate_similarity(memory_embeddings, memory_embeddings)

            # 要删除的记忆索引
            memories_to_delete = []
            # 要更新类型的记忆
            memories_to_update = []

            # 定义无意义记忆的模式
            meaningless_patterns = [
                r"用户(说|表示|问).*(吗|呢)\\??$",  # 问句模式
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
                delete_result = self.delete_memories(user_id, [self.memories[user_id][i]['id'] for i in memories_to_delete])
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

    def _rerank_results(self, query: str, memories: List[Dict]) -> List[Dict]:
        """使用 BGE-Rerank 对检索结果进行重排序
        
        Args:
            query: 查询文本
            memories: 第一阶段检索的候选记忆
            
        Returns:
            重排序后的结果列表
        """
        if not memories:
            return memories
            
        # 准备 rerank 的文本对
        pairs = []
        for memory in memories:
            pairs.append([query, memory['content']])
        
        # 计算 rerank 分数
        with torch.no_grad():
            inputs = self.reranker_tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512
            )
            scores = self.reranker(**inputs).logits.squeeze()
            scores = torch.sigmoid(scores).tolist()
        
        # 如果只有一个结果，确保 scores 是列表
        if not isinstance(scores, list):
            scores = [scores]
        
        # 更新记忆分数并重排序
        for memory, score in zip(memories, scores):
            memory['rerank_score'] = float(score)
            # 综合考虑向量相似度和 rerank 分数
            memory['final_score'] = 0.3 * memory['similarity'] + 0.7 * memory['rerank_score']
        
        # 按照综合分数重排序
        memories.sort(key=lambda x: x['final_score'], reverse=True)
        
        return memories

    def encode(self, texts: List[str], convert_to_tensor: bool = False) -> np.ndarray:
        """将文本编码为向量
        
        Args:
            texts: 要编码的文本列表
            convert_to_tensor: 是否转换为tensor
            
        Returns:
            np.ndarray: 文本向量
        """
        # 对输入进行编码
        inputs = self.tokenizer(texts, 
                              padding=True, 
                              truncation=True, 
                              max_length=512, 
                              return_tensors="pt")
        
        # 获取向量表示
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # 使用[CLS] token的输出作为文本表示
        embeddings = outputs.last_hidden_state[:, 0]
        
        # 归一化
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # 根据需要返回tensor或numpy数组
        if convert_to_tensor:
            return embeddings
        return embeddings.numpy()

    def calculate_similarity(self, embeddings1: torch.Tensor, embeddings2: torch.Tensor) -> torch.Tensor:
        """计算两组向量之间的余弦相似度
        
        Args:
            embeddings1: 第一组向量
            embeddings2: 第二组向量
            
        Returns:
            torch.Tensor: 相似度矩阵
        """
        normalized_embeddings1 = torch.nn.functional.normalize(embeddings1, p=2, dim=1)
        normalized_embeddings2 = torch.nn.functional.normalize(embeddings2, p=2, dim=1)
        return torch.matmul(normalized_embeddings1, normalized_embeddings2.transpose(0, 1))

    def calculate_priority_score(self, memory: Dict) -> float:
        """计算记忆的优先级分数"""
        current_time = datetime.now()
        W_t, W_i, W_a = (self.priority_config['weights'][k] for k in ['time', 'importance', 'access'])
        
        # 时间因子
        last_access = datetime.fromisoformat(
            memory.get('access_stats', {}).get('last_access', current_time.isoformat())
        )
        days_since_access = (current_time - last_access).days
        time_factor = 1 / (1 + math.exp(days_since_access - 7))
        
        # 重要性因子
        memory_type = memory.get('type', 'general')
        base_weight = self.priority_config['type_weights'].get(memory_type, 0.5)
        key_weight = 0.2 if memory.get('is_key_memory', False) else 0.0
        importance_factor = base_weight * (1 + key_weight)
        
        # 访问因子 - 使用持久化的访问统计
        access_count = memory.get('access_stats', {}).get('count', 0)
        access_factor = min(1, access_count / 10 + 5 / (1 + days_since_access))
        
        # 最终优先级分数
        priority_score = W_t * time_factor + W_i * importance_factor + W_a * access_factor
        return round(priority_score, 3)

    def update_memory_priority(self, user_id: str, memory_id: int) -> Dict:
        """更新指定记忆的优先级信息"""
        try:
            if user_id not in self.memories:
                return {"success": False, "error": "User not found"}
                
            if 0 <= memory_id < len(self.memories[user_id]):
                memory = self.memories[user_id][memory_id]
                memory['last_access'] = datetime.now().isoformat()
                memory['access_count'] = memory.get('access_count', 0) + 1
                memory['priority_score'] = self.calculate_priority_score(memory)
                self.save_memories()
                return {"success": True, "priority_score": memory['priority_score']}
            
            return {"success": False, "error": "Memory ID out of range"}
            
        except Exception as e:
            logging.error(f"Error updating memory priority: {str(e)}")
            return {"success": False, "error": str(e)}

    def clean_low_priority_memories(self, user_id: str) -> Dict:
        """清理低优先级记忆"""
        try:
            if user_id not in self.memories:
                return {"success": False, "error": "User not found"}
                
            memories = self.memories[user_id]
            if len(memories) <= self.priority_config['max_memories']:
                return {"success": True, "cleaned": 0}
            
            # 更新所有记忆的优先级分数
            for memory in memories:
                if 'priority_score' not in memory:
                    memory['priority_score'] = self.calculate_priority_score(memory)
            
            # 按优先级排序
            memories.sort(key=lambda x: x['priority_score'])
            
            # 删除低优先级记忆
            to_delete = len(memories) - self.priority_config['max_memories']
            deleted_memories = memories[:to_delete]
            self.memories[user_id] = memories[to_delete:]
            
            # 保存更改
            self.save_memories()
            
            logging.info(f"Cleaned {to_delete} low-priority memories for user {user_id}")
            return {
                "success": True,
                "cleaned": to_delete,
                "lowest_score": deleted_memories[0]['priority_score'] if deleted_memories else None,
                "highest_score": deleted_memories[-1]['priority_score'] if deleted_memories else None
            }
            
        except Exception as e:
            logging.error(f"Error cleaning low priority memories: {str(e)}")
            return {"success": False, "error": str(e)} 

    def access_memory(self, user_id: str, memory_id: int):
        """访问记忆的便捷方法"""
        return self.update_memory_access(user_id, memory_id, 'read')

    def _initialize_memory_stats(self):
        """初始化或迁移记忆访问统计"""
        for user_id, memories in self.memories.items():
            for memory in memories:
                if 'access_stats' not in memory:
                    memory['access_stats'] = {
                        'count': memory.get('access_count', 0),  # 迁移旧的访问计数
                        'first_access': memory.get('created_at', datetime.now().isoformat()),
                        'last_access': memory.get('last_access', datetime.now().isoformat()),
                        'access_history': []  # 用于记录详细的访问历史
                    }
                # 删除旧的统计字段
                memory.pop('access_count', None)
                memory.pop('last_access', None)
        self.save_memories()

    def update_memory_access(self, user_id: str, memory_id: int, access_type: str = 'read') -> Dict:
        """更新记忆的访问统计信息
        
        Args:
            user_id: 用户ID
            memory_id: 记忆ID
            access_type: 访问类型（'read', 'write', 'reference'等）
        """
        try:
            if user_id not in self.memories:
                return {"success": False, "error": "User not found"}
                
            if 0 <= memory_id < len(self.memories[user_id]):
                memory = self.memories[user_id][memory_id]
                current_time = datetime.now().isoformat()
                
                # 确保存在访问统计结构
                if 'access_stats' not in memory:
                    memory['access_stats'] = {
                        'count': 0,
                        'first_access': current_time,
                        'last_access': current_time,
                        'access_history': []
                    }
                
                # 更新访问统计
                memory['access_stats']['count'] += 1
                memory['access_stats']['last_access'] = current_time
                
                # 记录访问历史（保留最近10次）
                memory['access_stats']['access_history'].append({
                    'timestamp': current_time,
                    'type': access_type
                })
                memory['access_stats']['access_history'] = \
                    memory['access_stats']['access_history'][-10:]
                
                # 更新优先级分数
                memory['priority_score'] = self.calculate_priority_score(memory)
                
                self.save_memories()
                return {
                    "success": True, 
                    "access_count": memory['access_stats']['count'],
                    "priority_score": memory['priority_score']
                }
            
            return {"success": False, "error": "Memory ID out of range"}
            
        except Exception as e:
            logging.error(f"Error updating memory access: {str(e)}")
            return {"success": False, "error": str(e)} 