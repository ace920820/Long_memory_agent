import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import torch
from transformers import BertTokenizer, BertModel
from sentence_transformers import util
import re
import math

class MemoryManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", 
                 memory_file: str = "config/user_memories.json",
                 similarity_threshold: float = 0.5,
                 new_memory_similarity_threshold: float = 0.9):
        """初始化记忆管理器
        
        Args:
            model_name: 使用的嵌入模型名称
            memory_file: 记忆存储文件路径
            similarity_threshold: 相似度阈值
        """
        self.model = SentenceTransformer(model_name)
        self.memory_file = memory_file
        self.similarity_threshold = similarity_threshold
        self.new_memory_similarity_threshold = new_memory_similarity_threshold
        self.memories = self.load_memories()
        
        # 初始化向量索引
        self.dimension = 384  # all-MiniLM-L6-v2 的向量维度
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
        try:
            if user_id not in self.indices:
                self.indices[user_id] = faiss.IndexFlatL2(self.dimension)
                
            if user_id in self.memories and self.memories[user_id]:
                memory_texts = [m['content'] for m in self.memories[user_id]]
                vectors = self.model.encode(memory_texts)
                self.indices[user_id].add(vectors.astype('float32'))
                
        except Exception as e:
            logging.error(f"Error creating index for user {user_id}: {str(e)}")
            self.indices[user_id] = faiss.IndexFlatL2(self.dimension)

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
                self.indices[user_id] = faiss.IndexFlatL2(self.dimension)

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
                new_vector = self.model.encode([content], convert_to_tensor=True)
                
                # 获取现有记忆的内容
                existing_contents = [m['content'] for m in self.memories[user_id]]
                existing_vectors = self.model.encode(existing_contents, convert_to_tensor=True)
                
                # 计算相似度
                similarities = util.pytorch_cos_sim(new_vector, existing_vectors)[0]
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
                vector = self.model.encode([content]).astype('float32')
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
                        # 更新访问统计
                        self.update_memory_access(user_id, idx, 'reference')
                        
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

    def get_relevant_memories(self, user_id: str, query: str, top_k: int = 5) -> List[Dict]:
        """获取与查询相关的记忆
        
        Args:
            user_id: 用户ID
            query: 查询文本
            top_k: 返回的最相关记忆数量
            
        Returns:
            相关记忆列表，每个记忆包含相似度分数
        """
        try:
            # 获取用户的所有记忆
            all_memories = self.get_all_memories(user_id)
            if not all_memories:
                return []

            # 计算查询的嵌入向量
            query_embedding = self.model.encode([query], convert_to_tensor=True)
            
            # 为每条记忆计算相似度并记录
            memory_scores = []
            for idx, memory in enumerate(all_memories):
                content = memory['content']
                memory_embedding = self.model.encode([content], convert_to_tensor=True)
                similarity = util.pytorch_cos_sim(query_embedding, memory_embedding)[0][0].item()
                
                # 记录详细的相似度信息
                memory_scores.append({
                    'content': content,
                    'similarity': similarity,
                    'memory_id': memory.get('id', '未知'),
                    'timestamp': memory.get('timestamp', '未知'),
                    'index': idx  # 添加索引以便后续更新访问统计
                })
                logging.debug(f"记忆相似度计算:\n"
                             f"查询: {query}\n"
                             f"记忆: {content}\n"
                             f"相似度: {similarity:.4f}")

            # 按相似度排序
            memory_scores.sort(key=lambda x: x['similarity'], reverse=True)
            
            # 获取相似度超过阈值的记忆
            relevant_memories = []
            for score in memory_scores[:top_k]:
                if score['similarity'] >= self.similarity_threshold:
                    memory_idx = score['index']
                    memory = all_memories[memory_idx].copy()
                    memory['similarity'] = score['similarity']
                    
                    # 更新记忆访问统计
                    self.update_memory_access(user_id, memory_idx, 'reference')
                    
                    relevant_memories.append(memory)
                    logging.info(f"记忆匹配 (得分: {score['similarity']:.4f}):\n"
                               f"文本: {score['content']}")
                else:
                    logging.info(f"记忆因相似度过低被过滤 (得分: {score['similarity']:.4f}):\n"
                               f"文本: {score['content']}")

            return relevant_memories
            
        except Exception as e:
            logging.error(f"获取相关记忆失败: {str(e)}")
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