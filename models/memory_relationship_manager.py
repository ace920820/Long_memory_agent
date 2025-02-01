#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
记忆关系管理模块
负责计算和管理不同记忆项之间的语义关联
"""

import logging
import yaml
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel
import torch
import networkx as nx
from scipy.spatial.distance import cosine

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MemoryRelationshipManager:
    """
    记忆关系管理器
    负责计算不同记忆项之间的语义关联，构建记忆网络
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        初始化记忆关系管理器
        Args:
            config_path: 配置文件路径
        """
        try:
            # 加载配置文件
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 获取embedding配置
            embedding_config = config['embedding']
            self.model_path = embedding_config['model_path']
            self.dimension = embedding_config['dimension']
            self.similarity_threshold = embedding_config['similarity_threshold']
            
            logger.info(f"正在加载文本向量化模型: {self.model_path}")
            # 初始化BERT模型和分词器
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModel.from_pretrained(self.model_path)
            
            # 初始化TF-IDF向量化器
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                token_pattern=r'(?u)\b\w+\b'
            )
            
            # 初始化关系图
            self.relationship_graph = nx.Graph()
            
            logger.info("记忆关系管理器初始化成功")
            
        except Exception as e:
            logger.error(f"初始化记忆关系管理器失败: {str(e)}")
            raise

    def _encode_text_bert(self, text: str) -> np.ndarray:
        """
        使用BERT模型将文本编码为向量
        Args:
            text: 输入文本
        Returns:
            文本向量
        """
        try:
            # 对文本进行编码
            inputs = self.tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True)
            
            # 获取文本向量
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :].numpy()  # 使用[CLS]标记的输出作为文本向量
                
            return embeddings[0]
            
        except Exception as e:
            logger.error(f"BERT文本编码失败: {str(e)}")
            raise

    def _calculate_tfidf_similarity(self, text1: str, text2: str) -> float:
        """
        计算两段文本的TF-IDF相似度
        Args:
            text1: 第一段文本
            text2: 第二段文本
        Returns:
            相似度分数 (0-1)
        """
        try:
            # 将文本转换为TF-IDF向量
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([text1, text2])
            
            # 计算余弦相似度
            similarity = (tfidf_matrix * tfidf_matrix.T).A[0, 1]
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"计算TF-IDF相似度失败: {str(e)}")
            return 0.0

    def calculate_relationships(self, memory1: str, memory2: str) -> float:
        """
        计算两个记忆之间的语义关联度
        Args:
            memory1: 第一个记忆内容
            memory2: 第二个记忆内容
        Returns:
            语义关联度（0-1之间的浮点数）
        """
        try:
            if not memory1 or not memory2:
                logger.warning("记忆内容为空")
                return 0.0

            logger.info(f"计算记忆关联度:")
            logger.info(f"记忆1: {memory1[:50]}...")
            logger.info(f"记忆2: {memory2[:50]}...")
            
            # BERT相似度计算
            bert_similarity = self._calculate_bert_similarity(memory1, memory2)
            logger.info(f"BERT相似度: {bert_similarity:.4f}")
            
            # TF-IDF相似度计算
            tfidf_similarity = self._calculate_tfidf_similarity(memory1, memory2)
            logger.info(f"TF-IDF相似度: {tfidf_similarity:.4f}")
            
            # 组合两种相似度
            similarity = 0.7 * bert_similarity + 0.3 * tfidf_similarity
            logger.info(f"综合相似度: {similarity:.4f}")
            
            return similarity
            
        except Exception as e:
            logger.error(f"计算记忆关联度失败: {str(e)}")
            return 0.0

    def _calculate_bert_similarity(self, text1: str, text2: str) -> float:
        """
        使用BERT模型计算文本相似度
        """
        try:
            # 对文本进行编码
            inputs1 = self.tokenizer(text1, return_tensors="pt", 
                                   truncation=True, max_length=512)
            inputs2 = self.tokenizer(text2, return_tensors="pt", 
                                   truncation=True, max_length=512)
            
            # 获取文本向量表示
            with torch.no_grad():
                outputs1 = self.model(**inputs1)
                outputs2 = self.model(**inputs2)
            
            # 使用[CLS]向量作为文本表示
            embedding1 = outputs1.last_hidden_state[:, 0, :].numpy()
            embedding2 = outputs2.last_hidden_state[:, 0, :].numpy()
            
            # 计算余弦相似度
            similarity = 1 - cosine(embedding1.flatten(), embedding2.flatten())
            return float(similarity)
            
        except Exception as e:
            logger.error(f"BERT相似度计算失败: {str(e)}")
            return 0.0

    def _calculate_tfidf_similarity(self, text1: str, text2: str) -> float:
        """
        使用TF-IDF计算文本相似度
        """
        try:
            # 创建TF-IDF向量化器
            vectorizer = TfidfVectorizer()
            
            # 转换文本为TF-IDF向量
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            
            # 计算余弦相似度
            similarity = float((tfidf_matrix * tfidf_matrix.T).A[0, 1])
            return similarity
            
        except Exception as e:
            logger.error(f"TF-IDF相似度计算失败: {str(e)}")
            return 0.0

    def build_relationship_graph(self, memories: List[str]) -> Dict:
        """
        构建完整的记忆间关系图
        Args:
            memories: 记忆列表
        Returns:
            记忆关系字典，键为记忆内容，值为相关记忆及其权重的列表
        """
        try:
            logger.info(f"开始构建记忆关系图，共有 {len(memories)} 条记忆")
            
            if not memories:
                logger.info("记忆列表为空，返回空字典")
                return {}
            
            # 构建记忆关系字典，使用索引来区分相同内容的记忆
            relationship_dict = {}
            for i, memory in enumerate(memories):
                # 如果记忆内容已存在，添加索引后缀
                key = memory
                if memory in relationship_dict:
                    suffix = 1
                    while f"{memory}_{suffix}" in relationship_dict:
                        suffix += 1
                    key = f"{memory}_{suffix}"
                    logger.debug(f"处理重复记忆，添加后缀: {key}")
                relationship_dict[key] = []
            
            # 计算记忆间的关系
            memory_keys = list(relationship_dict.keys())
            for i, memory1 in enumerate(memory_keys):
                for j, memory2 in enumerate(memory_keys[i + 1:], start=i + 1):
                    # 去除后缀计算相似度
                    memory1_content = memory1.rsplit('_', 1)[0]
                    memory2_content = memory2.rsplit('_', 1)[0]
                    weight = self.calculate_relationships(memory1_content, memory2_content)
                    logger.debug(f"记忆 {memory1} 和记忆 {memory2} 的相似度: {weight:.4f}")
                    
                    # 只添加权重超过阈值的关系
                    if weight > self.similarity_threshold:
                        relationship_dict[memory1].append((memory2, weight))
                        relationship_dict[memory2].append((memory1, weight))
                        logger.debug(f"添加关系: {memory1[:20]}... <-> {memory2[:20]}...")
            
            logger.info(f"记忆关系图构建完成，共有 {len(relationship_dict)} 个记忆节点")
            return relationship_dict
            
        except Exception as e:
            logger.error(f"构建记忆关系图失败: {str(e)}")
            raise

    def get_related_memories(self, memory: str, threshold: float = 0.5) -> List[Tuple[str, float]]:
        """
        获取与给定记忆最相关的其他记忆
        Args:
            memory: 目标记忆
            threshold: 相似度阈值，只返回相似度高于此值的记忆
        Returns:
            相关记忆及其权重的列表
        """
        try:
            logger.info(f"开始查找与记忆相关的其他记忆，阈值: {threshold}")
            
            # 在关系字典中查找记忆（可能带有后缀）
            memory_key = None
            for key in self.relationship_graph:
                if key == memory or key.startswith(memory + "_"):
                    memory_key = key
                    break
            
            if memory_key is None:
                logger.warning(f"未找到记忆: {memory[:50]}...")
                return []
            
            # 获取所有相关记忆
            related_memories = []
            for related_memory, weight in self.relationship_graph[memory_key]:
                if weight >= threshold:
                    # 去除后缀返回原始记忆内容
                    original_memory = related_memory.rsplit('_', 1)[0]
                    related_memories.append((original_memory, weight))
            
            # 按相似度降序排序
            related_memories.sort(key=lambda x: x[1], reverse=True)
            
            logger.info(f"找到 {len(related_memories)} 个相关记忆")
            return related_memories
            
        except Exception as e:
            logger.error(f"获取相关记忆失败: {str(e)}")
            return []

    def get_memory_clusters(self, memories: List[str], min_similarity: float = 0.6) -> List[List[str]]:
        """
        基于相似度将记忆聚类
        Args:
            memories: 记忆列表
            min_similarity: 最小相似度阈值
        Returns:
            记忆簇列表
        """
        try:
            # 构建关系图
            self.build_relationship_graph(memories)
            
            # 使用社区检测算法找出记忆簇
            communities = list(nx.community.louvain_communities(self.relationship_graph))
            
            # 将节点ID转换为实际的记忆内容
            memory_clusters = []
            for community in communities:
                cluster = [self.relationship_graph.nodes[node_id]["content"] for node_id in community]
                memory_clusters.append(cluster)
            
            return memory_clusters
            
        except Exception as e:
            logger.error(f"获取记忆簇失败: {str(e)}")
            return []
