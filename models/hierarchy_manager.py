#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
层级结构管理模块
负责维护记忆的层级结构，支持动态调整和更新
"""

import json
import logging
import os
import yaml
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np
from scipy.spatial.distance import cosine
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HierarchyManager:
    """
    层级结构管理器
    负责构建和维护记忆的层级结构，支持动态调整
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        初始化层级管理器
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
            # 初始化模型和分词器
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModel.from_pretrained(self.model_path)
            logger.info("文本向量化模型加载成功")
            
            # 层级结构，使用字典存储，格式：
            # {
            #   "cluster_id": {
            #     "content": str,      # 类别内容摘要
            #     "parent": str,       # 父类别ID
            #     "children": List[str] # 子类别ID列表
            #     "memories": List[str] # 该类别包含的记忆ID列表
            #   }
            # }
            self.hierarchy = {}
            
        except Exception as e:
            logger.error(f"初始化层级管理器失败: {str(e)}")
            raise

    def _encode_text(self, text: str) -> np.ndarray:
        """
        将文本编码为向量
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
                
            return embeddings[0]  # 返回第一个样本的向量（因为batch_size=1）
            
        except Exception as e:
            logger.error(f"文本编码失败: {str(e)}")
            raise

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        使用 BERT 模型获取文本的嵌入向量
        Args:
            texts: 单个文本或文本列表
        Returns:
            文本的嵌入向量
        """
        try:
            # 确保输入是列表形式
            if isinstance(texts, str):
                texts = [texts]
            
            # 对文本进行分词和编码
            encoded_input = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            # 将输入移到GPU（如果可用）
            if torch.cuda.is_available():
                encoded_input = {k: v.cuda() for k, v in encoded_input.items()}
                self.model = self.model.cuda()
            
            # 获取BERT输出
            with torch.no_grad():
                model_output = self.model(**encoded_input)
                
            # 使用[CLS]标记的输出作为句子表示
            sentence_embeddings = model_output[0][:, 0]
            
            # 如果在GPU上，移回CPU
            if torch.cuda.is_available():
                sentence_embeddings = sentence_embeddings.cpu()
            
            # 转换为numpy数组
            embeddings = sentence_embeddings.numpy()
            
            logger.debug(f"成功编码 {len(texts)} 条文本")
            return embeddings
            
        except Exception as e:
            logger.error(f"获取文本嵌入向量失败: {str(e)}")
            raise

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        计算两段文本的相似度
        Args:
            text1: 第一段文本
            text2: 第二段文本
        Returns:
            相似度分数 (0-1)
        """
        try:
            # 计算文本向量
            embedding1 = self._encode_text(text1)
            embedding2 = self._encode_text(text2)
            
            # 计算余弦相似度
            similarity = 1 - cosine(embedding1, embedding2)
            logger.debug(f"文本相似度计算 - text1: {text1[:50]}... text2: {text2[:50]}... 相似度: {similarity}")
            
            return float(similarity)
        except Exception as e:
            logger.error(f"计算文本相似度失败: {str(e)}")
            return 0.0

    def _find_best_parent(self, cluster_content: str) -> Optional[str]:
        """
        为新类别找到最合适的父类别
        Args:
            cluster_content: 类别内容
        Returns:
            最合适的父类别ID，如果没有合适的则返回None
        """
        best_parent = None
        max_similarity = self.similarity_threshold
        
        for cluster_id, cluster_info in self.hierarchy.items():
            similarity = self._calculate_similarity(cluster_content, cluster_info["content"])
            if similarity > max_similarity:
                max_similarity = similarity
                best_parent = cluster_id
                
        logger.info(f"为类别内容 '{cluster_content[:50]}...' 找到最佳父类别: {best_parent}")
        return best_parent

    def _merge_clusters(self, cluster1_id: str, cluster2_id: str) -> str:
        """
        合并两个相似的类别
        Args:
            cluster1_id: 第一个类别ID
            cluster2_id: 第二个类别ID
        Returns:
            合并后的类别ID
        """
        logger.info(f"开始合并类别 {cluster1_id} 和 {cluster2_id}")
        
        try:
            # 获取类别信息
            cluster1 = self.hierarchy[cluster1_id]
            cluster2 = self.hierarchy[cluster2_id]
            
            # 计算相似度以确认是否应该合并
            similarity = self._calculate_similarity(cluster1["content"], cluster2["content"])
            if similarity <= self.similarity_threshold:
                logger.info(f"类别相似度 {similarity} 低于阈值 {self.similarity_threshold}，取消合并")
                return cluster1_id
            
            # 合并内容
            merged_content = f"{cluster1['content']} {cluster2['content']}"
            
            # 合并子类别和记忆
            merged_children = list(set(cluster1["children"] + cluster2["children"]))
            merged_memories = list(set(cluster1["memories"] + cluster2["memories"]))
            
            # 创建新的合并类别
            new_cluster_id = f"merged_{cluster1_id}_{cluster2_id}"
            self.hierarchy[new_cluster_id] = {
                "content": merged_content,
                "parent": None,
                "children": merged_children,
                "memories": merged_memories
            }
            
            # 更新子类别的父类别指向
            for child_id in merged_children:
                if child_id in self.hierarchy:
                    self.hierarchy[child_id]["parent"] = new_cluster_id
                    
            # 删除原类别
            del self.hierarchy[cluster1_id]
            del self.hierarchy[cluster2_id]
            
            logger.info(f"类别合并完成，新类别ID: {new_cluster_id}")
            return new_cluster_id
            
        except Exception as e:
            logger.error(f"合并类别失败: {str(e)}")
            raise

    def build_hierarchy(self, clusters: Dict[str, Dict]) -> Dict:
        """
        从聚类结果构建层级结构
        Args:
            clusters: 聚类结果字典
        Returns:
            构建的层级结构
        """
        logger.info("开始构建层级结构")
        
        try:
            # 重置当前层级结构
            self.hierarchy = {}
            
            # 首先添加所有类别
            for cluster_id, cluster_info in clusters.items():
                self.hierarchy[cluster_id] = {
                    "content": cluster_info["content"],
                    "parent": None,
                    "children": [],
                    "memories": cluster_info.get("memories", [])
                }
                
            # 构建层级关系
            for cluster_id, cluster_info in self.hierarchy.items():
                # 寻找最佳父类别
                parent_id = self._find_best_parent(cluster_info["content"])
                if parent_id and parent_id != cluster_id:
                    cluster_info["parent"] = parent_id
                    self.hierarchy[parent_id]["children"].append(cluster_id)
                    
            # 检查是否需要合并类别
            merged_clusters = set()
            for cluster1_id in list(self.hierarchy.keys()):
                if cluster1_id in merged_clusters:
                    continue
                    
                for cluster2_id in list(self.hierarchy.keys()):
                    if (cluster2_id not in merged_clusters and 
                        cluster1_id != cluster2_id and
                        self._calculate_similarity(
                            self.hierarchy[cluster1_id]["content"],
                            self.hierarchy[cluster2_id]["content"]
                        ) > self.similarity_threshold):
                        # 合并类别
                        new_cluster_id = self._merge_clusters(cluster1_id, cluster2_id)
                        merged_clusters.add(cluster1_id)
                        merged_clusters.add(cluster2_id)
                        break
                        
            logger.info(f"层级结构构建完成，共有 {len(self.hierarchy)} 个类别")
            return self.hierarchy
            
        except Exception as e:
            logger.error(f"构建层级结构失败: {str(e)}")
            raise

    def update_hierarchy(self, new_memory: str) -> None:
        """
        根据新记忆更新层级结构
        Args:
            new_memory: 新记忆内容
        """
        logger.info(f"开始处理新记忆: {new_memory[:50]}...")
        
        try:
            # 寻找最相似的类别
            best_cluster_id = None
            max_similarity = 0.0  # 初始化为0，而不是相似度阈值
            
            for cluster_id, cluster_info in self.hierarchy.items():
                similarity = self._calculate_similarity(new_memory, cluster_info["content"])
                logger.info(f"与类别 {cluster_id} 的相似度: {similarity}")
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_cluster_id = cluster_id
            
            # 只有当相似度超过阈值时才添加到现有类别
            if best_cluster_id and max_similarity > self.similarity_threshold:
                # 将新记忆添加到现有类别
                logger.info(f"将新记忆添加到类别 {best_cluster_id}，相似度: {max_similarity}")
                self.hierarchy[best_cluster_id]["memories"].append(new_memory)
                # 更新类别内容，加入新记忆的信息
                self.hierarchy[best_cluster_id]["content"] = f"{self.hierarchy[best_cluster_id]['content']} {new_memory}"
            else:
                # 创建新类别
                new_cluster_id = f"cluster_{len(self.hierarchy)}"
                logger.info(f"为新记忆创建新类别: {new_cluster_id}")
                
                self.hierarchy[new_cluster_id] = {
                    "content": new_memory,
                    "parent": None,
                    "children": [],
                    "memories": [new_memory]
                }
                
                # 寻找潜在的父类别（使用较低的相似度阈值）
                parent_threshold = self.similarity_threshold * 0.8  # 降低父类别的相似度要求
                best_parent_id = None
                max_parent_similarity = parent_threshold
                
                for cluster_id, cluster_info in self.hierarchy.items():
                    if cluster_id != new_cluster_id:  # 不与自己比较
                        similarity = self._calculate_similarity(new_memory, cluster_info["content"])
                        if similarity > max_parent_similarity:
                            max_parent_similarity = similarity
                            best_parent_id = cluster_id
                
                if best_parent_id:
                    logger.info(f"将新类别 {new_cluster_id} 设置为 {best_parent_id} 的子类别")
                    self.hierarchy[new_cluster_id]["parent"] = best_parent_id
                    self.hierarchy[best_parent_id]["children"].append(new_cluster_id)
                    
        except Exception as e:
            logger.error(f"更新层级结构失败: {str(e)}")
            raise

    def analyze_content_hierarchies(self, contents: List[str]) -> List[Dict]:
        """
        分析多个记忆内容的层级关系
        Args:
            contents: 记忆内容列表
        Returns:
            层级信息列表，每个元素包含 cluster, parent, children 等信息
        """
        try:
            logger.info(f"开始分析 {len(contents)} 条记忆的层级关系")
            
            # 使用 BERT 计算内容的嵌入向量
            embeddings = np.array([self._encode_text(content) for content in contents])
            
            # 使用 UMAP 进行降维，减少维度灾难的影响
            from umap import UMAP
            n_neighbors = min(15, len(contents) - 1)  # 根据数据量调整邻居数
            umap = UMAP(
                n_components=min(16, len(contents) - 1),  # 降维后的维度
                n_neighbors=n_neighbors,
                min_dist=0.1,
                metric='cosine',
                random_state=42
            )
            embeddings_reduced = umap.fit_transform(embeddings)
            
            # 使用 DBSCAN 进行聚类，调整参数使其更容易形成簇
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            embeddings_scaled = scaler.fit_transform(embeddings_reduced)
            
            clustering = DBSCAN(
                eps=0.5,           # 增大邻域半径
                min_samples=2,     # 保持较小的最小样本数
                metric='euclidean'
            ).fit(embeddings_scaled)
            
            clusters = clustering.labels_
            
            # 如果所有点都是噪声点，则使用 KMeans 进行聚类
            if len(set(clusters)) <= 1:  # 如果只有一个簇或者都是噪声点
                logger.warning("DBSCAN 聚类效果不理想，尝试使用 KMeans")
                from sklearn.cluster import KMeans
                n_clusters = min(5, len(contents))  # 最多5个簇
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(embeddings_scaled)
            
            # 构建层级结构
            hierarchies = []
            for i, content in enumerate(contents):
                cluster = int(clusters[i])
                # 为每个簇生成一个有意义的名称
                if cluster >= 0:
                    # 获取同簇的所有内容
                    cluster_contents = [contents[j] for j, c in enumerate(clusters) if c == cluster]
                    # 使用最短的内容作为簇的名称
                    shortest_content = min(cluster_contents, key=len)
                    if len(shortest_content) > 10:
                        shortest_content = shortest_content[:10] + "..."
                    cluster_name = f"簇_{cluster}: {shortest_content}"
                else:
                    cluster_name = "未分类"
                
                # 计算与其他记忆的关系（使用降维后的向量）
                similarities = cosine_similarity(
                    [embeddings_scaled[i]], 
                    embeddings_scaled
                )[0]
                
                # 找出最相似的记忆作为父节点（排除自身）
                similarities[i] = -1  # 将自身的相似度设为最小
                parent_idx = np.argmax(similarities)
                parent_sim = similarities[parent_idx]
                
                # 调整相似度阈值，使其更容易建立父子关系
                threshold = 0.5  # 进一步降低阈值
                
                # 只有当相似度超过阈值时才建立父子关系
                parent = None
                if parent_sim > threshold:
                    parent = parent_idx
                
                # 找出作为子节点的记忆
                children = []
                for j, sim in enumerate(similarities):
                    if j != i and sim > threshold:
                        children.append(j)
                
                # 构建层级信息
                hierarchy = {
                    'cluster': cluster_name,
                    'parent': parent,
                    'children': children,
                    'similarity_scores': {
                        j: sim for j, sim in enumerate(similarities)
                        if j != i and sim > 0.3  # 降低相似度阈值，显示更多关系
                    }
                }
                
                hierarchies.append(hierarchy)
                logger.debug(f"记忆 {i} 的层级信息: {hierarchy}")
            
            # 记录聚类结果统计
            unique_clusters = set(clusters)
            cluster_sizes = {c: sum(1 for x in clusters if x == c) for c in unique_clusters}
            logger.info(f"聚类结果: 共 {len(unique_clusters)} 个簇")
            for c, size in cluster_sizes.items():
                logger.info(f"簇_{c}: {size} 个记忆")
            
            return hierarchies
            
        except Exception as e:
            logger.error(f"分析记忆层级关系失败: {str(e)}")
            raise
