"""
综合召回模块

该模块负责实现综合召回功能，通过分析查询内容，返回最相关记忆簇的摘要信息。
综合召回适用于用户需要全面了解某个主题的场景。
"""
from typing import Optional
import logging
from models.memory_manager import MemoryManager


class ComprehensiveRetriever:
    """
    综合召回类：负责从记忆库中检索与查询最相关的记忆簇摘要。
    
    主要功能：
    1. 根据用户查询返回最相关的记忆簇摘要
    2. 通过记忆簇提供主题的整体概览
    3. 支持自定义相似度阈值
    """
    
    def __init__(self, memory_manager: MemoryManager,
                 min_similarity_score: float = 0.5):
        """
        初始化综合召回器
        
        参数:
            memory_manager: 记忆管理器实例
            min_similarity_score: 最小相似度阈值，低于此阈值的记忆簇将被过滤
        """
        self.memory_manager = memory_manager
        self.min_similarity_score = min_similarity_score
        logging.info("ComprehensiveRetriever initialized with min_score=%.2f",
                    min_similarity_score)
    
    def retrieve_comprehensive(self, query: str, user_id: str = "default_user") -> Optional[str]:
        """
        根据查询综合召回相关记忆簇的摘要
        
        参数:
            query: 用户查询文本
            user_id: 用户ID，默认为"default_user"
            
        返回:
            str: 相关记忆簇的摘要信息，如果没有找到相关簇则返回None
        """
        try:
            # 获取查询的向量表示
            query_vector = self.memory_manager.encode([query])[0]
            
            # 使用聚类管理器找到相关的记忆簇
            cluster_id = self.memory_manager.cluster_manager.find_related_cluster(query_vector)
            
            if cluster_id == -1:
                logging.info(f"No related memory cluster found for query: {query}")
                return None
                
            # 获取簇的摘要信息
            cluster_info = self.memory_manager.cluster_manager.get_cluster_info(cluster_id)
            if not cluster_info:
                logging.warning(f"Cluster {cluster_id} not found")
                return None
                
            # 获取簇中的记忆向量ID列表
            memory_vector_ids = cluster_info['vectors']
            
            # 获取对应的记忆内容
            memory_contents = []
            for vector_id in memory_vector_ids:
                # 通过向量ID获取记忆ID
                memory_id = self.memory_manager.get_memory_id_by_vector_id(user_id, vector_id)
                if memory_id:
                    memory = self.memory_manager.get_memory_by_id(user_id, memory_id)
                    if memory and 'content' in memory:
                        memory_contents.append(memory['content'])
            
            if not memory_contents:
                logging.warning(f"No memory contents found in cluster {cluster_id}")
                return None
            
            # 生成并返回簇的摘要
            summary = self.memory_manager.memory_summarizer.generate_cluster_summary(memory_contents)
            logging.info(f"Generated summary for cluster {cluster_id} with {len(memory_contents)} memories")
            return summary
            
        except Exception as e:
            logging.error(f"Error in comprehensive retrieval for query '{query}': {str(e)}")
            return None
    
    def set_similarity_threshold(self, threshold: float) -> None:
        """
        设置相似度阈值
        
        参数:
            threshold: 新的相似度阈值（0.0-1.0）
        """
        if 0.0 <= threshold <= 1.0:
            self.min_similarity_score = threshold
            logging.info(f"Similarity threshold updated to {threshold}")
        else:
            logging.error(f"Invalid similarity threshold: {threshold}. Must be between 0.0 and 1.0")
            raise ValueError("Similarity threshold must be between 0.0 and 1.0")
