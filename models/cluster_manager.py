from typing import Dict, List, Optional, Tuple
import numpy as np
import faiss
from loguru import logger


class ClusterManager:
    def __init__(
        self,
        min_cluster_size: int = 3,
        similarity_threshold: float = 0.75,
        vector_dim: int = 512
    ):
        """初始化簇管理器

        Args:
            min_cluster_size: 最小簇大小
            similarity_threshold: 归入已有簇的相似度阈值
            vector_dim: 向量维度
        """
        self.min_cluster_size = min_cluster_size
        self.similarity_threshold = similarity_threshold
        self.vector_dim = vector_dim

        # 初始化簇存储
        self.clusters: Dict[int, Dict] = {}  # cluster_id -> cluster_info
        self.next_cluster_id = 0

        # 使用FAISS进行向量检索
        self.index = faiss.IndexFlatIP(vector_dim)  # 内积相似度
        
        # 记录向量ID到簇ID的映射
        self.vector_to_cluster: Dict[int, int] = {}  # vector_id -> cluster_id
        self.next_vector_id = 0

    def _find_most_similar_cluster(self, vector: np.ndarray) -> Tuple[int, float]:
        """找到与给定向量最相似的簇

        Args:
            vector: 输入向量

        Returns:
            (最相似簇的ID, 相似度分数)
        """
        if self.index.ntotal == 0:
            return -1, 0.0

        # 搜索最相似的向量
        D, I = self.index.search(vector.reshape(1, -1), 1)
        most_similar_vector_id = I[0][0]
        similarity_score = float(D[0][0])

        # 获取对应的簇ID
        cluster_id = self.vector_to_cluster.get(most_similar_vector_id, -1)
        
        return cluster_id, similarity_score

    def add_memory_vector(self, vector: np.ndarray) -> int:
        """添加向量化后的记忆到簇中

        Args:
            vector: 记忆的向量表示

        Returns:
            记忆所属的簇ID
        """
        # 确保向量维度正确
        if vector.shape[0] != self.vector_dim:
            raise ValueError(f"Vector dimension mismatch. Expected {self.vector_dim}, got {vector.shape[0]}")
            
        # 找到最相似的簇
        closest_cluster_id, similarity = self._find_most_similar_cluster(vector)
        
        # 如果没有找到足够相似的簇，创建新簇
        if closest_cluster_id == -1 or similarity < self.similarity_threshold:
            cluster_id = self.next_cluster_id
            self.next_cluster_id += 1
            
            self.clusters[cluster_id] = {
                'center': vector.copy(),
                'size': 1,
                'vectors': [self.next_vector_id]
            }
        else:
            cluster_id = closest_cluster_id
            cluster = self.clusters[cluster_id]
            
            # 更新簇的信息
            cluster['center'] = (cluster['center'] * cluster['size'] + vector) / (cluster['size'] + 1)
            cluster['size'] += 1
            cluster['vectors'].append(self.next_vector_id)
        
        # 添加向量到索引
        self.index.add(vector.reshape(1, -1))
        self.vector_to_cluster[self.next_vector_id] = cluster_id
        self.next_vector_id += 1
        
        logger.info(f"Added memory vector to cluster {cluster_id} (size: {self.clusters[cluster_id]['size']})")
        return cluster_id

    def find_related_cluster(self, vector: np.ndarray) -> int:
        """查找与给定向量最相关的簇

        Args:
            vector: 查询向量

        Returns:
            最相关簇的ID，如果没有找到相关簇则返回-1
        """
        cluster_id, similarity = self._find_most_similar_cluster(vector)
        
        if similarity < self.similarity_threshold:
            return -1
            
        return cluster_id

    def get_cluster_info(self, cluster_id: int) -> Optional[Dict]:
        """获取簇的信息

        Args:
            cluster_id: 簇ID

        Returns:
            簇的信息字典，如果簇不存在则返回None
        """
        return self.clusters.get(cluster_id)

    def get_all_clusters(self) -> Dict[int, Dict]:
        """获取所有簇的信息

        Returns:
            所有簇的信息字典
        """
        return self.clusters.copy()


if __name__ == "__main__":
    # 初始化簇管理器
    cluster_manager = ClusterManager()

    # 创建示例向量
    vector1 = np.random.randn(512).astype(np.float32)  # 随机生成一个512维向量
    vector2 = vector1 + 0.1 * np.random.randn(512).astype(np.float32)  # 生成一个相似的向量

    # 添加向量到簇
    cluster_id1 = cluster_manager.add_memory_vector(vector1)
    cluster_id2 = cluster_manager.add_memory_vector(vector2)

    # 查找相关簇
    query_vector = vector1 + 0.2 * np.random.randn(512).astype(np.float32)
    found_cluster_id = cluster_manager.find_related_cluster(query_vector)

    # 获取簇信息
    cluster_info = cluster_manager.get_cluster_info(found_cluster_id)