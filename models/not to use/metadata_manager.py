from typing import Dict, List, Optional, Set
import numpy as np
from collections import Counter
from datetime import datetime
import jieba.analyse
from loguru import logger
import os


"""
【暂不使用】

"""

class MetadataManager:
    def __init__(
        self,
        decay_factor: float = 0.1,
        max_keywords: int = 10,
        min_keyword_freq: int = 2,
        time_weight_factor: float = 0.3
    ):
        """初始化元数据管理器

        Args:
            decay_factor: 时间衰减因子
            max_keywords: 每个簇保留的最大关键词数量
            min_keyword_freq: 关键词最小出现频率
            time_weight_factor: 时间权重因子
        """
        self.decay_factor = decay_factor
        self.max_keywords = max_keywords
        self.min_keyword_freq = min_keyword_freq
        self.time_weight_factor = time_weight_factor

        # 存储簇的元数据
        self.cluster_metadata: Dict[int, Dict] = {}
        
        # 初始化jieba分析器
        self.stop_words_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "stopwords.txt")
        jieba.analyse.set_stop_words(self.stop_words_path)  # 设置停用词
        self.text_rank = jieba.analyse.textrank
        self.tfidf = jieba.analyse.extract_tags

    def _calculate_time_weight(self, last_update: datetime) -> float:
        """计算时间权重
        
        Args:
            last_update: 最后更新时间

        Returns:
            时间权重分数
        """
        time_diff = (datetime.now() - last_update).total_seconds() / (24 * 3600)  # 转换为天数
        weight = np.exp(-self.decay_factor * time_diff)
        return float(weight)

    def _extract_keywords(self, texts: List[str]) -> List[str]:
        """从文本中提取关键词
        
        Args:
            texts: 文本列表

        Returns:
            关键词列表
        """
        # 合并所有文本
        combined_text = " ".join(texts)
        
        # 使用TextRank和TF-IDF结合的方式提取关键词
        textrank_keywords = set(self.text_rank(combined_text, topK=self.max_keywords))
        tfidf_keywords = set(self.tfidf(combined_text, topK=self.max_keywords))
        
        # 取两种方法的并集
        all_keywords = textrank_keywords.union(tfidf_keywords)
        
        # 统计词频
        word_freq = Counter()
        for text in texts:
            words = jieba.lcut(text)
            word_freq.update(words)
        
        # 筛选高频关键词
        filtered_keywords = [
            word for word in all_keywords 
            if word_freq[word] >= self.min_keyword_freq
        ]
        
        # 按词频排序并限制数量
        sorted_keywords = sorted(
            filtered_keywords,
            key=lambda x: word_freq[x],
            reverse=True
        )[:self.max_keywords]
        
        return sorted_keywords

    def initialize_metadata(self, cluster_id: int, texts: List[str]) -> None:
        """初始化簇的元数据
        
        Args:
            cluster_id: 簇ID
            texts: 簇中的文本列表
        """
        keywords = self._extract_keywords(texts)
        
        self.cluster_metadata[cluster_id] = {
            'size': len(texts),
            'last_update': datetime.now(),
            'keywords': keywords,
            'time_weight': 1.0,
            'access_count': 0
        }
        
        logger.info(f"Initialized metadata for cluster {cluster_id} with {len(keywords)} keywords")

    def update_metadata(self, cluster_id: int, texts: Optional[List[str]] = None) -> None:
        """更新簇的元数据
        
        Args:
            cluster_id: 簇ID
            texts: 可选的新文本列表
        """
        if cluster_id not in self.cluster_metadata:
            if texts is None:
                raise ValueError(f"Cluster {cluster_id} not found and no texts provided")
            self.initialize_metadata(cluster_id, texts)
            return

        metadata = self.cluster_metadata[cluster_id]
        
        # 更新时间权重
        metadata['last_update'] = datetime.now()
        metadata['time_weight'] = self._calculate_time_weight(metadata['last_update'])
        
        # 如果提供了新文本，更新关键词
        if texts is not None:
            metadata['size'] = len(texts)
            metadata['keywords'] = self._extract_keywords(texts)
        
        logger.info(f"Updated metadata for cluster {cluster_id}")

    def get_cluster_keywords(self, cluster_id: int) -> List[str]:
        """获取簇的关键词列表
        
        Args:
            cluster_id: 簇ID

        Returns:
            关键词列表
        """
        if cluster_id not in self.cluster_metadata:
            return []
        return self.cluster_metadata[cluster_id]['keywords']

    def get_cluster_priority(self, cluster_id: int) -> float:
        """计算簇的优先级分数
        
        Args:
            cluster_id: 簇ID

        Returns:
            优先级分数
        """
        if cluster_id not in self.cluster_metadata:
            return 0.0
            
        metadata = self.cluster_metadata[cluster_id]
        
        # 计算时间权重
        time_weight = metadata['time_weight']
        
        # 计算大小权重
        size_weight = np.log1p(metadata['size'])  # 使用log避免大簇权重过高
        
        # 计算访问频率权重
        access_weight = np.log1p(metadata['access_count'])
        
        # 综合计算优先级分数
        priority = (
            self.time_weight_factor * time_weight +
            (1 - self.time_weight_factor) * (size_weight + access_weight) / 2
        )
        
        return float(priority)

    def record_access(self, cluster_id: int) -> None:
        """记录簇被访问
        
        Args:
            cluster_id: 簇ID
        """
        if cluster_id in self.cluster_metadata:
            self.cluster_metadata[cluster_id]['access_count'] += 1
            logger.debug(f"Recorded access for cluster {cluster_id}")

    def get_metadata(self, cluster_id: int) -> Optional[Dict]:
        """获取簇的完整元数据
        
        Args:
            cluster_id: 簇ID

        Returns:
            元数据字典
        """
        return self.cluster_metadata.get(cluster_id)

if __name__ == "__main__":
    # 初始化元数据管理器
    metadata_manager = MetadataManager()

    # 初始化簇的元数据
    texts = ["这是第一段文本", "这是第二段相关文本"]
    metadata_manager.initialize_metadata(cluster_id=1, texts=texts)

    # 获取簇的关键词
    keywords = metadata_manager.get_cluster_keywords(cluster_id=1)

    # 更新元数据
    metadata_manager.update_metadata(cluster_id=1, texts=["新的文本内容"])

    # 获取簇的优先级
    priority = metadata_manager.get_cluster_priority(cluster_id=1)

    # 记录簇的访问
    metadata_manager.record_access(cluster_id=1)