from typing import Dict, List, Optional
import numpy as np
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from loguru import logger
from .metadata_manager import MetadataManager


class MemorySummarizer:
    def __init__(
        self,
        metadata_manager: MetadataManager,
        min_cluster_size: int = 3,
        max_summary_sentences: int = 5,
        similarity_threshold: float = 0.8
    ):
        """初始化记忆摘要器

        Args:
            metadata_manager: 元数据管理器实例
            min_cluster_size: 触发摘要生成的最小簇大小
            max_summary_sentences: 摘要最大句子数
            similarity_threshold: 句子去重的相似度阈值
        """
        self.metadata_manager = metadata_manager
        self.min_cluster_size = min_cluster_size
        self.max_summary_sentences = max_summary_sentences
        self.similarity_threshold = similarity_threshold

        # 存储簇的摘要
        self.summaries: Dict[int, str] = {}
        
        # 初始化TF-IDF向量化器
        self.tfidf = TfidfVectorizer(
            tokenizer=lambda x: jieba.lcut(x),
            stop_words='english'  # 可以添加中文停用词
        )

    def _split_sentences(self, text: str) -> List[str]:
        """将文本分割成句子

        Args:
            text: 输入文本

        Returns:
            句子列表
        """
        # 简单的基于标点符号的分句
        delimiters = ['。', '！', '？', '!', '?', '\n']
        sentences = []
        start = 0
        
        for i, char in enumerate(text):
            if char in delimiters:
                sentence = text[start:i+1].strip()
                if sentence:
                    sentences.append(sentence)
                start = i + 1
                
        # 处理最后一个句子
        if start < len(text):
            sentence = text[start:].strip()
            if sentence:
                sentences.append(sentence)
                
        return sentences

    def _calculate_sentence_scores(
        self,
        sentences: List[str],
        keywords: List[str]
    ) -> List[float]:
        """计算句子的重要性分数

        Args:
            sentences: 句子列表
            keywords: 关键词列表

        Returns:
            句子分数列表
        """
        if not sentences:
            return []

        # 计算TF-IDF分数
        try:
            tfidf_matrix = self.tfidf.fit_transform(sentences)
            sentence_importance = np.array(tfidf_matrix.sum(axis=1)).flatten()
        except:
            # 如果TF-IDF计算失败，使用简单的关键词匹配
            sentence_importance = np.zeros(len(sentences))
            for i, sentence in enumerate(sentences):
                words = set(jieba.lcut(sentence))
                sentence_importance[i] = len(words.intersection(keywords))

        # 归一化分数
        if sentence_importance.max() > 0:
            sentence_importance = sentence_importance / sentence_importance.max()

        return sentence_importance.tolist()

    def _remove_redundant_sentences(
        self,
        sentences: List[str],
        scores: List[float]
    ) -> List[str]:
        """去除冗余句子

        Args:
            sentences: 候选句子列表
            scores: 句子分数列表

        Returns:
            去重后的句子列表
        """
        if not sentences:
            return []

        selected_sentences = []
        selected_indices = set()

        # 按分数排序
        sorted_pairs = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True
        )

        for idx, score in sorted_pairs:
            if len(selected_sentences) >= self.max_summary_sentences:
                break

            current_sentence = sentences[idx]
            is_redundant = False

            # 检查与已选句子的相似度
            for selected_idx in selected_indices:
                selected_sentence = sentences[selected_idx]
                
                # 计算简单的词重叠率
                current_words = set(jieba.lcut(current_sentence))
                selected_words = set(jieba.lcut(selected_sentence))
                
                if len(current_words) == 0:
                    continue
                    
                overlap_ratio = len(current_words.intersection(selected_words)) / len(current_words)
                
                if overlap_ratio > self.similarity_threshold:
                    is_redundant = True
                    break

            if not is_redundant:
                selected_sentences.append(current_sentence)
                selected_indices.add(idx)

        return selected_sentences

    def _generate_summary(self, cluster_texts: List[str], keywords: List[str]) -> str:
        """生成摘要

        Args:
            cluster_texts: 簇中的文本列表
            keywords: 关键词列表

        Returns:
            生成的摘要
        """
        # 分句
        all_sentences = []
        for text in cluster_texts:
            all_sentences.extend(self._split_sentences(text))

        if not all_sentences:
            return ""

        # 计算句子分数
        sentence_scores = self._calculate_sentence_scores(all_sentences, keywords)
        
        # 去除冗余句子
        selected_sentences = self._remove_redundant_sentences(all_sentences, sentence_scores)
        
        if not selected_sentences:
            return ""

        # 使用模板组织摘要
        summary_template = "该记忆簇包含{count}条记录，主要涉及{keywords}等内容。主要内容：{content}"
        
        content = "；".join(selected_sentences)
        keywords_text = "、".join(keywords[:3])  # 只使用前3个关键词
        
        summary = summary_template.format(
            count=len(cluster_texts),
            keywords=keywords_text,
            content=content
        )
        
        return summary

    def update_summary(self, cluster_id: int, texts: List[str]) -> None:
        """更新簇的摘要

        Args:
            cluster_id: 簇ID
            texts: 簇中的文本列表
        """
        if len(texts) < self.min_cluster_size:
            self.summaries.pop(cluster_id, None)
            return

        # 获取关键词
        keywords = self.metadata_manager.get_cluster_keywords(cluster_id)
        if not keywords:
            logger.warning(f"No keywords found for cluster {cluster_id}")
            return

        # 生成摘要
        summary = self._generate_summary(texts, keywords)
        if summary:
            self.summaries[cluster_id] = summary
            logger.info(f"Updated summary for cluster {cluster_id}")
        else:
            logger.warning(f"Failed to generate summary for cluster {cluster_id}")

    def get_cluster_summary(self, cluster_id: int) -> Optional[str]:
        """获取簇的摘要

        Args:
            cluster_id: 簇ID

        Returns:
            簇的摘要，如果不存在则返回None
        """
        return self.summaries.get(cluster_id)


if __name__ == "__main__":
    # 示例用法
    from metadata_manager import MetadataManager
    
    # 初始化管理器
    metadata_manager = MetadataManager()
    summarizer = MemorySummarizer(metadata_manager)
    
    # 示例文本
    texts = [
        "人工智能技术正在快速发展。深度学习模型在图像识别领域取得重大突破。",
        "机器学习算法在自然语言处理任务中表现出色。",
        "神经网络模型在语音识别方面达到了接近人类的水平。"
    ]
    
    # 初始化元数据
    cluster_id = 1
    metadata_manager.initialize_metadata(cluster_id, texts)
    
    # 更新摘要
    summarizer.update_summary(cluster_id, texts)
    
    # 获取摘要
    summary = summarizer.get_cluster_summary(cluster_id)
    print(f"生成的摘要：\n{summary}")
