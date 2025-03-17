#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试Raptor的重排序和嵌入功能
直接测试distances_from_embeddings函数
"""

import logging
import sys
import os
import numpy as np
from typing import List, Dict

# 设置详细的日志输出
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# 导入要测试的距离计算函数
from raptor.utils import distances_from_embeddings

class MockEmbeddingModel:
    """模拟嵌入模型类，用于测试"""
    def create_embedding(self, text):
        """模拟生成嵌入向量的方法"""
        # 根据文本长度生成简单的随机向量
        np.random.seed(hash(text) % 10000)  # 使用文本hash作为随机种子，确保相同文本生成相同向量
        return np.random.rand(128).tolist()  # 生成128维随机向量

def test_distances_calculation():
    """测试距离计算功能"""
    try:
        logging.info("测试距离计算功能...")
        
        # 模拟文本和查询
        texts = [
            "Raptor是一个基于树状结构的知识检索系统，能够高效地存储和检索大量信息。",
            "Python是一种广泛使用的高级编程语言，强调代码的可读性和简洁性。",
            "人工智能(AI)是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
            "机器学习是人工智能的一个子领域，主要关注如何让计算机系统自动从数据中学习和改进。",
            "深度学习是机器学习的一种方法，使用多层神经网络处理复杂的模式和抽象。"
        ]
        query = "什么是机器学习和人工智能?"
        
        # 创建模拟嵌入模型
        mock_model = MockEmbeddingModel()
        
        # 生成嵌入向量
        query_embedding = mock_model.create_embedding(query)
        text_embeddings = [mock_model.create_embedding(text) for text in texts]
        
        # 测试距离计算
        distances = distances_from_embeddings(
            query_embedding=query_embedding,
            embeddings=text_embeddings,
            distance_metric="cosine",
            rerank=False
        )
        
        logging.info("原始余弦距离:")
        for i, (text, distance) in enumerate(zip(texts, distances)):
            logging.info(f"文本 {i+1} (距离: {distance:.4f}): {text[:50]}...")
        
        # 测试None值处理
        text_embeddings_with_none = text_embeddings.copy()
        text_embeddings_with_none[1] = None  # 将一个嵌入设为None
        
        distances_with_none = distances_from_embeddings(
            query_embedding=query_embedding,
            embeddings=text_embeddings_with_none,
            distance_metric="cosine",
            rerank=False
        )
        
        logging.info("\n处理None值的距离计算结果:")
        for i, (text, distance) in enumerate(zip(texts, distances_with_none)):
            logging.info(f"文本 {i+1} (距离: {distance:.4f}): {text[:50]}...")
        
        # 测试查询嵌入为None的情况
        distances_query_none = distances_from_embeddings(
            query_embedding=None,
            embeddings=text_embeddings,
            distance_metric="cosine",
            rerank=False
        )
        
        logging.info("\n查询嵌入为None的距离计算结果:")
        for i, (text, distance) in enumerate(zip(texts, distances_query_none)):
            logging.info(f"文本 {i+1} (距离: {distance:.4f}): {text[:50]}...")
        
        logging.info("测试成功完成!")
        return True
        
    except Exception as e:
        logging.error(f"测试过程中发生错误: {str(e)}")
        return False

if __name__ == "__main__":
    test_distances_calculation()
