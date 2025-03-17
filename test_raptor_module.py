#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试修复后的RaptorModule功能
包括嵌入向量获取和重排序功能
"""

import logging
import sys
import os
import time
from typing import List, Dict

# 设置详细的日志输出
logging.basicConfig(
    level=logging.DEBUG,  # 使用DEBUG级别以显示更多日志
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# 抑制第三方库的日志
logging.getLogger('transformers').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)

def test_raptor_search():
    """测试RaptorModule的搜索功能"""
    try:
        # 仅导入需要的模块
        from raptor.utils import distances_from_embeddings
        from raptor.EmbeddingModels import OpenAIEmbeddingModel
        import numpy as np
        
        # 创建模拟的嵌入向量
        logging.info("创建模拟的嵌入向量进行测试")
        
        # 设置随机种子以保证结果可重现
        np.random.seed(42)
        
        # 创建模拟的查询向量和文档向量
        query_embedding = np.random.rand(128).tolist()
        embeddings = [np.random.rand(128).tolist() for _ in range(5)]
        
        # 测试包含None值的嵌入向量
        embeddings_with_none = embeddings.copy()
        embeddings_with_none[2] = None
        
        # 创建模拟文本
        query_text = "这是一个测试查询"
        node_texts = [
            "第一个测试文档，与查询相关度较高",
            "第二个测试文档，相关度中等",
            "第三个测试文档，相关度较低",
            "第四个测试文档，几乎不相关",
            "第五个测试文档，完全不相关"
        ]
        
        # 测试距离计算（不使用重排序）
        logging.info("测试基本距离计算功能（不使用重排序）")
        distances = distances_from_embeddings(
            query_embedding=query_embedding,
            embeddings=embeddings,
            distance_metric="cosine",
            rerank=False
        )
        
        logging.info("计算得到的余弦距离:")
        for i, (text, distance) in enumerate(zip(node_texts, distances)):
            logging.info(f"  文档 {i+1} (距离: {distance:.4f}): {text}")
        
        # 测试处理None值
        logging.info("\n测试处理包含None值的嵌入向量")
        distances_with_none = distances_from_embeddings(
            query_embedding=query_embedding,
            embeddings=embeddings_with_none,
            distance_metric="cosine",
            rerank=False
        )
        
        logging.info("处理None值的距离计算结果:")
        for i, (text, distance) in enumerate(zip(node_texts, distances_with_none)):
            logging.info(f"  文档 {i+1} (距离: {distance:.4f}): {text}")
        
        # 尝试从RaptorConfigManager加载重排序模型
        try:
            logging.info("\n尝试加载重排序模型进行测试")
            from utils.raptor_config_manager import RaptorConfigManager
            config_manager = RaptorConfigManager()
            
            if config_manager.is_rerank_enabled():
                reranker = config_manager.get_reranker()
                reranker_tokenizer = config_manager.get_reranker_tokenizer()
                
                if reranker and reranker_tokenizer:
                    logging.info("成功加载重排序模型，测试重排序功能")
                    
                    # 使用重排序功能
                    distances, rerank_scores = distances_from_embeddings(
                        query_embedding=query_embedding,
                        embeddings=embeddings,
                        distance_metric="cosine",
                        rerank=True,
                        query_text=query_text,
                        node_texts=node_texts,
                        reranker=reranker,
                        reranker_tokenizer=reranker_tokenizer
                    )
                    
                    # 合并结果并排序
                    combined = list(zip(node_texts, distances, rerank_scores))
                    sorted_by_distance = sorted(combined, key=lambda x: x[1])
                    sorted_by_rerank = sorted(combined, key=lambda x: x[2], reverse=True)
                    
                    logging.info("\n基于余弦距离的排序结果:")
                    for i, (text, distance, score) in enumerate(sorted_by_distance):
                        logging.info(f"  排名 {i+1} (距离: {distance:.4f}): {text}")
                    
                    logging.info("\n基于重排序分数的排序结果:")
                    for i, (text, distance, score) in enumerate(sorted_by_rerank):
                        logging.info(f"  排名 {i+1} (分数: {score:.4f}): {text}")
                else:
                    logging.warning("未能加载重排序模型或分词器，跳过重排序测试")
            else:
                logging.warning("重排序功能未在配置中启用，跳过重排序测试")
        except Exception as e:
            logging.error(f"加载或使用重排序模型时发生错误: {str(e)}")
        
        logging.info("\n所有测试完成！")
        return True
        
    except Exception as e:
        logging.error(f"测试过程中发生错误: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    test_raptor_search()
    
    # 测试 RaptorModule 的 search 方法
    logging.info("\n开始测试 RaptorModule 的 search 方法")
    try:
        from models.raptor_module import RaptorModule
        
        # 创建 RaptorModule 实例
        rm = RaptorModule()
        logging.info("RaptorModule 实例创建成功")
        
        # 尝试对空树进行搜索（应该返回空列表并记录警告）
        logging.info("测试空树搜索处理...")
        results = rm.search("测试查询", top_k=5)
        
        if isinstance(results, list) and len(results) == 0:
            logging.info("✓ 空树测试通过：返回了空列表")
        else:
            logging.error(f"✗ 空树测试失败：返回了非空结果: {results}")
            
        logging.info("RaptorModule 测试完成")
    except Exception as e:
        logging.error(f"测试 RaptorModule 时发生错误: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
