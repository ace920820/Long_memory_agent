#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试RaptorModule的search方法
验证修复后的嵌入模型访问和重排序功能
"""

import logging
import sys
import os
from typing import List, Dict

# 设置详细的日志输出
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# 导入必要的模块
from models.raptor_module import RaptorModule

def test_raptor_search():
    """测试RaptorModule的search方法"""
    try:
        # 初始化RaptorModule
        logging.info("正在初始化RaptorModule...")
        raptor = RaptorModule()
        
        # 加载测试文档
        test_docs = [
            "Raptor是一个基于树状结构的知识检索系统，能够高效地存储和检索大量信息。",
            "Python是一种广泛使用的高级编程语言，强调代码的可读性和简洁性。",
            "人工智能(AI)是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
            "机器学习是人工智能的一个子领域，主要关注如何让计算机系统自动从数据中学习和改进。",
            "深度学习是机器学习的一种方法，使用多层神经网络处理复杂的模式和抽象。"
        ]
        
        # 将文档添加到RaptorModule
        logging.info("正在添加测试文档...")
        for doc in test_docs:
            raptor.add(doc)
        
        # 测试查询
        test_queries = [
            "什么是Raptor系统?",
            "Python的特点是什么?",
            "人工智能和机器学习的关系是什么?",
            "深度学习是如何工作的?"
        ]
        
        # 执行查询测试
        logging.info("开始执行查询测试...")
        for query in test_queries:
            logging.info(f"\n测试查询: '{query}'")
            try:
                result = raptor.search(query)
                if isinstance(result, dict) and 'context' in result:
                    logging.info(f"查询结果共有 {len(result['context'])} 个匹配项")
                    for i, ctx in enumerate(result['context']):
                        score = result.get('rerank_scores', [0])[i] if i < len(result.get('rerank_scores', [])) else 0
                        logging.info(f"  结果 {i+1} (分数: {score:.4f}): {ctx}")
                elif isinstance(result, str):
                    logging.info(f"查询结果 (字符串格式): {result}")
                else:
                    logging.info(f"查询结果类型: {type(result)}")
            except Exception as e:
                logging.error(f"查询'{query}'时发生错误: {str(e)}")
        
        logging.info("测试完成！")
        return True
        
    except Exception as e:
        logging.error(f"测试过程中发生错误: {str(e)}")
        return False

if __name__ == "__main__":
    test_raptor_search()
