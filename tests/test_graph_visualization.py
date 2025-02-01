#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
图谱可视化测试
用于测试记忆图谱的可视化功能，包括记忆聚类和层级关系的展示
"""

import unittest
import json
import os
import logging
from models.graph_manager import GraphManager

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestGraphVisualization(unittest.TestCase):
    """
    测试 GraphManager 的图谱可视化功能
    包括记忆的聚类展示和层级关系可视化
    """

    def setUp(self):
        """
        测试前的初始化工作
        1. 创建 GraphManager 实例
        2. 读取用户记忆数据
        3. 构建记忆图谱
        """
        logger.info("初始化图谱可视化测试...")
        
        # 创建 GraphManager 实例
        self.graph_manager = GraphManager()
        
        # 读取 user_memories.json 数据
        with open('config/user_memories.json', 'r', encoding='utf-8') as f:
            self.user_memories = json.load(f)
            logger.info(f"成功读取用户记忆数据，共 {len(self.user_memories['default_user'])} 条记忆")

        # 添加记忆节点到图谱
        for memory in self.user_memories['default_user']:
            # 构建元数据
            metadata = {
                'id': memory['id'],
                'type': memory['type'],
                'timestamp': memory['timestamp'],
                'cluster': memory.get('cluster', '未分类'),  # 添加簇信息
                'access_stats': memory.get('access_stats', {})
            }
            
            # 添加记忆节点
            success = self.graph_manager.add_memory_with_metadata(
                memory['content'],
                metadata
            )
            if not success:
                logger.warning(f"添加记忆节点失败: {memory['id']}")

    def test_visualization(self):
        """
        测试图谱可视化功能
        1. 生成包含聚类信息的图谱
        2. 验证图像文件的生成
        3. 检查图谱中的节点和关系
        """
        logger.info("开始测试图谱可视化...")
        
        # 生成图谱
        output_path = "user_memory_graph.png"
        success = self.graph_manager.visualize(output_path)
        
        # 验证结果
        self.assertTrue(success, "图谱生成失败")
        self.assertTrue(os.path.exists(output_path), "图像文件未生成")
        
        # 验证图谱结构
        self.assertGreater(len(self.graph_manager.graph.nodes), 0, "图谱中没有节点")
        
        # 检查节点属性
        for node, data in self.graph_manager.graph.nodes(data=True):
            self.assertIn('content', data, f"节点 {node} 缺少内容")
            self.assertIn('cluster', data, f"节点 {node} 缺少簇信息")
            self.assertIn('type', data, f"节点 {node} 缺少类型信息")
        
        logger.info(f"图谱可视化测试完成，输出文件: {output_path}")

if __name__ == '__main__':
    unittest.main()
