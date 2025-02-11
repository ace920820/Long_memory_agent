#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
记忆图谱管理器测试模块
"""

import unittest
import pytest
import os
from models.graph_manager import GraphManager

class TestGraphManager(unittest.TestCase):
    """
    记忆图谱管理器测试类
    """
    
    def setUp(self):
        """
        测试前的初始化工作
        """
        self.graph_manager = GraphManager()
        # 降低合并阈值，便于测试
        self.graph_manager.merge_threshold = 0.5
        self.test_memories = [
            "这是第一条测试记忆",
            "这是第二条相关的测试记忆",
            "这是一条完全不相关的记忆内容",
            "这是第四条测试记忆，与第一条相关"
        ]
        self.test_metadata = {
            "timestamp": "2025-02-01",
            "source": "test",
            "tags": ["测试", "记忆"]
        }

    def test_add_memory(self):
        """
        测试添加记忆节点功能
        """
        # 测试添加单个记忆
        success = self.graph_manager.add_memory(
            self.test_memories[0],
            self.test_metadata
        )
        assert success
        assert len(self.graph_manager.graph.nodes) == 1
        
        # 验证节点属性
        node_data = self.graph_manager.graph.nodes[0]
        assert node_data['content'] == self.test_memories[0]
        assert node_data['metadata'] == self.test_metadata
        
        # 测试添加多个记忆
        for memory in self.test_memories[1:]:
            success = self.graph_manager.add_memory(memory)
            assert success
        
        # 验证节点数量
        assert len(self.graph_manager.graph.nodes) == len(self.test_memories)

    def test_remove_memory(self):
        """
        测试移除记忆节点功能
        """
        # 先添加测试记忆
        for memory in self.test_memories:
            self.graph_manager.add_memory(memory)
        
        initial_nodes = len(self.graph_manager.graph.nodes)
        
        # 测试移除存在的节点
        success = self.graph_manager.remove_memory(0)
        assert success
        assert len(self.graph_manager.graph.nodes) == initial_nodes - 1
        
        # 测试移除不存在的节点
        success = self.graph_manager.remove_memory(999)
        assert not success
        assert len(self.graph_manager.graph.nodes) == initial_nodes - 1

    def test_merge_memories(self):
        """
        测试合并记忆节点功能
        """
        # 添加两个高度相似的记忆
        self.graph_manager.add_memory("Python是一种广受欢迎的编程语言")
        self.graph_manager.add_memory("Python是一种流行的编程语言")
        
        # 测试合并
        new_id = self.graph_manager.merge_memories(0, 1)
        assert new_id is not None
        
        # 验证合并结果
        merged_content = self.graph_manager.graph.nodes[new_id]['content']
        assert "Python" in merged_content
        assert len(self.graph_manager.graph.nodes) == 1
        
        # 测试合并不相似的记忆
        self.graph_manager.add_memory("今天天气真不错")
        new_id = self.graph_manager.merge_memories(new_id, 2)
        assert new_id is None  # 不相似的记忆不应该被合并

    def test_relationships(self):
        """
        测试记忆关系管理功能
        """
        # 添加相关的记忆
        self.graph_manager.add_memory("机器学习是AI的一个分支")
        self.graph_manager.add_memory("深度学习是机器学习的一部分")
        
        # 验证关系
        assert self.graph_manager.graph.has_edge(0, 1)
        weight = self.graph_manager.graph[0][1]['weight']
        assert 0 <= weight <= 1
        
        # 添加不相关的记忆
        self.graph_manager.add_memory("今天天气真好")
        assert not self.graph_manager.graph.has_edge(0, 2)

    def test_hierarchy(self):
        """
        测试层级结构管理功能
        """
        # 添加具有层级关系的记忆
        self.graph_manager.add_memory("计算机科学")
        self.graph_manager.add_memory("编程语言")
        self.graph_manager.add_memory("Python语言")
        
        # 验证层级信息
        for node in self.graph_manager.graph.nodes:
            assert 'hierarchy' in self.graph_manager.graph.nodes[node]['metadata']

    def test_visualization(self):
        """
        测试图谱可视化功能
        """
        # 添加测试数据
        for memory in self.test_memories:
            self.graph_manager.add_memory(memory)
        
        # 测试可视化
        output_path = "test_graph.png"
        success = self.graph_manager.visualize(output_path)
        assert success
        assert os.path.exists(output_path)
        
        # 清理测试文件
        if os.path.exists(output_path):
            os.remove(output_path)

    def test_edge_cases(self):
        """
        测试边界情况
        """
        # 测试空图的可视化
        success = self.graph_manager.visualize("empty_graph.png")
        assert success
        
        # 测试添加空记忆
        success = self.graph_manager.add_memory("")
        assert not success
        
        # 测试添加None
        success = self.graph_manager.add_memory(None)
        assert not success
        
        # 清理测试文件
        if os.path.exists("empty_graph.png"):
            os.remove("empty_graph.png")

if __name__ == '__main__':
    pytest.main([__file__])
