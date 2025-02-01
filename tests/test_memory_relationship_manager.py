import unittest
import pytest
import torch
from models.memory_relationship_manager import MemoryRelationshipManager

class TestMemoryRelationshipManager(unittest.TestCase):
    def setUp(self):
        """
        测试前的初始化工作
        """
        self.manager = MemoryRelationshipManager()
        self.test_memories = [
            "这是第一条测试记忆",
            "这是第二条相关的测试记忆",
            "这是一条完全不相关的记忆内容"
        ]

    def test_calculate_relationships(self):
        """
        测试记忆关系计算功能
        """
        # 测试相似记忆的关系分数
        similarity = self.manager.calculate_relationships(
            self.test_memories[0], 
            self.test_memories[1]
        )
        assert 0 <= similarity <= 1
        
        # 测试不相关记忆的关系分数
        similarity = self.manager.calculate_relationships(
            self.test_memories[0], 
            self.test_memories[2]
        )
        assert 0 <= similarity <= 1
        assert similarity < 0.5  # 不相关的记忆应该有较低的相似度

    def test_build_relationship_graph(self):
        """
        测试关系图构建功能
        """
        graph = self.manager.build_relationship_graph(self.test_memories)
        
        # 验证图的基本属性
        assert isinstance(graph, dict)
        assert len(graph) == len(self.test_memories)
        
        # 验证每个记忆都有关系列表
        for memory in self.test_memories:
            assert memory in graph
            assert isinstance(graph[memory], list)
            
        # 验证关系的对称性
        for memory1 in graph:
            for memory2, score in graph[memory1]:
                # 在memory2的关系列表中应该能找到memory1
                found = False
                for m, s in graph[memory2]:
                    if m == memory1:
                        assert abs(s - score) < 1e-6  # 分数应该相同
                        found = True
                        break
                assert found

    def test_get_related_memories(self):
        """
        测试相关记忆检索功能
        """
        # 构建关系图
        self.manager.build_relationship_graph(self.test_memories)
        
        # 测试获取相关记忆
        related = self.manager.get_related_memories(self.test_memories[0], threshold=0.5)
        assert isinstance(related, list)
        
        # 验证返回的相关记忆格式
        for memory, score in related:
            assert isinstance(memory, str)
            assert isinstance(score, float)
            assert 0 <= score <= 1
            assert score >= 0.5  # 应该满足阈值要求

    def test_edge_cases(self):
        """
        测试边界情况
        """
        # 测试空记忆列表
        empty_graph = self.manager.build_relationship_graph([])
        assert isinstance(empty_graph, dict)
        assert len(empty_graph) == 0
        
        # 测试单个记忆
        single_memory = ["单个测试记忆"]
        single_graph = self.manager.build_relationship_graph(single_memory)
        assert len(single_graph) == 1
        assert len(single_graph[single_memory[0]]) == 0
        
        # 测试相同的记忆
        same_memories = ["测试记忆", "测试记忆"]
        same_graph = self.manager.build_relationship_graph(same_memories)
        assert len(same_graph) == 2
        
        # 验证相同记忆的相似度为1
        similarity = self.manager.calculate_relationships(
            same_memories[0],
            same_memories[1]
        )
        assert abs(similarity - 1.0) < 1e-6

if __name__ == '__main__':
    pytest.main([__file__])
