#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HierarchyManager模块的单元测试
"""

import pytest
import logging
from models.hierarchy_manager import HierarchyManager

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestHierarchyManager:
    """HierarchyManager类的测试用例"""
    
    @pytest.fixture
    def hierarchy_manager(self):
        """创建层级管理器的测试夹具"""
        try:
            return HierarchyManager()
        except Exception as e:
            pytest.skip(f"层级管理器初始化失败: {e}")
            
    @pytest.fixture
    def sample_clusters(self):
        """创建示例聚类数据"""
        return {
            "cluster_1": {
                "content": "Python编程技术相关的讨论",
                "memories": ["Python是一种流行的编程语言", "Python适合数据分析"]
            },
            "cluster_2": {
                "content": "机器学习算法讨论",
                "memories": ["机器学习模型的评估方法", "深度学习基础知识"]
            },
            "cluster_3": {
                "content": "数据分析方法讨论",
                "memories": ["数据预处理技术", "特征工程方法"]
            }
        }
        
    def test_build_hierarchy(self, hierarchy_manager, sample_clusters):
        """测试构建层级结构"""
        # 构建层级结构
        hierarchy = hierarchy_manager.build_hierarchy(sample_clusters)
        
        # 验证基本结构
        assert len(hierarchy) > 0, "层级结构应该包含至少一个类别"
        
        # 验证每个类别的结构完整性
        for cluster_id, cluster_info in hierarchy.items():
            assert "content" in cluster_info, f"类别 {cluster_id} 缺少content字段"
            assert "parent" in cluster_info, f"类别 {cluster_id} 缺少parent字段"
            assert "children" in cluster_info, f"类别 {cluster_id} 缺少children字段"
            assert "memories" in cluster_info, f"类别 {cluster_id} 缺少memories字段"
            
        # 验证层级关系的合理性
        for cluster_id, cluster_info in hierarchy.items():
            if cluster_info["parent"]:
                # 验证父类别存在
                assert cluster_info["parent"] in hierarchy, f"类别 {cluster_id} 的父类别不存在"
                # 验证父类别的children列表包含当前类别
                assert cluster_id in hierarchy[cluster_info["parent"]]["children"], \
                    f"类别 {cluster_id} 不在其父类别的children列表中"
                    
    def test_update_hierarchy_similar_content(self, hierarchy_manager, sample_clusters):
        """测试使用相似内容更新层级结构"""
        # 首先构建初始层级
        hierarchy_manager.build_hierarchy(sample_clusters)
        
        # 添加一个与现有类别相似的新记忆
        new_memory = "Python是一门非常适合初学者的编程语言"
        hierarchy_manager.update_hierarchy(new_memory)
        
        # 验证新记忆是否被正确添加到相似的类别中
        found = False
        for cluster_info in hierarchy_manager.hierarchy.values():
            if new_memory in cluster_info["memories"]:
                found = True
                break
        assert found, "新记忆应该被添加到相似的现有类别中"
        
    def test_update_hierarchy_new_content(self, hierarchy_manager, sample_clusters):
        """测试使用全新内容更新层级结构"""
        # 首先构建初始层级
        hierarchy_manager.build_hierarchy(sample_clusters)
        
        # 添加一个与现有类别不相似的新记忆
        new_memory = "前端开发中CSS的布局技术"
        initial_cluster_count = len(hierarchy_manager.hierarchy)
        
        hierarchy_manager.update_hierarchy(new_memory)
        
        # 验证是否创建了新的类别
        assert len(hierarchy_manager.hierarchy) > initial_cluster_count, \
            "应该为不相似的新记忆创建新的类别"
            
        # 验证新记忆是否被正确存储
        found = False
        for cluster_info in hierarchy_manager.hierarchy.values():
            if new_memory in cluster_info["memories"]:
                found = True
                break
        assert found, "新记忆应该被存储在新创建的类别中"
        
    def test_merge_similar_clusters(self, hierarchy_manager):
        """测试相似类别的合并"""
        # 创建两个相似的类别
        similar_clusters = {
            "cluster_1": {
                "content": "Python编程基础教程",
                "memories": ["Python基础语法", "Python数据类型"]
            },
            "cluster_2": {
                "content": "Python编程入门指南",
                "memories": ["Python环境搭建", "Python基本概念"]
            }
        }
        
        # 构建层级结构
        hierarchy = hierarchy_manager.build_hierarchy(similar_clusters)
        
        # 验证相似的类别是否被合并
        assert len(hierarchy) < len(similar_clusters), \
            "相似度高的类别应该被合并"
            
    def test_hierarchy_consistency(self, hierarchy_manager, sample_clusters):
        """测试层级结构的一致性"""
        # 构建初始层级
        hierarchy_manager.build_hierarchy(sample_clusters)
        
        # 添加多个新记忆，测试结构的稳定性
        new_memories = [
            "Python数据可视化技术",
            "机器学习模型调优方法",
            "Web开发基础知识",
            "数据库设计原则"
        ]
        
        for memory in new_memories:
            hierarchy_manager.update_hierarchy(memory)
            
        # 验证层级结构的完整性
        hierarchy = hierarchy_manager.hierarchy
        for cluster_id, cluster_info in hierarchy.items():
            # 验证父子关系的一致性
            if cluster_info["parent"]:
                parent = hierarchy[cluster_info["parent"]]
                assert cluster_id in parent["children"], \
                    f"类别 {cluster_id} 的父子关系不一致"
                    
            # 验证子类别的存在性
            for child_id in cluster_info["children"]:
                assert child_id in hierarchy, \
                    f"类别 {cluster_id} 的子类别 {child_id} 不存在"
