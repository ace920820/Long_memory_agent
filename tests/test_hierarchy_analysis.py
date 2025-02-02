#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试记忆层级结构分析
用于验证 HierarchyManager 对实际记忆数据的聚类和层级分析效果
"""

import os
import sys
import logging
import json
from typing import Dict, List
from models.hierarchy_manager import HierarchyManager

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
logger = logging.getLogger(__name__)

def load_user_memories() -> List[Dict]:
    """加载用户记忆数据"""
    try:
        memories_path = os.path.join("config", "user_memories.json")
        with open(memories_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 获取默认用户的记忆
            if isinstance(data, dict) and 'default_user' in data:
                return data['default_user']
            else:
                logger.error(f"无效的记忆数据格式: {type(data)}")
                return []
    except Exception as e:
        logger.error(f"加载用户记忆数据失败: {str(e)}")
        return []

def test_memory_hierarchy_analysis():
    """测试记忆层级结构分析"""
    try:
        # 初始化层级管理器
        hierarchy_manager = HierarchyManager()
        
        # 加载记忆数据
        memories = load_user_memories()
        logger.info(f"加载了 {len(memories)} 条记忆数据")
        
        # 提取记忆内容
        memory_contents = []
        for memory in memories:
            if isinstance(memory, dict):
                content = memory.get('content', str(memory))
                memory_contents.append(content)
            else:
                content = str(memory)
                memory_contents.append(content)
        
        logger.info(f"提取了 {len(memory_contents)} 条有效记忆内容")
        
        if not memory_contents:
            logger.error("没有找到有效的记忆内容")
            return
        
        # 分析层级关系
        logger.info("开始分析记忆层级关系...")
        hierarchies = hierarchy_manager.analyze_content_hierarchies(memory_contents)
        
        # 输出分析结果
        logger.info("\n=== 记忆层级分析结果 ===")
        
        # 统计簇的信息
        clusters = {}
        for i, hierarchy in enumerate(hierarchies):
            cluster_name = hierarchy['cluster']
            if cluster_name not in clusters:
                clusters[cluster_name] = {
                    'members': [],
                    'parent_connections': set(),
                    'child_connections': set()
                }
            
            # 添加成员
            clusters[cluster_name]['members'].append({
                'content': memory_contents[i][:100] + "..." if len(memory_contents[i]) > 100 else memory_contents[i],
                'parent': hierarchy['parent'],
                'children': hierarchy['children']
            })
            
            # 记录父子关系
            if hierarchy['parent'] is not None:
                clusters[cluster_name]['parent_connections'].add(hierarchy['parent'])
            clusters[cluster_name]['child_connections'].update(hierarchy['children'])
        
        # 输出每个簇的详细信息
        for cluster_name, cluster_info in clusters.items():
            logger.info(f"\n簇名称: {cluster_name}")
            logger.info(f"成员数量: {len(cluster_info['members'])}")
            logger.info(f"父节点连接数: {len(cluster_info['parent_connections'])}")
            logger.info(f"子节点连接数: {len(cluster_info['child_connections'])}")
            logger.info("成员列表:")
            for member in cluster_info['members']:
                logger.info(f"  - 内容: {member['content']}")
                if member['parent'] is not None:
                    parent_content = memory_contents[member['parent']][:50] + "..."
                    logger.info(f"    父节点: {parent_content}")
                if member['children']:
                    logger.info(f"    子节点数: {len(member['children'])}")
        
        # 验证基本属性
        assert len(hierarchies) == len(memory_contents), "层级结构数量应与有效记忆数量相同"
        for hierarchy in hierarchies:
            assert 'cluster' in hierarchy, "每个层级结构都应该有簇标签"
            assert 'parent' in hierarchy, "每个层级结构都应该有父节点信息"
            assert 'children' in hierarchy, "每个层级结构都应该有子节点信息"
            assert 'similarity_scores' in hierarchy, "每个层级结构都应该有相似度分数"
        
        logger.info("\n测试完成！")
        
    except Exception as e:
        logger.error(f"测试失败: {str(e)}")
        raise

if __name__ == "__main__":
    test_memory_hierarchy_analysis()
