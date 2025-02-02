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
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from models.hierarchy_manager import HierarchyManager

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
logger = logging.getLogger(__name__)

def visualize_memory_hierarchy(memory_contents: List[str], hierarchies: List[Dict], output_path: str = "memory_hierarchy.png"):
    """
    将记忆层级结构可视化为树形图（优化版本）
    
    Args:
        memory_contents: 记忆内容列表
        hierarchies: 层级结构信息
        output_path: 输出图片路径
    """
    # 预先创建所有节点和边的列表，避免重复操作
    nodes = []
    edges = []
    labels = {}
    levels = {}  # 节点层级缓存
    
    # 一次性构建图结构
    for i, (content, hierarchy) in enumerate(zip(memory_contents, hierarchies)):
        nodes.append(i)
        labels[i] = content[:30] + "..." if len(content) > 30 else content
        if hierarchy['parent'] is not None:
            edges.append((hierarchy['parent'], i))
            # 同时计算层级
            parent_level = levels.get(hierarchy['parent'], -1)
            levels[i] = parent_level + 1 if parent_level >= 0 else 0
        else:
            levels[i] = 0
    
    # 创建图并一次性添加所有节点和边
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    
    # 优化层级计算
    if not levels:  # 如果没有找到任何层级关系
        for node in G.nodes():
            levels[node] = len(nx.ancestors(G, node))
    
    max_level = max(levels.values())
    
    # 预计算每层节点
    nodes_by_level = {level: [] for level in range(max_level + 1)}
    for node, level in levels.items():
        nodes_by_level[level].append(node)
    
    # 设置绘图参数（减少重复设置）
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig = plt.figure(figsize=(20, 15), facecolor='white')
    ax = fig.add_subplot(111)
    
    # 使用更快的布局算法
    pos = {}
    for level, level_nodes in nodes_by_level.items():
        x = level / max_level
        y_positions = np.linspace(-0.8, 0.8, len(level_nodes))
        for node, y in zip(level_nodes, y_positions):
            pos[node] = np.array([x, y])
    
    # 一次性生成所有颜色
    level_colors = plt.cm.viridis(np.linspace(0, 1, max_level + 1))
    
    # 批量绘制边（减少绘制调用次数）
    nx.draw_networkx_edges(G, pos,
                          edge_color='gray',
                          arrows=True,
                          arrowsize=20,
                          width=1.5,
                          alpha=0.6,
                          connectionstyle="arc3,rad=0.1",  # 减小弧度以加快渲染
                          ax=ax)
    
    # 批量绘制节点（按层级）
    for level, nodes_in_level in nodes_by_level.items():
        if nodes_in_level:  # 避免空列表
            nx.draw_networkx_nodes(G, pos,
                                 nodelist=nodes_in_level,
                                 node_color=[level_colors[level]],
                                 node_size=2500,  # 稍微减小节点大小以加快渲染
                                 alpha=0.7,
                                 ax=ax)
    
    # 批量绘制标签
    nx.draw_networkx_labels(G, pos, labels,
                          font_size=8,
                          font_family='SimHei',
                          font_weight='bold',
                          ax=ax)
    
    # 优化图例（减少元素数量）
    step = max(1, max_level // 4)  # 只显示部分层级的图例
    legend_elements = [plt.Line2D([0], [0],
                                marker='o',
                                color='w',
                                label=f'第{i}层',
                                markerfacecolor=level_colors[i],
                                markersize=10)
                      for i in range(0, max_level + 1, step)]
    
    ax.legend(handles=legend_elements,
             loc='center left',
             bbox_to_anchor=(1, 0.5),
             title='层级',
             title_fontsize=12,
             fontsize=10)
    
    ax.axis('off')
    
    # 优化保存过程
    plt.savefig(output_path,
                dpi=200,  # 稍微降低DPI以加快保存
                bbox_inches='tight',
                facecolor='white',
                format='png')
    plt.close()
    
    logger.info(f"层级结构图已保存至: {output_path}")

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
        
        # 生成可视化图形
        visualize_memory_hierarchy(memory_contents, hierarchies)
        
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
