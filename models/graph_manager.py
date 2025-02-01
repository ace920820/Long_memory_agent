#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
记忆图谱管理模块
负责构建和管理完整的记忆图谱，包括层级关系和记忆关联
"""

import logging
import yaml
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from .memory_relationship_manager import MemoryRelationshipManager
from .hierarchy_manager import HierarchyManager
import datetime

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GraphManager:
    """
    记忆图谱管理器
    结合层级信息和记忆关联信息构建完整的记忆图谱
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        初始化图谱管理器
        Args:
            config_path: 配置文件路径
        """
        try:
            # 加载配置文件
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 获取图谱配置
            graph_config = config.get('graph', {})
            self.merge_threshold = graph_config.get('merge_threshold', 0.8)
            
            # 初始化关系管理器和层级管理器
            self.relationship_manager = MemoryRelationshipManager(config_path)
            self.hierarchy_manager = HierarchyManager(config_path)
            
            # 初始化图结构
            self.graph = nx.Graph()
            
            logger.info("图谱管理器初始化成功")
            
        except Exception as e:
            logger.error(f"初始化图谱管理器失败: {str(e)}")
            raise

    def add_memory(self, memory: str, metadata: Optional[Dict] = None) -> bool:
        """
        向图谱中添加新的记忆节点
        Args:
            memory: 记忆内容
            metadata: 记忆相关的元数据
        Returns:
            是否添加成功
        """
        try:
            # 验证输入
            if not memory or not isinstance(memory, str):
                logger.warning("记忆内容无效")
                return False
                
            logger.info(f"添加新记忆节点: {memory[:50]}...")
            
            # 添加节点
            node_id = len(self.graph.nodes)
            self.graph.add_node(node_id, content=memory, metadata=metadata or {})
            
            # 更新与现有节点的关系
            self._update_relationships(node_id)
            
            # 更新层级结构
            self._update_hierarchy(node_id)
            
            logger.info(f"记忆节点添加成功，ID: {node_id}")
            return True
            
        except Exception as e:
            logger.error(f"添加记忆节点失败: {str(e)}")
            return False

    def remove_memory(self, memory_id: int) -> bool:
        """
        从图谱中移除指定的记忆节点
        Args:
            memory_id: 记忆节点ID
        Returns:
            是否移除成功
        """
        try:
            if memory_id not in self.graph:
                logger.warning(f"记忆节点不存在: {memory_id}")
                return False
            
            logger.info(f"移除记忆节点: {memory_id}")
            
            # 移除节点及其关联的边
            self.graph.remove_node(memory_id)
            
            # 更新层级结构
            self._update_hierarchy_after_removal(memory_id)
            
            logger.info(f"记忆节点移除成功")
            return True
            
        except Exception as e:
            logger.error(f"移除记忆节点失败: {str(e)}")
            return False

    def merge_memories(self, memory_id1: int, memory_id2: int) -> Optional[int]:
        """
        合并两个记忆节点
        Args:
            memory_id1: 第一个记忆节点ID
            memory_id2: 第二个记忆节点ID
        Returns:
            合并后的节点ID，失败则返回None
        """
        try:
            if memory_id1 not in self.graph or memory_id2 not in self.graph:
                logger.warning("待合并的记忆节点不存在")
                return None
            
            # 获取节点内容
            content1 = self.graph.nodes[memory_id1]['content']
            content2 = self.graph.nodes[memory_id2]['content']
            
            logger.info(f"尝试合并记忆节点:")
            logger.info(f"节点1 ({memory_id1}): {content1[:50]}...")
            logger.info(f"节点2 ({memory_id2}): {content2[:50]}...")
            
            # 计算相似度
            similarity = self.relationship_manager.calculate_relationships(content1, content2)
            logger.info(f"记忆节点相似度: {similarity:.4f}")
            
            if similarity < self.merge_threshold:
                logger.warning(f"记忆节点相似度 ({similarity:.4f}) 低于合并阈值 ({self.merge_threshold})")
                return None
            
            logger.info(f"记忆节点相似度超过阈值，开始合并")
            
            # 合并元数据
            merged_metadata = {
                **self.graph.nodes[memory_id1].get('metadata', {}),
                **self.graph.nodes[memory_id2].get('metadata', {})
            }
            merged_metadata['original_ids'] = [memory_id1, memory_id2]
            merged_metadata['merge_similarity'] = similarity
            merged_metadata['merge_time'] = str(datetime.datetime.now())
            
            # 创建新节点
            new_id = len(self.graph.nodes)
            merged_content = (
                f"{content1}\n"
                f"---\n"  # 使用分隔符分隔两个记忆
                f"{content2}"
            )
            
            self.graph.add_node(new_id, 
                              content=merged_content,
                              metadata=merged_metadata)
            
            # 继承关系
            for neighbor in self.graph.neighbors(memory_id1):
                if neighbor != memory_id2:
                    weight = self.graph[memory_id1][neighbor]['weight']
                    self.graph.add_edge(new_id, neighbor, weight=weight)
                    logger.debug(f"继承节点1关系: {new_id} -> {neighbor}, 权重: {weight:.4f}")
            
            for neighbor in self.graph.neighbors(memory_id2):
                if neighbor != memory_id1:
                    weight = self.graph[memory_id2][neighbor]['weight']
                    self.graph.add_edge(new_id, neighbor, weight=weight)
                    logger.debug(f"继承节点2关系: {new_id} -> {neighbor}, 权重: {weight:.4f}")
            
            # 移除原始节点
            self.graph.remove_node(memory_id1)
            self.graph.remove_node(memory_id2)
            
            # 更新层级结构
            self._update_hierarchy(new_id)
            
            logger.info(f"记忆节点合并成功，新节点ID: {new_id}")
            return new_id
            
        except Exception as e:
            logger.error(f"合并记忆节点失败: {str(e)}")
            return None

    def visualize(self, output_path: str = "memory_graph.png") -> bool:
        """
        可视化记忆图谱
        Args:
            output_path: 输出文件路径
        Returns:
            是否成功生成可视化图像
        """
        try:
            logger.info("开始生成记忆图谱可视化")
            
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
            plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
            
            if not self.graph.nodes:
                logger.warning("图谱为空，生成空白图像")
                plt.figure(figsize=(12, 8))
                plt.text(0.5, 0.5, "空图谱", 
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontproperties='SimHei')  # 使用中文字体
                plt.savefig(output_path, bbox_inches='tight')
                plt.close()
                return True
            
            # 设置绘图参数
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(self.graph)
            
            # 绘制节点
            nx.draw_networkx_nodes(self.graph, pos, 
                                 node_color='lightblue',
                                 node_size=1000)
            
            # 绘制边
            edges = self.graph.edges(data=True)
            if edges:
                weights = [d['weight'] for (u, v, d) in edges]
                nx.draw_networkx_edges(self.graph, pos, 
                                     width=[w * 2 for w in weights],
                                     alpha=0.5)
            
            # 添加标签，使用英文字符替代中文
            labels = {node: f"Memory_{node}" for node in self.graph.nodes()}
            nx.draw_networkx_labels(self.graph, pos, labels)
            
            # 保存图像
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()
            
            logger.info(f"记忆图谱可视化已保存至: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"生成记忆图谱可视化失败: {str(e)}")
            return False

    def _update_relationships(self, node_id: int) -> None:
        """
        更新指定节点与其他节点的关系
        Args:
            node_id: 节点ID
        """
        try:
            content = self.graph.nodes[node_id]['content']
            
            # 计算与所有现有节点的关系
            for other_id in self.graph.nodes:
                if other_id != node_id:
                    other_content = self.graph.nodes[other_id]['content']
                    weight = self.relationship_manager.calculate_relationships(
                        content, other_content)
                    
                    # 只添加超过阈值的关系
                    if weight > self.relationship_manager.similarity_threshold:
                        self.graph.add_edge(node_id, other_id, weight=weight)
                        logger.debug(f"添加关系: {node_id} -> {other_id}, 权重: {weight:.4f}")
            
        except Exception as e:
            logger.error(f"更新节点关系失败: {str(e)}")
            raise

    def _update_hierarchy(self, node_id: int) -> None:
        """
        更新层级结构
        Args:
            node_id: 节点ID
        """
        try:
            content = self.graph.nodes[node_id]['content']
            
            # 获取所有记忆内容
            all_contents = [data['content'] 
                          for _, data in self.graph.nodes(data=True)]
            
            # 更新层级结构
            hierarchy = self.hierarchy_manager.update_hierarchy(content)
            
            # 将层级信息添加到节点元数据
            self.graph.nodes[node_id]['metadata']['hierarchy'] = hierarchy
            
        except Exception as e:
            logger.error(f"更新层级结构失败: {str(e)}")
            raise

    def _update_hierarchy_after_removal(self, removed_id: int) -> None:
        """
        节点移除后更新层级结构
        Args:
            removed_id: 被移除的节点ID
        """
        try:
            # 获取剩余的所有记忆内容
            remaining_contents = [data['content'] 
                                for _, data in self.graph.nodes(data=True)]
            
            # 重新构建层级结构
            clusters = {}  # 这里需要根据实际情况构建聚类信息
            hierarchy = self.hierarchy_manager.build_hierarchy(clusters)
            
            # 更新所有节点的层级信息
            for node_id in self.graph.nodes:
                self.graph.nodes[node_id]['metadata']['hierarchy'] = hierarchy
            
        except Exception as e:
            logger.error(f"移除节点后更新层级结构失败: {str(e)}")
            raise
