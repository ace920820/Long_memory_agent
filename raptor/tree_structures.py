from typing import Dict, List, Set
import logging


class Node:
    """
    表示层次树结构中的节点
    """

    def __init__(self, text: str, index: int, children: Set[int], embeddings) -> None:
        self.text = text
        self.index = index
        self.children = children
        self.embeddings = embeddings
        self.layer = 0  # 默认为叶子节点层


class Tree:
    """
    表示整个层次树结构
    """

    def __init__(
        self, all_nodes, root_nodes, leaf_nodes, num_layers, layer_to_nodes
    ) -> None:
        self.all_nodes = all_nodes
        self.root_nodes = root_nodes
        self.leaf_nodes = leaf_nodes
        self.num_layers = num_layers
        self.layer_to_nodes = layer_to_nodes

    def add_node(self, node: Node) -> bool:
        """
        向树中添加新节点
        新节点将被添加为叶子节点，并更新相关的树结构信息

        参数:
            node (Node): 要添加的新节点

        返回:
            bool: 添加成功返回True，失败返回False
        """
        try:
            # 设置节点层级为0（叶子层）
            node.layer = 0
            
            # 更新节点索引，避免冲突
            new_index = max(self.all_nodes.keys()) + 1 if self.all_nodes else 0
            old_index = node.index
            node.index = new_index
            
            logging.info(f"正在添加新节点，原始索引: {old_index}，新索引: {new_index}")
            
            # 更新节点集合
            self.all_nodes[new_index] = node
            self.leaf_nodes[new_index] = node
            
            # 更新层级信息
            if 0 not in self.layer_to_nodes:
                self.layer_to_nodes[0] = []
            self.layer_to_nodes[0].append(node)
            
            logging.info(f"节点 {new_index} 已添加到树中")
            return True
            
        except Exception as e:
            logging.error(f"添加节点失败: {str(e)}")
            return False
