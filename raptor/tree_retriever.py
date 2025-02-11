import logging
import os
from typing import Dict, List, Set

import tiktoken
from tenacity import retry, stop_after_attempt, wait_random_exponential

from .EmbeddingModels import BaseEmbeddingModel, OpenAIEmbeddingModel
from .Retrievers import BaseRetriever
from .tree_structures import Node, Tree
from .utils import (distances_from_embeddings, get_children, get_embeddings,
                    get_node_list, get_text,
                    indices_of_nearest_neighbors_from_distances,
                    reverse_mapping)

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class TreeRetrieverConfig:
    """
    TreeRetriever的配置类，用于设置和验证树检索器的各项参数
    """
    def __init__(
        self,
        tokenizer=None,
        threshold=None,
        top_k=None,
        selection_mode=None,
        context_embedding_model=None,
        embedding_model=None,
        num_layers=None,
        start_layer=None,
    ):
        # 初始化分词器，默认使用cl100k_base
        if tokenizer is None:
            tokenizer = tiktoken.get_encoding("cl100k_base")
        self.tokenizer = tokenizer

        # 设置相似度阈值，默认为0.5
        if threshold is None:
            threshold = 0.7
        if not isinstance(threshold, float) or not (0 <= threshold <= 1):
            raise ValueError("threshold必须是0到1之间的浮点数")
        self.threshold = threshold

        # 设置top_k值，默认为5
        if top_k is None:
            top_k = 5
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k必须是大于等于1的整数")
        self.top_k = top_k

        # 设置选择模式，可以是top_k或threshold
        if selection_mode is None:
            selection_mode = "top_k"
        if not isinstance(selection_mode, str) or selection_mode not in [
            "top_k",
            "threshold",
        ]:
            raise ValueError("selection_mode必须是'top_k'或'threshold'")
        self.selection_mode = selection_mode

        # 设置上下文嵌入模型，默认为OpenAI
        if context_embedding_model is None:
            context_embedding_model = "OpenAI"
        if not isinstance(context_embedding_model, str):
            raise ValueError("context_embedding_model必须是字符串")
        self.context_embedding_model = context_embedding_model

        # 设置嵌入模型，默认使用OpenAI
        if embedding_model is None:
            embedding_model = OpenAIEmbeddingModel()
        if not isinstance(embedding_model, BaseEmbeddingModel):
            raise ValueError("embedding_model必须是BaseEmbeddingModel的实例")
        self.embedding_model = embedding_model

        # 设置层数
        if num_layers is not None:
            if not isinstance(num_layers, int) or num_layers < 0:
                raise ValueError("num_layers必须是大于等于0的整数")
        self.num_layers = num_layers

        # 设置起始层
        if start_layer is not None:
            if not isinstance(start_layer, int) or start_layer < 0:
                raise ValueError("start_layer必须是大于等于0的整数")
        self.start_layer = start_layer

    def log_config(self):
        config_log = """
        TreeRetrieverConfig:
            分词器: {tokenizer}
            阈值: {threshold}
            Top K: {top_k}
            选择模式: {selection_mode}
            上下文嵌入模型: {context_embedding_model}
            嵌入模型: {embedding_model}
            层数: {num_layers}
            起始层: {start_layer}
        """.format(
            tokenizer=self.tokenizer,
            threshold=self.threshold,
            top_k=self.top_k,
            selection_mode=self.selection_mode,
            context_embedding_model=self.context_embedding_model,
            embedding_model=self.embedding_model,
            num_layers=self.num_layers,
            start_layer=self.start_layer,
        )
        return config_log


class TreeRetriever(BaseRetriever):
    """
    树检索器类，负责从树结构中检索相关信息
    """

    def __init__(self, config, tree) -> None:
        """
        初始化树检索器

        参数:
            config: 检索器配置
            tree: 要检索的树结构
        """
        if not isinstance(tree, Tree):
            raise ValueError("tree必须是Tree类的实例")

        if config.num_layers is not None and config.num_layers > tree.num_layers + 1:
            raise ValueError("config中的num_layers必须小于等于tree.num_layers + 1")

        if config.start_layer is not None and config.start_layer > tree.num_layers:
            raise ValueError("config中的start_layer必须小于等于tree.num_layers")

        self.tree = tree
        self.num_layers = (
            config.num_layers if config.num_layers is not None else tree.num_layers + 1
        )
        self.start_layer = (
            config.start_layer if config.start_layer is not None else tree.num_layers
        )

        if self.num_layers > self.start_layer + 1:
            raise ValueError("num_layers必须小于等于start_layer + 1")

        self.tokenizer = config.tokenizer
        self.top_k = config.top_k
        self.threshold = config.threshold
        self.selection_mode = config.selection_mode
        self.embedding_model = config.embedding_model
        self.context_embedding_model = config.context_embedding_model

        self.tree_node_index_to_layer = reverse_mapping(self.tree.layer_to_nodes)

        logging.info(
            f"成功初始化TreeRetriever，配置为：{config.log_config()}"
        )

    def create_embedding(self, text: str) -> List[float]:
        """
        使用指定的嵌入模型为给定文本生成嵌入向量

        参数:
            text (str): 需要生成嵌入向量的文本

        返回:
            List[float]: 生成的嵌入向量
        """
        return self.embedding_model.create_embedding(text)

    def retrieve_information_collapse_tree(self, query: str, top_k: int, max_tokens: int) -> str:
        """
        基于查询从树中检索最相关的信息（扁平化检索方式）

        参数:
            query (str): 查询文本
            top_k (int): 返回的最相关节点数量
            max_tokens (int): 最大token数量

        返回:
            str: 使用最相关节点创建的上下文
        """
        query_embedding = self.create_embedding(query)

        selected_nodes = []

        node_list = get_node_list(self.tree.all_nodes)

        embeddings = get_embeddings(node_list, self.context_embedding_model)

        distances = distances_from_embeddings(query_embedding, embeddings)

        indices = indices_of_nearest_neighbors_from_distances(distances)

        # 先筛选出相似度大于阈值的节点索引
        filtered_indices = [idx for idx in indices if distances[idx] > self.threshold]
        logging.info(f"扁平化检索 - 相似度阈值: {self.threshold}")
        logging.info(f"原始检索节点数: {len(indices)}, 相似度大于阈值的节点数: {len(filtered_indices)}")

        # 从筛选后的结果中取top_k个
        filtered_indices = filtered_indices[:top_k]
        logging.info(f"取出前{top_k}个节点")

        total_tokens = 0
        for idx in filtered_indices:
            node = node_list[idx]
            node_tokens = len(self.tokenizer.encode(node.text))
            # 添加相似度得分日志
            logging.info(f"检索到节点 - 相似度得分: {(1-distances[idx]):.4f}")
            logging.info(f"节点内容: {node.text[:100]}...")  # 只显示前100个字符

            if total_tokens + node_tokens > max_tokens:
                break

            selected_nodes.append(node)
            total_tokens += node_tokens

        context = get_text(selected_nodes)
        return selected_nodes, context

    def retrieve_information(
        self, current_nodes: List[Node], query: str, num_layers: int
    ) -> str:
        """
        基于查询从树中检索最相关的信息（层次检索方式）

        参数:
            current_nodes (List[Node]): 当前节点列表
            query (str): 查询文本   
            num_layers (int): 要遍历的层数

        返回:
            str: 使用最相关节点创建的上下文
        """
        query_embedding = self.create_embedding(query)

        selected_nodes = []

        node_list = current_nodes

        for layer in range(num_layers):
            embeddings = get_embeddings(node_list, self.context_embedding_model)

            distances = distances_from_embeddings(query_embedding, embeddings)

            indices = indices_of_nearest_neighbors_from_distances(distances)

            if self.selection_mode == "threshold":
                best_indices = [
                    index for index in indices if (1-distances[index]) > self.threshold
                ]
                # 添加阈值模式下的相似度得分日志
                logging.info(f"第{layer}层检索 - 阈值模式 (阈值={self.threshold})")
                for idx in best_indices:
                    logging.info(f"检索到节点 - 相似度得分: {(1-distances[idx]):.4f}")
                    logging.info(f"节点内容: {node_list[idx].text[:100]}...")  # 只显示前100个字符

            elif self.selection_mode == "top_k":
                best_indices = indices[: self.top_k]
                # 添加top_k模式下的相似度得分日志
                logging.info(f"第{layer}层检索 - Top K模式 (K={self.top_k})")
                for idx in best_indices:
                    logging.info(f"检索到节点 - 相似度得分: {(1-distances[idx]):.4f}")
                    logging.info(f"节点内容: {node_list[idx].text[:100]}...")  # 只显示前100个字符

            nodes_to_add = [node_list[idx] for idx in best_indices]

            selected_nodes.extend(nodes_to_add)

            if layer != num_layers - 1:
                child_nodes = []

                for index in best_indices:
                    child_nodes.extend(node_list[index].children)

                # 取唯一值
                child_nodes = list(dict.fromkeys(child_nodes))
                node_list = [self.tree.all_nodes[i] for i in child_nodes]

        context = get_text(selected_nodes)
        return selected_nodes, context

    def retrieve(
        self,
        query: str,
        start_layer: int = None,
        num_layers: int = None,
        top_k: int = 10, 
        max_tokens: int = 3500,
        collapse_tree: bool = True,
        return_layer_information: bool = False,
    ) -> str:
        """
        查询树并返回最相关的信息

        参数:
            query (str): 查询文本
            start_layer (int): 起始层，默认为self.start_layer
            num_layers (int): 要遍历的层数，默认为self.num_layers
            max_tokens (int): 最大token数，默认为3500
            collapse_tree (bool): 是否检索所有节点，默认为True
            return_layer_information (bool): 是否返回层级信息，默认为False

        返回:
            str: 查询结果
        """
        if not isinstance(query, str):
            raise ValueError("query必须是字符串")

        if not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("max_tokens必须是大于等于1的整数")

        if not isinstance(collapse_tree, bool):
            raise ValueError("collapse_tree必须是布尔值")

        # 设置默认值
        start_layer = self.start_layer if start_layer is None else start_layer
        num_layers = self.num_layers if num_layers is None else num_layers

        if not isinstance(start_layer, int) or not (
            0 <= start_layer <= self.tree.num_layers
        ):
            raise ValueError("start_layer必须是0到tree.num_layers之间的整数")

        if not isinstance(num_layers, int) or num_layers < 1:
            raise ValueError("num_layers必须是大于等于1的整数")

        if num_layers > (start_layer + 1):
            raise ValueError("num_layers必须小于等于start_layer + 1")

        if collapse_tree:
            logging.info(f"使用扁平化树检索")
            selected_nodes, context = self.retrieve_information_collapse_tree(
                query, top_k, max_tokens
            )
        else:
            layer_nodes = self.tree.layer_to_nodes[start_layer]
            selected_nodes, context = self.retrieve_information(
                layer_nodes, query, num_layers
            )

        if return_layer_information:
            layer_information = []

            for node in selected_nodes:
                layer_information.append(
                    {
                        "node_index": node.index,
                        "layer_number": self.tree_node_index_to_layer[node.index],
                    }
                )

            return context, layer_information

        return context
