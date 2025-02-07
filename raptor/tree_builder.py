import copy
import logging
import os
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Dict, List, Optional, Set, Tuple

import openai
import tiktoken
from tenacity import retry, stop_after_attempt, wait_random_exponential

from .EmbeddingModels import BaseEmbeddingModel, OpenAIEmbeddingModel
from .SummarizationModels import (BaseSummarizationModel,
                                  GPT3TurboSummarizationModel)
from .tree_structures import Node, Tree
from .utils import (distances_from_embeddings, get_children, get_embeddings,
                    get_node_list, get_text,
                    indices_of_nearest_neighbors_from_distances, split_text)

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class TreeBuilderConfig:
    """
    TreeBuilder的配置类，用于设置和验证树构建器的各项参数
    """
    def __init__(
        self,
        tokenizer=None,
        max_tokens=None,
        num_layers=None,
        threshold=None,
        top_k=None,
        selection_mode=None,
        summarization_length=None,
        summarization_model=None,
        embedding_models=None,
        cluster_embedding_model=None,
    ):
        # 初始化分词器，默认使用cl100k_base
        if tokenizer is None:
            tokenizer = tiktoken.get_encoding("cl100k_base")
        self.tokenizer = tokenizer

        # 设置每个节点的最大token数，默认为100
        if max_tokens is None:
            max_tokens = 100
        if not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("max_tokens必须是大于等于1的整数")
        self.max_tokens = max_tokens

        # 设置树的层数，默认为5层
        if num_layers is None:
            num_layers = 5
        if not isinstance(num_layers, int) or num_layers < 1:
            raise ValueError("num_layers必须是大于等于1的整数")
        self.num_layers = num_layers

        # 设置相似度阈值，默认为0.5
        if threshold is None:
            threshold = 0.5
        if not isinstance(threshold, (int, float)) or not (0 <= threshold <= 1):
            raise ValueError("threshold必须是0到1之间的数")
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
        if selection_mode not in ["top_k", "threshold"]:
            raise ValueError("selection_mode必须是'top_k'或'threshold'")
        self.selection_mode = selection_mode

        # 设置摘要长度，默认为100
        if summarization_length is None:
            summarization_length = 100
        self.summarization_length = summarization_length

        # 设置摘要模型，默认使用GPT3Turbo
        if summarization_model is None:
            summarization_model = GPT3TurboSummarizationModel()
        if not isinstance(summarization_model, BaseSummarizationModel):
            raise ValueError("summarization_model必须是BaseSummarizationModel的实例")
        self.summarization_model = summarization_model

        # 设置嵌入模型，默认使用OpenAI
        if embedding_models is None:
            embedding_models = {"OpenAI": OpenAIEmbeddingModel()}
        if not isinstance(embedding_models, dict):
            raise ValueError("embedding_models必须是一个模型名称到实例的字典")
        for model in embedding_models.values():
            if not isinstance(model, BaseEmbeddingModel):
                raise ValueError("所有嵌入模型必须是BaseEmbeddingModel的实例")
        self.embedding_models = embedding_models

        # 设置聚类嵌入模型
        if cluster_embedding_model is None:
            cluster_embedding_model = "OpenAI"
        if cluster_embedding_model not in self.embedding_models:
            raise ValueError("cluster_embedding_model必须是embedding_models字典中的一个键")
        self.cluster_embedding_model = cluster_embedding_model

    def log_config(self):
        config_log = """
        TreeBuilderConfig:
            Tokenizer: {tokenizer}
            Max Tokens: {max_tokens}
            Num Layers: {num_layers}
            Threshold: {threshold}
            Top K: {top_k}
            Selection Mode: {selection_mode}
            Summarization Length: {summarization_length}
            Summarization Model: {summarization_model}
            Embedding Models: {embedding_models}
            Cluster Embedding Model: {cluster_embedding_model}
        """.format(
            tokenizer=self.tokenizer,
            max_tokens=self.max_tokens,
            num_layers=self.num_layers,
            threshold=self.threshold,
            top_k=self.top_k,
            selection_mode=self.selection_mode,
            summarization_length=self.summarization_length,
            summarization_model=self.summarization_model,
            embedding_models=self.embedding_models,
            cluster_embedding_model=self.cluster_embedding_model,
        )
        return config_log


class TreeBuilder:
    """
    TreeBuilder类负责使用摘要模型和嵌入模型构建分层文本抽象结构（树）。
    """

    def __init__(self, config) -> None:
        """
        初始化TreeBuilder，设置分词器、最大token数、层数、top-k值、阈值和选择模式。
        """
        self.tokenizer = config.tokenizer
        self.max_tokens = config.max_tokens
        self.num_layers = config.num_layers
        self.top_k = config.top_k
        self.threshold = config.threshold
        self.selection_mode = config.selection_mode
        self.summarization_length = config.summarization_length
        self.summarization_model = config.summarization_model
        self.embedding_models = config.embedding_models
        self.cluster_embedding_model = config.cluster_embedding_model

        logging.info(
            f"成功初始化TreeBuilder，配置为：{config.log_config()}"
        )

    def create_node(
        self, index: int, text: str, children_indices: Optional[Set[int]] = None
    ) -> Tuple[int, Node]:
        """
        创建一个新节点

        参数:
            index (int): 新节点的索引
            text (str): 与新节点关联的文本
            children_indices (Optional[Set[int]]): 子节点的索引集合，如果不提供则使用空集合

        返回:
            Tuple[int, Node]: 包含索引和新创建节点的元组
        """
        if children_indices is None:
            children_indices = set()

        embeddings = {
            model_name: model.create_embedding(text)
            for model_name, model in self.embedding_models.items()
        }
        return (index, Node(text, index, children_indices, embeddings))

    def create_embedding(self, text) -> List[float]:
        """
        使用指定的嵌入模型为给定文本生成嵌入向量

        参数:
            text (str): 需要生成嵌入向量的文本

        返回:
            List[float]: 生成的嵌入向量
        """
        return self.embedding_models[self.cluster_embedding_model].create_embedding(text)

    def summarize(self, context, max_tokens=150) -> str:
        """
        使用指定的摘要模型生成输入上下文的摘要

        参数:
            context (str): 需要摘要的上下文
            max_tokens (int, optional): 生成摘要的最大token数，默认为150

        返回:
            str: 生成的摘要
        """
        return self.summarization_model.summarize(context, max_tokens)

    def get_relevant_nodes(self, current_node, list_nodes) -> List[Node]:
        """
        从节点列表中检索与当前节点最相关的top-k个节点，
        基于嵌入空间中的余弦距离

        参数:
            current_node (Node): 当前节点
            list_nodes (List[Node]): 节点列表

        返回:
            List[Node]: top-k个最相关的节点
        """
        embeddings = get_embeddings(list_nodes, self.cluster_embedding_model)
        distances = distances_from_embeddings(
            current_node.embeddings[self.cluster_embedding_model], embeddings
        )
        indices = indices_of_nearest_neighbors_from_distances(distances)

        if self.selection_mode == "threshold":
            best_indices = [
                index for index in indices if distances[index] > self.threshold
            ]

        elif self.selection_mode == "top_k":
            best_indices = indices[: self.top_k]

        nodes_to_add = [list_nodes[idx] for idx in best_indices]

        return nodes_to_add

    def multithreaded_create_leaf_nodes(self, chunks: List[str]) -> Dict[int, Node]:
        """
        使用多线程从给定的文本块列表创建叶子节点

        参数:
            chunks (List[str]): 需要转换为叶子节点的文本块列表

        返回:
            Dict[int, Node]: 节点索引到对应叶子节点的映射字典
        """
        with ThreadPoolExecutor() as executor:
            future_nodes = {
                executor.submit(self.create_node, index, text): (index, text)
                for index, text in enumerate(chunks)
            }

            leaf_nodes = {}
            for future in as_completed(future_nodes):
                index, node = future.result()
                leaf_nodes[index] = node

        return leaf_nodes

    def build_from_text(self, text: str, use_multithreading: bool = True) -> Tree:
        """
        从输入文本构建RA树，可选择是否使用多线程

        参数:
            text (str): 输入文本
            use_multithreading (bool, optional): 创建叶子节点时是否使用多线程，默认为True

        返回:
            Tree: 构建的树结构
        """
        chunks = split_text(text, self.tokenizer, self.max_tokens)

        logging.info("正在创建叶子节点")

        if use_multithreading:
            leaf_nodes = self.multithreaded_create_leaf_nodes(chunks)
        else:
            leaf_nodes = {}
            for index, text in enumerate(chunks):
                __, node = self.create_node(index, text)
                leaf_nodes[index] = node

        layer_to_nodes = {0: list(leaf_nodes.values())}

        logging.info(f"已创建 {len(leaf_nodes)} 个叶子节点嵌入")

        logging.info("正在构建所有节点")

        all_nodes = copy.deepcopy(leaf_nodes)

        root_nodes = self.construct_tree(all_nodes, all_nodes, layer_to_nodes)

        logging.info("construct_tree根节点完成")

        tree = Tree(all_nodes, root_nodes, leaf_nodes, self.num_layers, layer_to_nodes)

        logging.info("RA树构建完成")

        return tree

    def build_from_leafnodes(self,leaf_nodes):
        '''
        从叶子节点字典构建RA树

        参数:
            leaf_nodes (Dict[int, Node]): 叶子节点的字典

        返回:
            Tree: 构建的树结构
        '''
        layer_to_nodes = {0: list(leaf_nodes.values())}

        logging.info("正在构建所有节点")
        
        all_nodes = copy.deepcopy(leaf_nodes)

        root_nodes = self.construct_tree(all_nodes, all_nodes, layer_to_nodes)

        tree = Tree(all_nodes, root_nodes, leaf_nodes, self.num_layers, layer_to_nodes)

        return tree

    @classmethod
    @abstractmethod
    def construct_tree(
        self,
        current_level_nodes: Dict[int, Node],
        all_tree_nodes: Dict[int, Node],
        layer_to_nodes: Dict[int, List[Node]],
        use_multithreading: bool = True,
    ) -> Dict[int, Node]:
        """
        通过逐层迭代摘要相关节点组并更新current_level_nodes和all_tree_nodes字典，
        构建分层树结构

        参数:
            current_level_nodes (Dict[int, Node]): 当前节点集
            all_tree_nodes (Dict[int, Node]): 所有节点的字典
            layer_to_nodes (Dict[int, List[Node]]): 层级到节点列表的映射
            use_multithreading (bool): 是否使用多线程加速处理

        返回:
            Dict[int, Node]: 最终的根节点集
        """
        pass

        # logging.info("使用Transformer-like TreeBuilder")

        # def process_node(idx, current_level_nodes, new_level_nodes, all_tree_nodes, next_node_index, lock):
        #     relevant_nodes_chunk = self.get_relevant_nodes(
        #         current_level_nodes[idx], current_level_nodes
        #     )

        #     node_texts = get_text(relevant_nodes_chunk)

        #     summarized_text = self.summarize(
        #         context=node_texts,
        #         max_tokens=self.summarization_length,
        #     )

        #     logging.info(
        #         f"Node Texts Length: {len(self.tokenizer.encode(node_texts))}, Summarized Text Length: {len(self.tokenizer.encode(summarized_text))}"
        #     )

        #     next_node_index, new_parent_node = self.create_node(
        #         next_node_index,
        #         summarized_text,
        #         {node.index for node in relevant_nodes_chunk}
        #     )

        #     with lock:
        #         new_level_nodes[next_node_index] = new_parent_node

        # for layer in range(self.num_layers):
        #     logging.info(f"Constructing Layer {layer}: ")

        #     node_list_current_layer = get_node_list(current_level_nodes)
        #     next_node_index = len(all_tree_nodes)

        #     new_level_nodes = {}
        #     lock = Lock()

        #     if use_multithreading:
        #         with ThreadPoolExecutor() as executor:
        #             for idx in range(0, len(node_list_current_layer)):
        #                 executor.submit(process_node, idx, node_list_current_layer, new_level_nodes, all_tree_nodes, next_node_index, lock)
        #                 next_node_index += 1
        #             executor.shutdown(wait=True)
        #     else:
        #         for idx in range(0, len(node_list_current_layer)):
        #             process_node(idx, node_list_current_layer, new_level_nodes, all_tree_nodes, next_node_index, lock)

        #     layer_to_nodes[layer + 1] = list(new_level_nodes.values())
        #     current_level_nodes = new_level_nodes
        #     all_tree_nodes.update(new_level_nodes)

        # return new_level_nodes
