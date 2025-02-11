import logging
import pickle

from .cluster_tree_builder import ClusterTreeBuilder, ClusterTreeConfig
from .EmbeddingModels import BaseEmbeddingModel
from .QAModels import BaseQAModel, GPT3TurboQAModel
from .SummarizationModels import BaseSummarizationModel
from .tree_builder import TreeBuilder, TreeBuilderConfig
from .tree_retriever import TreeRetriever, TreeRetrieverConfig
from .tree_structures import Node, Tree

# Define a dictionary to map supported tree builders to their respective configs
supported_tree_builders = {"cluster": (ClusterTreeBuilder, ClusterTreeConfig)}

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class RetrievalAugmentationConfig:
    def __init__(
        self,
        tree_builder_config=None,
        tree_retriever_config=None,  # Change from default instantiation
        qa_model=None,
        embedding_model=None,
        summarization_model=None,
        tree_builder_type="cluster",
        # New parameters for TreeRetrieverConfig and TreeBuilderConfig
        # TreeRetrieverConfig arguments
        tr_tokenizer=None,
        tr_threshold=0.5,
        tr_top_k=10,
        tr_selection_mode="threshold",
        tr_context_embedding_model="OpenAI",
        tr_embedding_model=None,
        tr_num_layers=None,
        tr_start_layer=None,
        # TreeBuilderConfig arguments
        tb_tokenizer=None,
        tb_max_tokens=100,
        tb_num_layers=5,
        tb_threshold=0.5,
        tb_top_k=5,
        tb_selection_mode="top_k",
        tb_summarization_length=100,
        tb_summarization_model=None,
        tb_embedding_models=None,
        tb_cluster_embedding_model="OpenAI",
    ):
        # Validate tree_builder_type
        if tree_builder_type not in supported_tree_builders:
            raise ValueError(
                f"tree_builder_type must be one of {list(supported_tree_builders.keys())}"
            )

        # Validate qa_model
        if qa_model is not None and not isinstance(qa_model, BaseQAModel):
            raise ValueError("qa_model must be an instance of BaseQAModel")

        if embedding_model is not None and not isinstance(
            embedding_model, BaseEmbeddingModel
        ):
            raise ValueError(
                "embedding_model must be an instance of BaseEmbeddingModel"
            )
        elif embedding_model is not None:
            if tb_embedding_models is not None:
                raise ValueError(
                    "Only one of 'tb_embedding_models' or 'embedding_model' should be provided, not both."
                )
            tb_embedding_models = {"EMB": embedding_model}
            tr_embedding_model = embedding_model
            tb_cluster_embedding_model = "EMB"
            tr_context_embedding_model = "EMB"

        if summarization_model is not None and not isinstance(
            summarization_model, BaseSummarizationModel
        ):
            raise ValueError(
                "summarization_model must be an instance of BaseSummarizationModel"
            )

        elif summarization_model is not None:
            if tb_summarization_model is not None:
                raise ValueError(
                    "Only one of 'tb_summarization_model' or 'summarization_model' should be provided, not both."
                )
            tb_summarization_model = summarization_model

        # Set TreeBuilderConfig
        tree_builder_class, tree_builder_config_class = supported_tree_builders[
            tree_builder_type
        ]
        if tree_builder_config is None:
            tree_builder_config = tree_builder_config_class(
                tokenizer=tb_tokenizer,
                max_tokens=tb_max_tokens,
                num_layers=tb_num_layers,
                threshold=tb_threshold,
                top_k=tb_top_k,
                selection_mode=tb_selection_mode,
                summarization_length=tb_summarization_length,
                summarization_model=tb_summarization_model,
                embedding_models=tb_embedding_models,
                cluster_embedding_model=tb_cluster_embedding_model,
            )

        elif not isinstance(tree_builder_config, tree_builder_config_class):
            raise ValueError(
                f"tree_builder_config must be a direct instance of {tree_builder_config_class} for tree_builder_type '{tree_builder_type}'"
            )

        # Set TreeRetrieverConfig
        if tree_retriever_config is None:
            tree_retriever_config = TreeRetrieverConfig(
                tokenizer=tr_tokenizer,
                threshold=tr_threshold,
                top_k=tr_top_k,
                selection_mode=tr_selection_mode,
                context_embedding_model=tr_context_embedding_model,
                embedding_model=tr_embedding_model,
                num_layers=tr_num_layers,
                start_layer=tr_start_layer
            )
        elif not isinstance(tree_retriever_config, TreeRetrieverConfig):
            raise ValueError(
                "tree_retriever_config must be an instance of TreeRetrieverConfig"
            )

        # Assign the created configurations to the instance
        self.tree_builder_config = tree_builder_config
        self.tree_retriever_config = tree_retriever_config
        self.qa_model = qa_model or GPT3TurboQAModel()
        self.tree_builder_type = tree_builder_type

    def log_config(self):
        config_summary = """
        RetrievalAugmentationConfig:
            {tree_builder_config}
            
            {tree_retriever_config}
            
            QA Model: {qa_model}
            Tree Builder Type: {tree_builder_type}
        """.format(
            tree_builder_config=self.tree_builder_config.log_config(),
            tree_retriever_config=self.tree_retriever_config.log_config(),
            qa_model=self.qa_model,
            tree_builder_type=self.tree_builder_type,
        )
        return config_summary


class RetrievalAugmentation:
    """
    一个检索增强类，它结合了 TreeBuilder 和 TreeRetriever 类。
    支持向树中添加文档、检索信息和回答问题。
    """
    def __init__(self, config=None, tree=None):
        """
        使用指定的配置初始化 RetrievalAugmentation 实例。
        参数:
            config (RetrievalAugmentationConfig): RetrievalAugmentation 实例的配置。
            tree: 树实例或序列化树文件的路径。
        """
        if config is None:
            config = RetrievalAugmentationConfig(tr_threshold=0.1)
        if not isinstance(config, RetrievalAugmentationConfig):
            raise ValueError(
                "config must be an instance of RetrievalAugmentationConfig"
            )

        # Check if tree is a string (indicating a path to a pickled tree)
        if isinstance(tree, str):
            try:
                with open(tree, "rb") as file:
                    self.tree = pickle.load(file)
                if not isinstance(self.tree, Tree):
                    raise ValueError("The loaded object is not an instance of Tree")
            except Exception as e:
                raise ValueError(f"Failed to load tree from {tree}: {e}")
        elif isinstance(tree, Tree) or tree is None:
            self.tree = tree
        else:
            raise ValueError(
                "tree must be an instance of Tree, a path to a pickled Tree, or None"
            )

        tree_builder_class = supported_tree_builders[config.tree_builder_type][0]
        self.tree_builder = tree_builder_class(config.tree_builder_config)

        self.tree_retriever_config = config.tree_retriever_config
        self.qa_model = config.qa_model

        if self.tree is not None:
            self.retriever = TreeRetriever(self.tree_retriever_config, self.tree)
        else:
            self.retriever = None

        logging.info(
            f"Successfully initialized RetrievalAugmentation with Config {config.log_config()}"
        )

    def add_to_existing(self, docs):
        """
        向现有的知识树中添加新文档
        该方法允许在不重建整个树的情况下添加新的文档内容

        参数:
            docs (str): 要添加到树中的输入文本
            
        返回:
            bool: 添加成功返回True，失败返回False
            
        异常:
            ValueError: 如果树未初始化
        """
        try:
            if self.tree is None:
                raise ValueError("树未初始化，请先调用add_documents方法创建树")
            
            logging.info("开始向现有树中添加新文档")
            
            # 使用tree_builder处理新文档
            new_tree = self.tree_builder.build_from_text(text=docs)
            if new_tree is None:
                logging.error("处理新文档失败")
                return False
                
            logging.info("新文档处理成功，开始合并到现有树中")
            
            # 获取新树的所有节点
            for node_idx in new_tree.leaf_nodes:
                node = new_tree.all_nodes[node_idx]
                # 将新节点添加到现有树中
                self.tree.add_node(node)
                logging.info(f"已添加新叶子节点: {node.index}")
            self.tree = self.tree_builder.build_from_leafnodes(self.tree.leaf_nodes)
            # 更新检索器
            self.retriever = TreeRetriever(self.tree_retriever_config, self.tree)
            logging.info("检索器已更新")
            
            return True
            
        except Exception as e:
            logging.error(f"向现有树添加文档失败: {str(e)}")
            return False


    def add_documents(self, docs):
        """
        向树中添加文档并创建一个 TreeRetriever 实例。
        如果树已存在，会提示是否使用add_to_existing方法添加到现有树中。

        参数:
            docs (str): 要添加到树中的输入文本。
        """
        if self.tree is not None:
            print("检查到目前树非空，改用'add_to_existing'方法添加到现有树中: ")
            self.add_to_existing(docs)
            return

        logging.info("开始创建新的知识树")
        self.tree = self.tree_builder.build_from_text(text=docs)
        self.retriever = TreeRetriever(self.tree_retriever_config, self.tree)
        logging.info("新的知识树创建完成")

    def retrieve(
        self,
        question,
        start_layer: int = None,
        num_layers: int = None,
        top_k: int = 10,
        max_tokens: int = 3500,
        collapse_tree: bool = True,
        return_layer_information: bool = True,
    ):
        """
        使用 TreeRetriever 实例检索信息并回答问题。

        参数:
            question (str): 要回答的问题。
            start_layer (int): 开始检索的层。默认为 self.start_layer。
            num_layers (int): 要遍历的层数。默认为 self.num_layers。
            max_tokens (int): 最大标记数。默认为 3500。
            collapse_tree (bool): 是否折叠树。默认为 True。
            return_layer_information (bool): 是否返回层信息。默认为 True。

        返回:
            str: 可以找到答案的上下文。

        异常:
            ValueError: 如果 TreeRetriever 实例未初始化。
        """
        if self.retriever is None:
            raise ValueError(
                "The TreeRetriever instance has not been initialized. Call 'add_documents' first."
            )

        return self.retriever.retrieve(
            question,
            start_layer,
            num_layers,
            top_k,
            max_tokens,
            collapse_tree,
            return_layer_information,
        )

    def answer_question(
        self,
        question,
        top_k: int = 10,
        start_layer: int = None,
        num_layers: int = None,
        max_tokens: int = 3500,
        collapse_tree: bool = True,
        return_layer_information: bool = False,
    ):
        """
        使用 TreeRetriever 实例检索信息并回答问题。

        参数:
            question (str): 要回答的问题。
            start_layer (int): 开始检索的层。默认为 self.start_layer。
            num_layers (int): 要遍历的层数。默认为 self.num_layers。
            max_tokens (int): 最大标记数。默认为 3500。
            collapse_tree (bool): 是否折叠树。默认为 True。
            return_layer_information (bool): 是否返回层信息。默认为 False。

        返回:
            str: 问题的答案。

        异常:
            ValueError: 如果 TreeRetriever 实例未初始化。
        """
        # if return_layer_information:
        context, layer_information = self.retrieve(
            question, start_layer, num_layers, top_k, max_tokens, collapse_tree, True
        )

        answer = self.qa_model.answer_question(context, question)

        if return_layer_information:
            return answer, layer_information

        return answer

    def save(self, path):
        if self.tree is None:
            raise ValueError("There is no tree to save.")
        with open(path, "wb") as file:
            pickle.dump(self.tree, file)
        logging.info(f"Tree successfully saved to {path}")

    def get_all_nodes_info(self):
        """
        获取树中所有节点的信息，包括文档块和摘要节点。

        返回:
            dict: 包含以下信息的字典：
                - leaf_nodes: 所有叶子节点（文档块）的列表，每个节点包含:
                    - text: 文档块内容
                    - index: 节点索引
                - summary_nodes: 按层级组织的摘要节点字典，每层包含:
                    - text: 摘要内容
                    - index: 节点索引
                    - children: 子节点索引集合
                - num_layers: 树的层数
                - total_nodes: 节点总数

        异常:
            ValueError: 如果树未初始化。
        """
        if self.tree is None:
            raise ValueError("树未初始化。请先调用add_documents方法添加文档。")

        result = {
            "leaf_nodes": [],
            "summary_nodes": {},
            "num_layers": self.tree.num_layers,
            "total_nodes": len(self.tree.all_nodes)
        }

        # 添加叶子节点信息
        for node_idx in self.tree.leaf_nodes:
            node = self.tree.all_nodes[node_idx]
            result["leaf_nodes"].append({
                "text": node.text,
                "index": node.index
            })
        result["summary_nodes"] = []
        for node_idx in self.tree.root_nodes:
            node = self.tree.root_nodes[node_idx]
            result["summary_nodes"].append({
                "text": node.text,
                "index": node.index,
                "children": list(node.children)  # 转换为列表以便序列化
            })
        return result
