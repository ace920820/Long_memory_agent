import os
import logging
import json
import time
from typing import Dict, List, Optional, Union

# 导入知识库存储
from knowledge_base.storage import DocumentStorage

# 导入 Raptor 相关组件
from raptor.RetrievalAugmentation import RetrievalAugmentation
from raptor.tree_retriever import TreeRetriever
from utils.raptor_config_manager import RaptorConfigManager
from utils import configure_logging

# 配置日志记录
configure_logging(level=logging.DEBUG)
logger = logging.getLogger(__name__)
# 降低Numba的日志级别
logging.getLogger('numba').setLevel(logging.WARNING)
# 降低umap的日志级别
logging.getLogger('umap').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)

class RaptorModule:
    """
    基于Raptor的检索增强模块
    用于文档的存储、检索和问答
    """
    
    def __init__(self, data_dir: str = "data/RAtree", tree_save_filename = 'document_tree', use_config_manager: bool = True):
        """
        初始化RaptorModule

        Args:
            data_dir (str): Raptor数据存储目录，默认为 "data/RAtree"
            tree_save_filename (str): 树结构保存的文件名，默认为 'document_tree'
            use_config_manager (bool): 是否使用配置管理器，默认为True
        """
        try:
            # 记录配置信息
            logger.info(f"初始化RaptorModule: data_dir={data_dir}, tree_save_filename={tree_save_filename}, use_config_manager={use_config_manager}")
            
            # 尝试使用配置管理器
            ra_config = None
            self.reranker = None
            self.reranker_tokenizer = None
            self.config_manager = None
            
            if use_config_manager and RaptorConfigManager is not None:
                try:
                    # 初始化配置管理器
                    self.config_manager = RaptorConfigManager()
                    # 获取RA配置
                    ra_config = self.config_manager.get_ra_config()
                    # 尝试加载重排序模型
                    logging.info("尝试加载重排序模型")
                    self.reranker = self.config_manager.get_reranker()
                    self.reranker_tokenizer = self.config_manager.get_reranker_tokenizer()
                    if self.reranker is not None:
                        logger.info("成功加载重排序模型")
                    logger.info("成功使用配置管理器初始化RAPTOR配置")
                except Exception as e:
                    logger.warning(f"使用配置管理器失败: {str(e)}，将使用默认配置")
            
            # 初始化DocumentStorage实例
            self.doc_storage = DocumentStorage(
                storage_dir=os.path.join(data_dir, "files"),
                tree_save_path=os.path.join(data_dir, tree_save_filename),
                ra_config=ra_config
            )
            # 获取DocumentStorage中的RA实例
            self.RA = self.doc_storage.RA
            logger.info("成功初始化Raptor检索增强模块")
            
        except Exception as e:
            logger.error(f"初始化RaptorModule时发生错误: {str(e)}")
            raise
    
    def add_documents_no_saving(self, text: str) -> Dict:
        """
        向知识库中添加文档内容，仅更新树结构，不保存元数据

        Args:
            text: 要添加的文本内容

        Returns:
            Dict: 包含操作结果的字典
        """
        try:
            # 使用DocumentStorage添加文档到树结构
            tree_success = self.doc_storage.add_document_in_tree("", text)
            if not tree_success:
                return {"status": "error", "message": "添加文档到树结构失败"}
            
            return {"status": "success", "message": "文档添加成功"}
            
        except Exception as e:
            error_msg = f"添加文档时发生错误: {str(e)}"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}

    def add_file(self, file_path: str, file_name: str = None) -> Dict:
        """添加新文件到知识库，包括更新元数据和树结构
        
        Args:
            file_path: 文件路径
            file_name: 文件名（可选）
            
        Returns:
            Dict: 包含操作结果的字典
        """
        try:
            # 检查文件是否存在
            if not os.path.exists(file_path):
                error_msg = f"文件不存在: {file_path}"
                logger.error(error_msg)
                return {"status": "error", "message": error_msg}
            
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 记录日志
            logger.info(f"成功读取文件: {file_path}, 内容长度: {len(content)}")
            
            # 生成文档ID
            doc_id = str(int(time.time()))
            
            # 添加文档到元数据
            metadata_success = self.doc_storage.add_document_in_metadata(doc_id, content)
            if not metadata_success:
                return {"status": "error", "message": "添加文档元数据失败"}
            
            # 添加文档到树结构
            tree_success = self.doc_storage.add_document_in_tree(doc_id, content)
            if not tree_success:
                return {"status": "error", "message": "添加文档到树结构失败"}
            
            # 保存树结构
            self.doc_storage.save_RA_tree()
            
            return {
                "status": "success", 
                "message": f"文件 {file_name or file_path} 添加成功",
                "file_path": file_path,
                "doc_id": doc_id
            }
                
        except Exception as e:
            error_msg = f"添加文件时发生错误: {str(e)}"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}

    def search(self, query: str, top_k: int = 5, use_rerank: bool = None) -> Union[Dict, str, List]:
        """搜索相关文档，使用两阶段检索策略

        Args:
            query: 查询文本
            top_k: 返回的最大结果数量，默认为5
            use_rerank: 是否使用重排序，None时使用配置文件设置

        Returns:
            Dict or str or List: 检索到的文档和相关信息
        """
        logging.info(f"开始执行查询: '{query}'，top_k={top_k}")
        
        # 查询参数
        start_layer: int = None
        num_layers: int = None
        max_tokens: int = 3500
        collapse_tree: bool = True
        
        # 获取全局重排序配置
        from utils.raptor_config_manager import RaptorConfigManager
        config_manager = RaptorConfigManager()
        
        # 如果没有指定是否重排序，则使用全局配置
        if use_rerank is None:
            rerank_enabled = config_manager.is_rerank_enabled()
            logging.debug(f"使用全局配置的重排序设置: {rerank_enabled}")
        else:
            rerank_enabled = use_rerank
            logging.debug(f"使用传入的重排序设置: {rerank_enabled}")
        
        # 如果启用重排序，则预加载重排序模型
        if rerank_enabled:
            if not hasattr(self, 'reranker') or self.reranker is None:
                logging.info("加载重排序模型和分词器")
                self.reranker = config_manager.get_reranker()
                self.reranker_tokenizer = config_manager.get_reranker_tokenizer()
        
        try:
            # 检查树是否为空
            if self.RA.tree is None:
                logging.warning("检索树为空，返回空列表")
                return []
            
            # 检查检索器是否正常初始化
            if self.RA.retriever is None:
                logging.warning("检索器未初始化，无法执行检索，返回空列表")
                return []
            
            # 执行实际查询
            logging.debug(f"调用RetrievalAugmentation执行检索，top_k={top_k}")
            context, layer_information = self.RA.retrieve(
                query, start_layer, num_layers, top_k, max_tokens, collapse_tree, True
            )
            
            logging.debug(f"检索结果类型: {type(context).__name__}")
            
            # 如果启用重排序且结果不为空，则进行重排序
            if rerank_enabled:
                logging.info("重排序功能已启用，开始处理检索结果")
                
                # 处理字典格式的结果
                if isinstance(context, dict) and 'context' in context and isinstance(context['context'], list) and len(context['context']) > 0:
                    # 获取上下文文本列表
                    node_texts = context['context']
                    logging.debug(f"处理字典格式的检索结果，包含 {len(node_texts)} 个文本段")
                    
                    # 使用自定义的距离计算函数进行重排序
                    try:
                        # 使用TreeRetriever的嵌入模型获取嵌入向量
                        logging.debug("开始获取查询嵌入向量")
                        query_embedding = self.RA.retriever.create_embedding(query)
                        
                        # 提取所有文本块的嵌入向量
                        logging.debug(f"开始获取 {len(node_texts)} 个文本段的嵌入向量")
                        node_embeddings = []
                        for i, text in enumerate(node_texts):
                            try:
                                node_embeddings.append(self.RA.retriever.create_embedding(text))
                                if (i+1) % 10 == 0:
                                    logging.debug(f"已处理 {i+1}/{len(node_texts)} 个文本段的嵌入向量")
                            except Exception as e:
                                logging.error(f"获取文本嵌入向量时发生错误 (文本段 {i+1}): {str(e)}")
                                node_embeddings.append(None)
                        
                        # 使用重排序功能计算距离和重排序分数
                        logging.debug("开始使用distances_from_embeddings函数计算嵌入距离和重排序得分")
                        from raptor.utils import distances_from_embeddings
                        start_time = time.time()
                        _, rerank_scores = distances_from_embeddings(
                            query_embedding=query_embedding,
                            embeddings=node_embeddings,
                            rerank=True,
                            query_text=query,
                            node_texts=node_texts,
                            reranker=self.reranker,
                            reranker_tokenizer=self.reranker_tokenizer
                        )
                        elapsed_time = time.time() - start_time
                        logging.debug(f"distances_from_embeddings函数执行完成，耗时 {elapsed_time:.2f} 秒")
                        
                        # 将文本和重排序分数结合
                        combined = list(zip(node_texts, rerank_scores))
                        # 按重排序分数排序（降序）
                        sorted_results = sorted(combined, key=lambda x: x[1], reverse=True)
                        
                        # 记录排序前后的变化
                        if len(sorted_results) > 1:
                            logging.debug("重排序前后的变化 (仅显示前5项):")
                            for i, (text, score) in enumerate(sorted_results[:min(5, len(sorted_results))]):
                                orig_idx = node_texts.index(text)
                                logging.debug(f"  现排名 {i+1}：原排名 {orig_idx+1}，得分 {score:.4f}")
                        
                        # 更新上下文
                        context['context'] = [item[0] for item in sorted_results]
                        # 记录重排序分数
                        context['rerank_scores'] = [item[1] for item in sorted_results]
                        logging.info(f"重排序完成，重新排序了 {len(sorted_results)} 个结果")
                    except Exception as e:
                        logging.error(f"重排序过程中发生错误: {str(e)}")
                        # 发生错误时保持原始结果不变
                
                # 处理字符串格式的结果（纯文本）
                elif isinstance(context, str) and context.strip():
                    # 分割文本成段落
                    paragraphs = [p.strip() for p in context.split("\n\n") if p.strip()]
                    if paragraphs:
                        logging.debug(f"处理字符串格式的检索结果，将其分割为 {len(paragraphs)} 个段落")
                        
                        # 获取查询嵌入向量
                        try:
                            # 使用TreeRetriever的嵌入模型获取嵌入向量
                            logging.debug("开始获取查询嵌入向量")
                            query_embedding = self.RA.retriever.create_embedding(query)
                            
                            # 提取所有段落的嵌入向量
                            logging.debug(f"开始获取 {len(paragraphs)} 个段落的嵌入向量")
                            paragraph_embeddings = []
                            for i, p in enumerate(paragraphs):
                                try:
                                    paragraph_embeddings.append(self.RA.retriever.create_embedding(p))
                                    if (i+1) % 10 == 0:
                                        logging.debug(f"已处理 {i+1}/{len(paragraphs)} 个段落的嵌入向量")
                                except Exception as e:
                                    logging.error(f"获取段落嵌入向量时发生错误 (段落 {i+1}): {str(e)}")
                                    paragraph_embeddings.append(None)
                            
                            # 使用重排序功能计算距离和重排序分数
                            logging.debug("开始使用distances_from_embeddings函数计算嵌入距离和重排序得分")
                            from raptor.utils import distances_from_embeddings
                            start_time = time.time()
                            _, rerank_scores = distances_from_embeddings(
                                query_embedding=query_embedding,
                                embeddings=paragraph_embeddings,
                                rerank=True,
                                query_text=query,
                                node_texts=paragraphs,
                                reranker=self.reranker,
                                reranker_tokenizer=self.reranker_tokenizer
                            )
                            elapsed_time = time.time() - start_time
                            logging.debug(f"distances_from_embeddings函数执行完成，耗时 {elapsed_time:.2f} 秒")
                            
                            # 将段落和重排序分数结合
                            combined = list(zip(paragraphs, rerank_scores))
                            # 按重排序分数排序（降序）
                            sorted_results = sorted(combined, key=lambda x: x[1], reverse=True)
                            
                            # 记录排序前后的变化
                            if len(sorted_results) > 1:
                                logging.debug("重排序前后的变化 (仅显示前5项):")
                                for i, (text, score) in enumerate(sorted_results[:min(5, len(sorted_results))]):
                                    orig_idx = paragraphs.index(text)
                                    logging.debug(f"  现排名 {i+1}：原排名 {orig_idx+1}，得分 {score:.4f}")
                            
                            # 返回排序后的文本，用双换行符连接
                            context = "\n\n".join([item[0] for item in sorted_results])
                            logging.info(f"重排序完成，重新排序了 {len(sorted_results)} 个段落")
                        except Exception as e:
                            logging.error(f"处理段落时发生错误: {str(e)}")
                            # 如果处理失败，保持原始文本不变
                    else:
                        logging.warning("无法对文本进行分段，跳过重排序")
                
                # 处理列表格式的结果
                elif isinstance(context, list) and len(context) > 0:
                    logging.debug(f"处理列表格式的检索结果，包含 {len(context)} 个项目")
                    if all(isinstance(item, str) for item in context):
                        try:
                            # 获取查询嵌入向量
                            logging.debug("开始获取查询嵌入向量")
                            query_embedding = self.RA.retriever.create_embedding(query)
                            
                            # 提取所有项目的嵌入向量
                            logging.debug(f"开始获取 {len(context)} 个项目的嵌入向量")
                            item_embeddings = []
                            for i, item in enumerate(context):
                                try:
                                    item_embeddings.append(self.RA.retriever.create_embedding(item))
                                    if (i+1) % 10 == 0:
                                        logging.debug(f"已处理 {i+1}/{len(context)} 个项目的嵌入向量")
                                except Exception as e:
                                    logging.error(f"获取项目嵌入向量时发生错误 (项目 {i+1}): {str(e)}")
                                    item_embeddings.append(None)
                            
                            # 使用重排序功能计算距离和重排序分数
                            logging.debug("开始使用distances_from_embeddings函数计算嵌入距离和重排序得分")
                            from raptor.utils import distances_from_embeddings
                            start_time = time.time()
                            _, rerank_scores = distances_from_embeddings(
                                query_embedding=query_embedding,
                                embeddings=item_embeddings,
                                rerank=True,
                                query_text=query,
                                node_texts=context,
                                reranker=self.reranker,
                                reranker_tokenizer=self.reranker_tokenizer
                            )
                            elapsed_time = time.time() - start_time
                            logging.debug(f"distances_from_embeddings函数执行完成，耗时 {elapsed_time:.2f} 秒")
                            
                            # 将项目和重排序分数结合
                            combined = list(zip(context, rerank_scores))
                            # 按重排序分数排序（降序）
                            sorted_results = sorted(combined, key=lambda x: x[1], reverse=True)
                            
                            # 记录排序前后的变化
                            if len(sorted_results) > 1:
                                logging.debug("重排序前后的变化 (仅显示前5项):")
                                for i, (text, score) in enumerate(sorted_results[:min(5, len(sorted_results))]):
                                    orig_idx = context.index(text)
                                    logging.debug(f"  现排名 {i+1}：原排名 {orig_idx+1}，得分 {score:.4f}")
                            
                            # 返回排序后的列表
                            context = [item[0] for item in sorted_results]
                            logging.info(f"重排序完成，重新排序了 {len(sorted_results)} 个项目")
                        except Exception as e:
                            logging.error(f"处理列表时发生错误: {str(e)}")
                            # 如果处理失败，保持原始列表不变
                    else:
                        logging.warning("列表中包含非字符串项目，跳过重排序")
            
            # 返回最终结果
            if isinstance(context, dict) and 'context' in context:
                # 如果是字典格式，转换为字符串以与原有接口兼容
                if isinstance(context['context'], list):
                    result = "\n\n".join(context['context'])
                    logging.info(f"查询完成，返回 {len(context['context'])} 个结果（已转换为字符串格式）")
                    return result
                return context
            else:
                logging.info(f"查询完成，返回结果类型: {type(context).__name__}")
                return context
            
        except Exception as e:
            error_msg = f"执行查询时发生错误: {str(e)}"
            logging.error(error_msg)
            import traceback
            logging.error(traceback.format_exc())
            return []  # 发生错误时返回空列表，与原接口保持一致

    def generate_rag_response(self, query: str, llm_model, context=None, role_prompt=None):
        """生成带有检索增强的响应"""
        try:
            # 2. 构建提示词
            prompt_parts = []

            # 添加角色提示（如果有）
            if role_prompt:
                prompt_parts.append({"role": "system", "content": role_prompt})

            # 构建上下文信息
            context_info = []

            # 添加context中的文本块信息
            if context and 'context' in context and isinstance(context['context'], list):
                context_info.append("\n已知信息：")
                temp_context = context['context']
                context_info.extend(temp_context)

            # 添加记忆信息（去重）
            memories = context['memories']
            if memories:
                seen_contents = set()
                memory_info = []
                for memory in memories:
                    content = memory
                    if content and content not in seen_contents:
                        memory_info.append(f"历史记忆: {content}")
                        seen_contents.add(content)
                if memory_info:
                    context_info.extend(memory_info)

            # 如果有上下文信息，添加到提示词中
            if context_info:
                context_message = "\n".join(context_info)
                prompt_parts.append({
                    "role": "system",
                    "content": f"请基于以下信息回答问题：\n{context_message}"
                })

            # 添加历史对话上下文（如果有）
            if context and 'chat_history' in context:
                # 只添加最近的对话历史，避免重复
                recent_context = [msg for msg in context['chat_history'] if isinstance(msg, dict) and
                                  msg['role'] not in ('system')][-5:]  # 保留最近5轮对话
                prompt_parts.extend(recent_context)

            # 添加当前查询
            prompt_parts.append({
                "role": "user",
                "content": f"{query}\n\n请根据上述信息提供准确、相关的回答。如果信息中没有相关内容，请明确说明。"
            })

            # 3. 生成响应
            response = llm_model.generate_response(
                prompt=query,
                context={"chat_history": prompt_parts}  # 修改为正确的格式
            )

            # 记录完整的提示词用于调试
            logging.debug("完整提示词结构：")
            for part in prompt_parts:
                logging.debug(f"Role: {part.get('role')}")
                logging.debug(f"Content: {part.get('content')}\n")

            return response

        except Exception as e:
            logging.error(f"Error in RAG response generation: {str(e)}")
            return "抱歉，处理您的请求时出现错误。"

    def answer_question(self, question: str) -> Dict:
        """
        使用Raptor回答问题

        Args:
            question: 用户的问题

        Returns:
            Dict: 包含答案和相关信息的字典
        """
        try:
            # 使用Raptor生成答案
            answer = self.RA.answer_question(question=question)
            logger.info(f"成功生成答案，问题: {question}")
            
            return {
                "status": "success",
                "answer": answer,
                "question": question
            }
            
        except Exception as e:
            error_msg = f"回答问题时发生错误: {str(e)}"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}

    def get_documents(self) -> List[Dict]:
        """获取所有文档的信息
        
        Returns:
            List[Dict]: 包含所有文档信息的列表，每个文档包含以下信息：
                - tree_info: 树结构信息
                - status: 操作状态
                - message: 状态信息
        """
        try:
            # 获取树结构信息
            tree_info = self.doc_storage.get_tree_info()
            logger.info("成功获取文档树结构信息")
            logger.info(f"{tree_info}")
            return [{
                "tree_info": tree_info,
                "status": "success",
                "message": "成功获取文档信息"
            }]
            
        except Exception as e:
            error_msg = f"获取文档信息时发生错误: {str(e)}"
            logger.error(error_msg)
            return [{
                "tree_info": {},
                "status": "error",
                "message": error_msg
            }]
