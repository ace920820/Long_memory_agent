import os
import time
import logging
from typing import Dict, List, Optional
from raptor import RetrievalAugmentation
from knowledge_base.storage import DocumentStorage

# 配置日志记录
logging.basicConfig(level=logging.INFO)
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
    
    def __init__(self, data_dir: str = "data/RAtree"):
        """
        初始化RaptorModule

        Args:
            data_dir: Raptor数据存储目录，默认为 "data/RAtree"
        """
        try:
            # 初始化DocumentStorage实例
            self.doc_storage = DocumentStorage(
                storage_dir=os.path.join(data_dir, "files"),
                tree_save_path=os.path.join(data_dir, "default_tree")
            )
            # 获取DocumentStorage中的RA实例
            self.RA = self.doc_storage.RA
            logger.info("成功初始化Raptor检索增强模块")
            
        except Exception as e:
            logger.error(f"初始化RaptorModule时发生错误: {str(e)}")
            raise
    
    def add_documents(self, text: str) -> Dict:
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
            
            # 保存树结构
            self.doc_storage.save_RA_tree()
            logger.info(f"成功添加文档到树结构，文本长度: {len(text)}")
            
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

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """搜索相关文档，使用两阶段检索策略

        Args:
            query: 查询文本
            top_k: 返回的最大结果数量

        Returns:
            检索到的文档列表，按相关度排序
        """
        start_layer: int = None
        num_layers: int = None
        max_tokens: int = 3500
        collapse_tree: bool = True
        try:
           context, layer_information = self.RA.retrieve(
               query, start_layer, num_layers, top_k, max_tokens, collapse_tree, True)
           return context
        except Exception as e:
            logging.error(f"搜索过程中出错: {str(e)}")
            return []

    def generate_response(self, query: str, llm_model, context=None, role_prompt=None):
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
