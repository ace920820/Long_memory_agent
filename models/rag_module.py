from sentence_transformers import SentenceTransformer, util
import faiss
import numpy as np
import os
import logging
import json
from typing import List, Dict


class RAGModule:
    def __init__(self,
                 model_name: str = "all-MiniLM-L6-v2",
                 similarity_threshold: float = 0.6,
                 index_path: str = "data/vector_store"):
        """初始化 RAG 模块

        Args:
            model_name: 使用的嵌入模型名称
            similarity_threshold: 相似度阈值
            index_path: 向量存储路径
        """
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        self.index_path = index_path
        
        # 初始化文档存储
        self.documents = []
        self.document_embeddings = None
        
        # 确保存储目录存在
        os.makedirs(index_path, exist_ok=True)
        
        # 加载现有索引
        self._load_index()

    def _load_index(self):
        """加载或创建向量索引"""
        try:
            # 加载文档
            docs_path = os.path.join(self.index_path, "documents.json")
            if os.path.exists(docs_path):
                with open(docs_path, 'r', encoding='utf-8') as f:
                    self.documents = json.load(f)
                    
            # 加载或创建向量索引
            index_path = os.path.join(self.index_path, "faiss_index.bin")
            if os.path.exists(index_path):
                self.index = faiss.read_index(index_path)
                # 重新计算文档向量
                if self.documents:
                    texts = [doc['content'] for doc in self.documents]
                    self.document_embeddings = self.model.encode(texts)
            else:
                self.index = faiss.IndexFlatL2(self.model.get_sentence_embedding_dimension())
                
        except Exception as e:
            logging.error(f"Error loading index: {str(e)}")
            # 创建新的索引
            self.documents = []
            self.document_embeddings = None
            self.index = faiss.IndexFlatL2(self.model.get_sentence_embedding_dimension())

    def add_documents(self, documents: List[Dict]):
        """添加新文档到索引
        
        Args:
            documents: 文档列表，每个文档应包含 'content' 和 'metadata' 字段
        """
        if not documents:
            return
            
        try:
            # 编码新文档
            texts = [doc['content'] for doc in documents]
            new_embeddings = self.model.encode(texts)
            
            # 更新索引
            self.index.add(new_embeddings.astype('float32'))
            
            # 更新文档存储
            start_idx = len(self.documents)
            for i, doc in enumerate(documents):
                doc['id'] = start_idx + i
                self.documents.append(doc)
            
            # 更新文档向量
            if self.document_embeddings is None:
                self.document_embeddings = new_embeddings
            else:
                self.document_embeddings = np.vstack([self.document_embeddings, new_embeddings])
            
            # 保存更新
            self._save_index()
            
        except Exception as e:
            logging.error(f"Error adding documents: {str(e)}")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """搜索相关文档
        
        Args:
            query: 查询文本
            top_k: 返回的最相关文档数量
            
        Returns:
            相关文档列表，每个文档包含相似度分数
        """
        try:
            if not self.documents:
                return []
                
            # 编码查询
            query_vector = self.model.encode([query])[0]
            
            # 搜索相似向量
            distances, indices = self.index.search(
                np.array([query_vector]).astype('float32'), 
                min(top_k, len(self.documents))
            )
            
            # 构建结果
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.documents):  # 确保索引有效
                    doc = self.documents[idx].copy()
                    doc['similarity'] = 1 - distances[0][i]  # 转换距离为相似度
                    if doc['similarity'] >= self.similarity_threshold:
                        results.append(doc)
            
            return results
            
        except Exception as e:
            logging.error(f"Error searching documents: {str(e)}")
            return []

    def _save_index(self):
        """保存索引和文档到磁盘"""
        try:
            # 保存文档
            docs_path = os.path.join(self.index_path, "documents.json")
            with open(docs_path, 'w', encoding='utf-8') as f:
                json.dump(self.documents, f, ensure_ascii=False, indent=2)
            
            # 保存索引
            index_path = os.path.join(self.index_path, "faiss_index.bin")
            faiss.write_index(self.index, index_path)
            
        except Exception as e:
            logging.error(f"Error saving index: {str(e)}")

    def generate_response(self, query: str, llm_model, role_prompt=None, context=None, memories=None):
        """生成带有检索增强的响应"""
        try:
            # 1. 获取相关文档
            relevant_docs = self.search(query, top_k=5)  # 使用现有的 search 方法
            
            # 2. 构建提示词
            prompt_parts = []
            
            # 添加角色提示（如果有）
            if role_prompt:
                prompt_parts.append({"role": "system", "content": role_prompt})
            
            # 构建上下文信息
            context_info = []
            
            # 添加相关文档
            if relevant_docs:
                context_info.append("参考信息：")
                # 从搜索结果中提取文档内容
                doc_contents = [doc["document"] for doc in relevant_docs]
                context_info.extend(doc_contents)
            
            # 添加记忆信息（去重）
            if memories:
                seen_contents = set()
                memory_info = []
                for memory in memories:
                    content = memory.get('content', '')
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
                    "content": f"请记住以下信息：\n{context_message}"
                })
            
            # 添加历史对话上下文（如果有）
            if context:
                # 只添加最近的对话历史，避免重复
                recent_context = [msg for msg in context if isinstance(msg, dict) and 
                                msg['role'] not in ('system')][-5:]  # 保留最近5轮对话
                prompt_parts.extend(recent_context)
            
            # 添加当前查询
            prompt_parts.append({
                "role": "user",
                "content": f"{query}\n\n请根据上述信息提供准确、相关的回答。"
            })
            
            # 3. 生成响应
            response = llm_model.generate_response(
                prompt=query,
                context=prompt_parts
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

    def update_document(self, doc_id: int, new_content: str) -> bool:
        """更新知识库中的文档。
        :param doc_id: 文档ID
        :param new_content: 新的文档内容
        :return: 是否成功更新
        """
        try:
            if 0 <= doc_id < len(self.documents):
                self.documents[doc_id] = new_content
                embedding = self.model.encode([new_content])
                self.index.remove_ids([doc_id])
                self.index.add(embedding)
                logging.info(f"Successfully updated document {doc_id}")
                return True
            return False
        except Exception as e:
            logging.error(f"Failed to update document: {str(e)}")
            return False

    def remove_document(self, doc_id: int) -> bool:
        """删除知识库中的文档。
        :param doc_id: 文档ID
        :return: 是否成功删除
        """
        try:
            if 0 <= doc_id < len(self.documents):
                self.documents.pop(doc_id)
                # 重建索引
                embeddings = self.model.encode(self.documents)
                self.index = faiss.IndexFlatL2(self.model.get_sentence_embedding_dimension())
                self.index.add(embeddings)
                logging.info(f"Successfully removed document {doc_id}")
                return True
            return False
        except Exception as e:
            logging.error(f"Failed to remove document: {str(e)}")
            return False 