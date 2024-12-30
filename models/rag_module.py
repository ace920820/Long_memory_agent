import faiss
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import logging

class RAGModule:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", index_type: str = "Flat", similarity_threshold: float = 0.6):
        """初始化RAG模块
        :param model_name: 嵌入模型的名称
        :param index_type: FAISS索引类型（如Flat, IVF, HNSW）
        :param similarity_threshold: 相似度阈值，低于此值的文档将被过滤掉
        """
        try:
            self.model = SentenceTransformer(model_name)
            self.index = faiss.IndexFlatL2(self.model.get_sentence_embedding_dimension())
            self.doc_store = []  # 用于存储原始文档及其ID
            self.similarity_threshold = similarity_threshold
            logging.info(f"Successfully initialized RAG module with model {model_name}")
        except Exception as e:
            logging.error(f"Failed to initialize RAG module: {str(e)}")
            raise

    def add_documents(self, documents: List[str]) -> bool:
        """向知识库中添加文档。
        :param documents: 文档列表
        :return: 是否成功添加
        """
        try:
            embeddings = self.model.encode(documents)
            self.index.add(embeddings)
            self.doc_store.extend(documents)
            logging.info(f"Successfully added {len(documents)} documents to knowledge base")
            return True
        except Exception as e:
            logging.error(f"Failed to add documents: {str(e)}")
            return False

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, str]]:
        """检索与查询最相关的文档。"""
        try:
            query_embedding = self.model.encode([query])
            distances, indices = self.index.search(query_embedding, top_k)
            results = []
            
            # FAISS返回的是L2距离，需要转换为相似度分数
            for i, idx in enumerate(indices[0]):
                if idx < len(self.doc_store):
                    similarity_score = 1 / (1 + float(distances[0][i]))
                    if similarity_score >= self.similarity_threshold:
                        doc = self.doc_store[idx]
                        results.append({
                            "document": doc,
                            "score": similarity_score
                        })
                        logging.info(f"知识库匹配 (得分: {similarity_score:.4f}):\n文本: {doc}")
                    else:
                        logging.info(f"知识库文本因相似度过低被过滤 (得分: {similarity_score:.4f}):\n文本: {self.doc_store[idx]}")
            
            return results
        except Exception as e:
            logging.error(f"Search failed: {str(e)}")
            return []

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
            if 0 <= doc_id < len(self.doc_store):
                self.doc_store[doc_id] = new_content
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
            if 0 <= doc_id < len(self.doc_store):
                self.doc_store.pop(doc_id)
                # 重建索引
                embeddings = self.model.encode(self.doc_store)
                self.index = faiss.IndexFlatL2(self.model.get_sentence_embedding_dimension())
                self.index.add(embeddings)
                logging.info(f"Successfully removed document {doc_id}")
                return True
            return False
        except Exception as e:
            logging.error(f"Failed to remove document: {str(e)}")
            return False 