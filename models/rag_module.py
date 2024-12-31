from sentence_transformers import SentenceTransformer, util
import faiss
import numpy as np
import os
import logging


class RAGModule:
    def __init__(self,
                 model_name: str = "all-MiniLM-L6-v2",
                 similarity_threshold: float = 0.5,
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
        self.dimension = 384  # all-MiniLM-L6-v2 的向量维度

        # 初始化或加载索引
        self._init_index()

        # 存储文档
        self.documents = []

        # 添加示例文档
        self.add_documents([
            "圣诞老人是一个传统的节日人物，他在圣诞夜乘坐驯鹿雪橇给孩子们送礼物。",
            "驯鹿是圣诞老人的好帮手，最著名的是红鼻子驯鹿鲁道夫。",
            "V认为117咖啡没有手冲咖啡好喝，但是比红茶好喝",
            "Jamie最喜欢的人是他的老婆和多米",
            "Jamie是这样一个人：是一位充满探索精神和求知欲的人，尤其在技术领域展现出非凡的好奇心与专注力。"
        ])

    def _init_index(self):
        """初始化或加载 FAISS 索引"""
        try:
            # 确保目录存在
            os.makedirs(self.index_path, exist_ok=True)

            # 创建新的索引
            self.index = faiss.IndexFlatL2(self.dimension)

            # 如果存在已保存的索引，则加载
            index_file = os.path.join(self.index_path, "index.faiss")
            if os.path.exists(index_file):
                self.index = faiss.read_index(index_file)
                logging.info(f"Loaded existing index from {index_file}")
                # 验证索引和文档的一致性
                if self.index.ntotal > len(self.documents):
                    logging.warning("Index contains more vectors than documents. Truncating.")
                    self.index = faiss.IndexFlatL2(self.dimension)  # 重新初始化索引

        except Exception as e:
            logging.error(f"Error initializing index: {str(e)}")
            # 创建空索引作为后备
            self.index = faiss.IndexFlatL2(self.dimension)

    def add_documents(self, documents: list):
        """添加文档到知识库"""
        try:
            if not documents:
                return

            # 编码文档
            embeddings = self.model.encode(documents)
            assert len(embeddings) == len(documents), "Mismatch between embeddings and documents"

            # 添加到索引
            self.index.add(embeddings.astype('float32'))

            # 保存文档
            self.documents.extend(documents)

            # 保存索引
            index_file = os.path.join(self.index_path, "index.faiss")
            try:
                faiss.write_index(self.index, index_file)
                logging.info(f"Successfully saved index to {index_file}")
            except Exception as e:
                logging.error(f"Error saving index: {str(e)}")

            logging.info(f"Added {len(documents)} documents to the knowledge base")

        except Exception as e:
            logging.error(f"Error adding documents: {str(e)}")

    def search(self, query: str, top_k: int = 5) -> list:
        """搜索相关文档

        Args:
            query: 查询文本
            top_k: 返回的最相关文档数量

        Returns:
            list: 相关文档列表，每个文档包含内容和相似度分数
        """
        try:
            # 编码查询
            query_vector = self.model.encode([query])

            # 确保搜索参数不超过索引大小
            k = min(top_k, self.index.ntotal)
            distances, indices = self.index.search(query_vector.astype('float32'), k)

            # 处理结果
            results = []
            for i, idx in enumerate(indices[0]):
                if idx >= len(self.documents):
                    logging.warning(f"Index {idx} is out of range for documents. Skipping.")
                    continue
                similarity = 1 / (1 + float(distances[0][i]))  # 转换距离为相似度
                if similarity >= self.similarity_threshold:
                    results.append({
                        'document': self.documents[idx],
                        'score': similarity
                    })
                    logging.info(f"知识库匹配 (得分: {similarity:.4f}):\n文本: {self.documents[idx]}")
                else:
                    logging.info(f"知识库文本因相似度过低被过滤 (得分: {similarity:.4f}):\n文本: {self.documents[idx]}")

            return results

        except Exception as e:
            logging.error(f"Error in search: {str(e)}")
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