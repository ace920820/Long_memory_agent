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

    def generate_response(self, query: str, llm_model, role_prompt: str = None, context: list = None, memories: list = None) -> str:
        """生成基于上下文增强的回答。
        :param query: 用户查询
        :param llm_model: LLM模型实例
        :param role_prompt: 角色提示词
        :param context: 对话历史上下文
        :param memories: 相关记忆列表
        :return: 回答文本
        """
        try:
            relevant_docs = self.search(query, top_k=5)
            
            # 构建知识库内容
            docs_context = []
            if relevant_docs:
                docs_context.extend([doc["document"] for doc in relevant_docs])
            
            # 添加记忆内容
            if memories:
                for m in memories:
                    first_line = m['content'].split('\n')[0]
                    docs_context.append(f"历史记忆: {first_line}")
            
            # 如果没有相关文档和记忆，且有上下文，直接使用上下文
            if not docs_context and context:
                context.append({"role": "user", "content": query})
                return llm_model.generate_response(query, context)

            # 构建完整的系统提示词
            system_prompt = "你是一个知识丰富的助手，需要基于提供的参考信息来回答问题。"
            if role_prompt:
                system_prompt = f"{role_prompt}\n\n同时，你需要基于提供的参考信息来回答问题。"

            # 构建带有知识库内容的提示词
            knowledge_prompt = f"""基于以下参考信息回答问题：

参考信息：
{chr(10).join(docs_context)}

问题：{query}

请根据上述参考信息提供准确、相关的回答。"""

            # 合并上下文
            conversation_context = context or []
            if not any(msg.get("role") == "system" for msg in conversation_context):
                conversation_context.insert(0, {"role": "system", "content": system_prompt})
            
            conversation_context.append({"role": "user", "content": knowledge_prompt})

            return llm_model.generate_response(knowledge_prompt, conversation_context)
            
        except Exception as e:
            logging.error(f"Response generation failed: {str(e)}")
            return "抱歉，生成回答时出现错误。"

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