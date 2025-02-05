from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
import faiss
import numpy as np
import os
import logging
import torch
import yaml
import json
from typing import List, Dict
from datetime import datetime
import hashlib
from pathlib import Path
from raptor import RetrievalAugmentation

class RAGModule:
    def __init__(self, config_path: str = "config/config.yaml"):
        """初始化 RAG 模块

        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        embedding_config = config['embedding']
        rerank_config = config['rerank']

        # 从配置中获取参数
        model_path = embedding_config['model_path']
        self.dimension = embedding_config['dimension']
        self.similarity_threshold = embedding_config['similarity_threshold']
        self.index_path = embedding_config['index_path']
        self.docs_path = os.path.join(self.index_path, "documents")

        self.RA = RetrievalAugmentation(tree="data/RAtree")

        try:
            # 初始化 BGE 模型
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModel.from_pretrained(model_path)
            self.model.eval()  # 设置为评估模式
            logging.info(f"Successfully loaded BGE model from {model_path}")
            
            # 初始化 BGE-Rerank 模型
            self.reranker_tokenizer = AutoTokenizer.from_pretrained(rerank_config['model_path'])
            self.reranker = AutoModelForSequenceClassification.from_pretrained(rerank_config['model_path'])
            self.reranker.eval()
            logging.info("Successfully loaded BGE-Rerank model")
        except Exception as e:
            logging.error(f"Error loading models: {str(e)}")
            raise

        # 存储文档和向量
        self.documents = []
        self.document_embeddings = None


        # 确保存储目录存在
        os.makedirs(self.index_path, exist_ok=True)
        os.makedirs(self.docs_path, exist_ok=True)
        
        # 加载或创建索引
        self._load_index()

        # 在索引加载后再添加示例文档
        example_docs = [
            {
                'content': "圣诞老人是一个传统的节日人物，他在圣诞夜乘坐驯鹿雪橇给孩子们送礼物。",
                'file_name': 'example1.txt',
                'file_hash': 'example1',
                'file_type': 'txt',
                'file_size': 100,
                'chunks': ["圣诞老人是一个传统的节日人物，他在圣诞夜乘坐驯鹿雪橇给孩子们送礼物。"],
                'timestamp': datetime.now().isoformat(),
                'status': 'indexed'
            },
            {
                'content': "驯鹿是圣诞老人的好帮手，最著名的是红鼻子驯鹿鲁道夫。",
                'file_name': 'example2.txt',
                'file_hash': 'example2',
                'file_type': 'txt',
                'file_size': 100,
                'chunks': ["驯鹿是圣诞老人的好帮手，最著名的是红鼻子驯鹿鲁道夫。"],
                'timestamp': datetime.now().isoformat(),
                'status': 'indexed'
            },
            {
                'content': "V认为117咖啡没有手冲咖啡好喝，但是比红茶好喝",
                'file_name': 'example3.txt',
                'file_hash': 'example3',
                'file_type': 'txt',
                'file_size': 100,
                'chunks': ["V认为117咖啡没有手冲咖啡好喝，但是比红茶好喝"],
                'timestamp': datetime.now().isoformat(),
                'status': 'indexed'
            },
            {
                'content': "Jamie最喜欢的人是他的老婆和多米",
                'file_name': 'example4.txt',
                'file_hash': 'example4',
                'file_type': 'txt',
                'file_size': 100,
                'chunks': ["Jamie最喜欢的人是他的老婆和多米"],
                'timestamp': datetime.now().isoformat(),
                'status': 'indexed'
            },
            {
                'content': "Jamie是这样一个人：是一位充满探索精神和求知欲的人，尤其在技术领域展现出非凡的好奇心与专注力。",
                'file_name': 'example5.txt',
                'file_hash': 'example5',
                'file_type': 'txt',
                'file_size': 100,
                'chunks': ["Jamie是这样一个人：是一位充满探索精神和求知欲的人，尤其在技术领域展现出非凡的好奇心与专注力。"],
                'timestamp': datetime.now().isoformat(),
                'status': 'indexed'
            }
        ]

        # 只有当文档为空时才添加示例文档
        if not self.documents:
            self.documents.extend(example_docs)
            # 重建索引
            self._rebuild_index()

    def add_file(self, file_path: str, file_name: str = None) -> Dict:
        """添加新文件到知识库
        
        Args:
            file_path: 文件路径
            file_name: 文件名（可选）
            
        Returns:
            Dict: 包含操作结果的字典
        """
        try:
            if not os.path.exists(file_path):
                return {"success": False, "error": "File not found"}
                
            # 计算文件哈希值作为唯一标识
            with open(file_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
                
            # 检查文件是否已存在
            for doc in self.documents:
                if doc.get('file_hash') == file_hash:
                    return {"success": False, "error": "File already exists"}
            
            # 提取文本内容
            text = self._extract_text_from_file(file_path)
            
            # 分割文本
            chunks = self._split_text(text)
            
            # 保存文件到文档目录
            file_name = file_name or os.path.basename(file_path)
            target_path = os.path.join(self.docs_path, file_hash + Path(file_path).suffix)
            with open(file_path, 'rb') as src, open(target_path, 'wb') as dst:
                dst.write(src.read())
            
            # 创建文档记录
            doc_info = {
                'id': len(self.documents),
                'file_name': file_name,
                'file_hash': file_hash,
                'file_type': Path(file_path).suffix[1:],
                'file_size': os.path.getsize(file_path),
                'chunks': chunks,
                'timestamp': datetime.now().isoformat(),
                'status': 'indexed'
            }
            
            # 添加文档内容到索引
            self.add_documents(chunks)
            
            # 更新文档记录
            self.documents.append(doc_info)
            self._save_index()
            
            return {
                "success": True,
                "message": "File added successfully",
                "document": doc_info
            }
            
        except Exception as e:
            logging.error(f"Error adding file: {str(e)}")
            return {"success": False, "error": str(e)}

    def delete_document(self, doc_id: int) -> Dict:
        """从知识库中删除文档

        Args:
            doc_id: 文档ID

        Returns:
            Dict: 包含操作结果的字典
        """
        try:
            if not 0 <= doc_id < len(self.documents):
                return {"success": False, "error": "Document not found"}

            doc = self.documents[doc_id]

            # 删除文件
            file_path = os.path.join(self.docs_path, doc['file_hash'] + '.' + doc['file_type'])
            if os.path.exists(file_path):
                os.remove(file_path)

            # 从列表中移除文档
            self.documents.pop(doc_id)

            # 重建索引
            self._rebuild_index()

            return {
                "success": True,
                "message": "Document deleted successfully"
            }

        except Exception as e:
            logging.error(f"Error deleting document: {str(e)}")
            return {"success": False, "error": str(e)}

    def get_documents(self) -> List[Dict]:
        """获取所有文档的信息"""
        return [{
            'id': i,  # 使用索引作为 id
            'file_name': doc.get('file_name', ''),
            'file_type': doc.get('file_type', ''),
            'file_size': doc.get('file_size', 0),
            'status': doc.get('status', 'unknown'),
            'timestamp': doc.get('timestamp', datetime.now().isoformat())
        } for i, doc in enumerate(self.documents)]

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
                    texts = []
                    for doc in self.documents:
                        texts.extend(doc['chunks'])
                    self.document_embeddings = self._encode_text(texts)
            else:
                self.index = faiss.IndexFlatL2(self.dimension)
                
        except Exception as e:
            logging.error(f"Error loading index: {str(e)}")
            # 创建新的索引
            self.documents = []
            self.document_embeddings = None
            self.index = faiss.IndexFlatL2(self.dimension)

    def _encode_text(self, texts: list) -> np.ndarray:
        """使用 BGE 模型编码文本"""
        try:
            # 添加特殊前缀
            texts = [f"为这个句子生成表示：{text}" for text in texts]
            
            # 对文本进行编码
            with torch.no_grad():
                inputs = self.tokenizer(texts, 
                                      padding=True, 
                                      truncation=True,      
                                      max_length=512, 
                                      return_tensors="pt")
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0].numpy()  # 使用 [CLS] token
                # 归一化
                embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                return embeddings
                
        except Exception as e:
            logging.error(f"Error encoding text: {str(e)}")
            raise

    def _rerank_results(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """使用 BGE-Rerank 对检索结果进行重排序

        Args:
            query: 查询文本
            candidates: 第一阶段检索的候选结果

        Returns:
            重排序后的结果列表
        """
        if not candidates:
            return candidates

        # 准备 rerank 的文本对
        pairs = []
        for doc in candidates:
            # 使用匹配到的文本块进行重排序
            for chunk in doc.get('matched_chunks', []):
                pairs.append([query, chunk])

        # 计算 rerank 分数
        with torch.no_grad():
            inputs = self.reranker_tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512
            )
            scores = self.reranker(**inputs).logits.squeeze()
            scores = torch.sigmoid(scores).tolist()

        # 如果只有一个结果，确保 scores 是列表
        if not isinstance(scores, list):
            scores = [scores]

        # 更新文档分数并重排序
        for doc, score in zip(candidates, scores):
            doc['rerank_score'] = float(score)
            # 综合考虑向量相似度和 rerank 分数
            doc['final_score'] = 0.3 * doc['similarity'] + 0.7 * doc['rerank_score']

        # 按照综合分数重排序
        candidates.sort(key=lambda x: x['final_score'], reverse=True)

        return candidates

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """搜索相关文档，使用两阶段检索策略

        Args:
            query: 查询文本
            top_k: 返回的最大结果数量

        Returns:
            检索到的文档列表，按相关度排序
        """
        try:
            # 如果没有文档或索引，返回空列表
            if not self.documents or self.document_embeddings is None:
                logging.warning("搜索失败：知识库为空或索引未建立")
                return []

            # 编码查询文本
            query_vector = self._encode_text([query])[0]
            logging.info(f"查询文本: {query}")

            # 构建 chunk 到文档的映射关系
            chunk_to_doc = []  # 存储每个 chunk 对应的文档索引和 chunk 内容
            for doc_idx, doc in enumerate(self.documents):
                for chunk in doc.get('chunks', []):
                    chunk_to_doc.append({
                        'doc_idx': doc_idx,
                        'chunk': chunk,
                        'doc_name': doc.get('file_name', '')
                    })

            # 确保有足够的 chunks 可供搜索
            available_chunks = len(chunk_to_doc)
            if available_chunks == 0:
                logging.warning("搜索失败：没有可用的文本块")
                return []

            # 第一阶段：向量检索（放宽相似度阈值，获取更多候选）
            search_k = min(top_k * 3, available_chunks)
            distances, indices = self.index.search(
                np.array([query_vector]).astype('float32'),
                search_k
            )

            # 用于存储每个文档的最佳匹配结果
            doc_best_matches = {}  # doc_idx -> {similarity, chunks, matched_chunks}

            # 处理搜索结果
            for i, idx in enumerate(indices[0]):
                if idx < available_chunks:  # 确保索引有效
                    chunk_info = chunk_to_doc[idx]
                    doc_idx = chunk_info['doc_idx']
                    similarity = 1 - distances[0][i]  # 转换距离为相似度

                    # 记录详细的匹配信息
                    logging.info(f"匹配块 {i+1}:")
                    logging.info(f"  文档: {chunk_info['doc_name']}")
                    logging.info(f"  相似度: {similarity:.4f}")
                    logging.info(f"  内容: {chunk_info['chunk']}")

                    # 更新文档的最佳匹配
                    if doc_idx not in doc_best_matches or similarity > doc_best_matches[doc_idx]['similarity']:
                        doc_best_matches[doc_idx] = {
                            'similarity': similarity,
                            'chunks': [chunk_info['chunk']],
                            'matched_chunks': [chunk_info['chunk']]
                        }
                    else:
                        doc_best_matches[doc_idx]['chunks'].append(chunk_info['chunk'])
                        doc_best_matches[doc_idx]['matched_chunks'].append(chunk_info['chunk'])

            # 准备第一阶段的结果
            initial_results = []
            for doc_idx, match_info in doc_best_matches.items():
                if match_info['similarity'] >= self.similarity_threshold * 0.8:  # 降低阈值获取更多候选
                    doc = self.documents[doc_idx].copy()
                    doc['similarity'] = match_info['similarity']
                    doc['matched_chunks'] = match_info['matched_chunks']
                    doc['chunks'] = match_info['chunks']
                    initial_results.append(doc)

            # 第二阶段：使用 reranker 重排序
            if initial_results:
                reranked_results = self._rerank_results(query, initial_results)
                final_results = reranked_results[:top_k]

                # 记录最终结果
                logging.info(f"\n最终结果 (共 {len(final_results)} 个文档):")
                for doc in final_results:
                    logging.info(f"\n文档: {doc['file_name']}")
                    logging.info(f"向量相似度: {doc['similarity']:.4f}")
                    logging.info(f"Rerank分数: {doc['rerank_score']:.4f}")
                    logging.info(f"综合分数: {doc['final_score']:.4f}")

                return final_results
            else:
                logging.info("没有找到相关文档")
                return []

        except Exception as e:
            logging.error(f"搜索过程中出错: {str(e)}")
            return []

    def generate_response(self, query: str, llm_model,context=None, role_prompt=None):
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
                temp_context = context['context'][0][:3] if len(context['context']) > 0 else []
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