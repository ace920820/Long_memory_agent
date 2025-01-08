from transformers import AutoModel, AutoTokenizer
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
import docx
import PyPDF2
import markdown


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
        
        # 从配置中获取参数
        model_path = embedding_config['model_path']
        self.dimension = embedding_config['dimension']
        self.similarity_threshold = embedding_config['similarity_threshold']
        self.index_path = embedding_config['index_path']
        self.docs_path = os.path.join(self.index_path, "documents")

        try:
            # 初始化 BGE 模型
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModel.from_pretrained(model_path)
            self.model.eval()  # 设置为评估模式
            logging.info(f"Successfully loaded BGE model from {model_path}")
        except Exception as e:
            logging.error(f"Error loading BGE model: {str(e)}")
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

    def _extract_text_from_file(self, file_path: str) -> str:
        """从不同类型的文件中提取文本内容"""
        file_ext = Path(file_path).suffix.lower()
        
        try:
            if file_ext == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
                    
            elif file_ext == '.docx':
                doc = docx.Document(file_path)
                return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
                
            elif file_ext == '.pdf':
                text = []
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    for page in pdf_reader.pages:
                        text.append(page.extract_text())
                return '\n'.join(text)
                
            elif file_ext == '.md':
                with open(file_path, 'r', encoding='utf-8') as f:
                    md_text = f.read()
                    html = markdown.markdown(md_text)
                    # 简单移除HTML标签
                    text = html.replace('<p>', '').replace('</p>', '\n')
                    return text
                    
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")
                
        except Exception as e:
            logging.error(f"Error extracting text from {file_path}: {str(e)}")
            raise

    def _split_text(self, text: str, max_length: int = 200, overlap: int = 30) -> List[str]:
        """将长文本分割成较小的片段，使用基于语义的递归分割策略

        Args:
            text: 要分割的文本
            max_length: 每个文本块的目标长度
            overlap: 相邻文本块之间的最小重叠字符数

        Returns:
            List[str]: 分割后的文本块列表
        """
        # 定义分隔符，按语义完整性优先级排序
        delimiters = [
            "\n\n",  # 段落分隔符
            "\n",    # 换行符
            "。",    # 句号
            "！",    # 感叹号
            "？",    # 问号
            "；",    # 分号
            "：",    # 冒号
            "，",    # 逗号
            "、",    # 顿号
            " ",    # 空格
            ""      # 无分隔符，按字符分割
        ]
        
        # 允许的长度浮动范围（±10%）
        length_margin = max_length * 0.1
        min_length = max_length - length_margin
        max_length_with_margin = max_length + length_margin

        def find_semantic_split_point(text: str, target_length: int, delimiter: str) -> int:
            """在目标长度附近找到最合适的分割点"""
            if delimiter == "":
                return target_length
                
            # 在目标长度前后寻找最近的分隔符
            left_bound = max(0, target_length - length_margin)
            right_bound = min(len(text), target_length + length_margin)
            
            # 在合理范围内寻找分隔符
            text_range = text[left_bound:right_bound]
            last_delimiter_pos = text_range.rfind(delimiter)
            
            if last_delimiter_pos != -1:
                return left_bound + last_delimiter_pos + len(delimiter)
            return target_length

        def split_by_delimiter(text: str, delimiter: str) -> List[str]:
            """使用指定的分隔符智能分割文本"""
            if delimiter == "":
                return [char for char in text]
            
            # 保留分隔符，确保语义完整性
            segments = []
            for segment in text.split(delimiter):
                if segment.strip():
                    # 如果不是最后一个分段，添加分隔符
                    if segment != text.split(delimiter)[-1]:
                        segments.append(segment.strip() + delimiter)
                    else:
                        segments.append(segment.strip())
            return segments

        def recursive_split(text: str, delimiters: List[str], current_level: int = 0) -> List[str]:
            """递归分割文本，保持语义完整性"""
            # 如果文本长度在可接受范围内，直接返回
            if len(text) <= max_length_with_margin:
                return [text]
            
            # 如果已经尝试了所有分隔符，则寻找最佳分割点
            if current_level >= len(delimiters):
                chunks = []
                start = 0
                while start < len(text):
                    # 计算当前块的理想长度
                    remaining_length = len(text) - start
                    current_max_length = min(max_length_with_margin, remaining_length)
                    
                    # 寻找最佳分割点
                    split_point = find_semantic_split_point(
                        text[start:], 
                        current_max_length, 
                        delimiters[-1]
                    )
                    
                    chunks.append(text[start:start + split_point])
                    start += split_point - overlap
                return chunks
            
            # 使用当前级别的分隔符分割
            delimiter = delimiters[current_level]
            segments = split_by_delimiter(text, delimiter)
            
            # 如果分割效果不理想，尝试下一个分隔符
            if len(segments) <= 1:
                return recursive_split(text, delimiters, current_level + 1)
            
            # 处理分割后的片段
            chunks = []
            current_chunk = []
            current_length = 0
            
            for segment in segments:
                # 如果当前片段过长，递归分割
                if len(segment) > max_length_with_margin:
                    # 处理当前累积的chunk
                    if current_chunk:
                        chunks.append(''.join(current_chunk))
                        current_chunk = []
                        current_length = 0
                    
                    # 递归处理长片段
                    sub_chunks = recursive_split(segment, delimiters, current_level + 1)
                    chunks.extend(sub_chunks)
                else:
                    # 检查添加当前片段是否会导致chunk过长
                    if current_length + len(segment) > max_length_with_margin:
                        if current_chunk:
                            chunks.append(''.join(current_chunk))
                        current_chunk = [segment]
                        current_length = len(segment)
                    else:
                        current_chunk.append(segment)
                        current_length += len(segment)
            
            # 处理最后一个chunk
            if current_chunk:
                chunks.append(''.join(current_chunk))
            
            # 智能处理重叠区域
            if overlap > 0 and len(chunks) > 1:
                overlapped_chunks = []
                for i in range(len(chunks)):
                    if i == 0:
                        overlapped_chunks.append(chunks[i])
                    else:
                        # 在重叠区域内寻找合适的语义边界
                        prev_chunk = chunks[i-1]
                        overlap_start = len(prev_chunk) - min(overlap * 2, len(prev_chunk))
                        overlap_text = prev_chunk[overlap_start:]
                        
                        # 在重叠文本中找到最后一个完整的语义单元
                        best_split_point = 0
                        for delimiter in delimiters:
                            if delimiter == "":
                                continue
                            last_pos = overlap_text.rfind(delimiter)
                            if last_pos != -1:
                                best_split_point = overlap_start + last_pos + len(delimiter)
                                break
                        
                        # 如果找不到合适的分割点，使用默认重叠长度
                        if best_split_point == 0:
                            best_split_point = len(prev_chunk) - overlap
                        
                        overlap_text = prev_chunk[best_split_point:]
                        overlapped_chunks.append(overlap_text + chunks[i])
                
                chunks = overlapped_chunks
            
            return chunks
        
        # 开始递归分割
        return recursive_split(text, delimiters)

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

    def _rebuild_index(self):
        """重建向量索引"""
        try:
            # 收集所有文档片段
            all_chunks = []
            for doc in self.documents:
                all_chunks.extend(doc['chunks'])
            
            # 重新编码所有文档
            if all_chunks:
                self.document_embeddings = self._encode_text(all_chunks)
                self.index = faiss.IndexFlatL2(self.dimension)
                self.index.add(self.document_embeddings.astype('float32'))
            else:
                self.document_embeddings = None
                self.index = faiss.IndexFlatL2(self.dimension)
            
            # 保存更新
            self._save_index()
            
        except Exception as e:
            logging.error(f"Error rebuilding index: {str(e)}")
            raise

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

    def add_documents(self, documents: List[Dict]) -> None:
        """添加新文档到知识库
        
        Args:
            documents: 文档列表，每个文档都是一个包含必要字段的字典
        """
        try:
            # 编码新文档
            all_chunks = []
            for doc in documents:
                if isinstance(doc, dict):
                    all_chunks.extend(doc['chunks'])
                else:
                    # 处理纯文本输入的情况
                    all_chunks.append(doc)
            
            new_embeddings = self._encode_text(all_chunks)
            
            # 更新索引
            self.index.add(new_embeddings.astype('float32'))
            
            # 更新文档向量
            if self.document_embeddings is None:
                self.document_embeddings = new_embeddings
            else:
                self.document_embeddings = np.vstack([self.document_embeddings, new_embeddings])
            
            # 保存更新
            self._save_index()
            
            logging.info(f"Successfully added {len(documents)} documents")
            
        except Exception as e:
            logging.error(f"Error adding documents: {str(e)}")
            raise

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
                logging.info("知识库为空，无法进行搜索")
                return []
                
            # 编码查询
            query_vector = self._encode_text([query])[0]
            logging.info(f"正在搜索查询: '{query}'")
            
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
                    similarity = 1 - distances[0][i]  # 转换距离为相似度
                    doc['similarity'] = similarity
                    
                    # 记录每个候选文档的信息
                    log_msg = (
                        f"\n候选文档 {idx}:"
                        f"\n - 内容: {doc['content']}"
                        f"\n - 相似度得分: {similarity:.4f}"
                    )
                    
                    if similarity >= self.similarity_threshold:
                        results.append(doc)
                        log_msg += f"\n - 状态: 采用 (得分 >= {self.similarity_threshold})"
                    else:
                        log_msg += f"\n - 状态: 丢弃 (得分 < {self.similarity_threshold})"
                    
                    logging.info(log_msg)
            
            logging.info(f"共找到 {len(results)}/{len(indices[0])} 个相关文档")
            
            return results
            
        except Exception as e:
            logging.error(f"搜索文档时发生错误: {str(e)}")
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

    def generate_response(self, query: str, llm_model,context=None, role_prompt=None):
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
            
            # 添加context中的文本块信息
            if context and 'context' in context and isinstance(context['context'], list):
                context_info.append("\n已知信息：")
                context_info.extend(context['context'])
            
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