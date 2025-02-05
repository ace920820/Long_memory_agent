import os
import logging
from typing import Dict, List, Optional
from raptor import RetrievalAugmentation

# 配置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
            # # 确保数据目录存在
            # os.makedirs(data_dir, exist_ok=True)
            # logger.info(f"数据目录已确认: {data_dir}")
            
            # 初始化Raptor的RetrievalAugmentation
            self.RA = RetrievalAugmentation(tree=data_dir)
            logger.info("成功初始化Raptor检索增强模块")
            
        except Exception as e:
            logger.error(f"初始化RaptorModule时发生错误: {str(e)}")
            raise
    
    def add_documents(self, text: str) -> Dict:
        """
        向知识库中添加文档

        Args:
            text: 要添加的文本内容

        Returns:
            Dict: 包含操作结果的字典
        """
        try:
            # 添加文档到Raptor
            self.RA.add_documents(text)
            logger.info(f"成功添加文档，文本长度: {len(text)}")
            return {"status": "success", "message": "文档添加成功"}
            
        except Exception as e:
            error_msg = f"添加文档时发生错误: {str(e)}"
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
