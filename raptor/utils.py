import logging
import re
from typing import Dict, List, Set, Optional, Union, Tuple

import numpy as np
import tiktoken
from scipy import spatial
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .tree_structures import Node

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.DEBUG)


def reverse_mapping(layer_to_nodes: Dict[int, List[Node]]) -> Dict[Node, int]:
    node_to_layer = {}
    for layer, nodes in layer_to_nodes.items():
        for node in nodes:
            node_to_layer[node.index] = layer
    return node_to_layer


def split_text(
        text: str, tokenizer: tiktoken.get_encoding("cl100k_base"), max_tokens: int, overlap: int = 0
):
    """
    Splits the input text into smaller chunks based on the tokenizer and maximum allowed tokens.

    Args:
        text (str): The text to be split.
        tokenizer (CustomTokenizer): The tokenizer to be used for splitting the text.
        max_tokens (int): The maximum allowed tokens.
        overlap (int, optional): The number of overlapping tokens between chunks. Defaults to 0.

    Returns:
        List[str]: A list of text chunks.
    """
    # Split the text into sentences using multiple delimiters
    delimiters = ["。", "！", "？", "!", "?", "\n"]
    regex_pattern = "|".join(map(re.escape, delimiters))
    sentences = re.split(regex_pattern, text)
    overlap = 0

    # Calculate the number of tokens for each sentence
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence, token_count in zip(sentences, n_tokens):
        # If the sentence is empty or consists only of whitespace, skip it
        if not sentence.strip():
            continue

        # If the sentence is too long, split it into smaller parts
        if token_count > max_tokens:
            sub_sentences = re.split(r"[,;:]", sentence)

            # there is no need to keep empty os only-spaced strings
            # since spaces will be inserted in the beginning of the full string
            # and in between the string in the sub_chuk list
            filtered_sub_sentences = [sub.strip() for sub in sub_sentences if sub.strip() != ""]
            sub_token_counts = [len(tokenizer.encode(" " + sub_sentence)) for sub_sentence in filtered_sub_sentences]

            sub_chunk = []
            sub_length = 0

            for sub_sentence, sub_token_count in zip(filtered_sub_sentences, sub_token_counts):
                if sub_length + sub_token_count > max_tokens:

                    # if the phrase does not have sub_sentences, it would create an empty chunk
                    # this big phrase would be added anyways in the next chunk append
                    if sub_chunk:
                        chunks.append(" ".join(sub_chunk))
                        sub_chunk = sub_chunk[-overlap:] if overlap > 0 else []
                        sub_length = sum(sub_token_counts[max(0, len(sub_chunk) - overlap):len(sub_chunk)])

                sub_chunk.append(sub_sentence)
                sub_length += sub_token_count

            if sub_chunk:
                chunks.append(" ".join(sub_chunk))

        # If adding the sentence to the current chunk exceeds the max tokens, start a new chunk
        elif current_length + token_count > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap:] if overlap > 0 else []
            current_length = sum(n_tokens[max(0, len(current_chunk) - overlap):len(current_chunk)])
            current_chunk.append(sentence)
            current_length += token_count

        # Otherwise, add the sentence to the current chunk
        else:
            current_chunk.append(sentence)
            current_length += token_count

    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def recursive_split_text(
    text: str, tokenizer: tiktoken.get_encoding("cl100k_base"), max_tokens: int, overlap: int = 0
):
    """
    使用递归方式将输入文本切分成更小的块，基于分词器和最大允许的token数。
    
    Args:
        text (str): 需要被切分的文本。
        tokenizer (CustomTokenizer): 用于切分文本的分词器。
        max_tokens (int): 每个文本块允许的最大token数。
        overlap (int, optional): 相邻文本块之间的重叠token数。默认为0。
    
    Returns:
        List[str]: 文本块列表。
    """
    logging.debug(f"开始递归切分文本，文本长度：{len(text)}，最大token数：{max_tokens}，重叠token数：{overlap}")
    
    # 定义递归切分函数
    def _recursive_split(text_segment, delimiters_idx=0):
        # 定义分隔符列表，按优先级排序
        delimiters = [
            [".", "!", "?", "。", "！", "？","\n"],  # 主要分隔符（句子边界）
            [",", ";", ":"],        # 次要分隔符（子句边界）
            [" "]                   # 最后的分隔符（单词边界）
        ]
        
        # 对文本进行编码，检查token数量
        tokens = tokenizer.encode(text_segment)
        logging.debug(f"当前文本段token数：{len(tokens)}")
        
        # 如果文本段token数量小于最大允许值，直接返回
        if len(tokens) <= max_tokens:
            return [text_segment]
        
        # 如果已经用尽所有分隔符，但文本仍然太长，强制切分
        if delimiters_idx >= len(delimiters):
            logging.warning(f"已用尽所有分隔符，文本仍然过长，进行强制切分。文本长度：{len(text_segment)}")
            # 计算可以切分出多少个块
            n_tokens = len(tokens)
            chunks = []
            for i in range(0, n_tokens, max_tokens - overlap):
                # 从tokens中切分出一部分，并解码回文本
                chunk_tokens = tokens[i:i + max_tokens]
                chunk_text = tokenizer.decode(chunk_tokens)
                chunks.append(chunk_text)
            return chunks
        
        # 使用当前级别的分隔符切分文本
        current_delimiters = delimiters[delimiters_idx]
        regex_pattern = "|".join(map(re.escape, current_delimiters))
        segments = re.split(f"({regex_pattern})", text_segment)
        
        # 重组segments，保留分隔符
        full_segments = []
        for i in range(0, len(segments), 2):
            if i + 1 < len(segments):
                full_segments.append(segments[i] + segments[i + 1])
            else:
                full_segments.append(segments[i])
        
        # 如果当前分隔符无法有效切分文本（只得到一个段落），尝试下一级分隔符
        if len(full_segments) <= 1:
            logging.debug(f"使用分隔符 {current_delimiters} 无法有效切分文本，尝试下一级分隔符")
            return _recursive_split(text_segment, delimiters_idx + 1)
        
        # 处理切分后的段落
        result_chunks = []
        current_chunk = []
        current_length = 0
        
        for segment in full_segments:
            # 跳过空段落
            if not segment.strip():
                continue
                
            segment_tokens = len(tokenizer.encode(segment))
            
            # 如果单个段落超过最大token数，递归切分
            if segment_tokens > max_tokens:
                logging.debug(f"单个段落token数 ({segment_tokens}) 超过最大限制 ({max_tokens})，递归切分此段落")
                sub_chunks = _recursive_split(segment, delimiters_idx + 1)
                result_chunks.extend(sub_chunks)
            
            # 如果加入当前段落会超出最大token数，先保存当前chunk，再开始新chunk
            elif current_length + segment_tokens > max_tokens:
                result_chunks.append("".join(current_chunk))
                # 处理重叠部分
                if overlap > 0 and current_chunk:
                    # 获取当前chunk的最后overlap个tokens对应的文本作为新chunk的开始
                    overlap_text = ""
                    overlap_tokens = 0
                    for i in range(len(current_chunk) - 1, -1, -1):
                        segment_overlap_tokens = len(tokenizer.encode(current_chunk[i]))
                        if overlap_tokens + segment_overlap_tokens <= overlap:
                            overlap_text = current_chunk[i] + overlap_text
                            overlap_tokens += segment_overlap_tokens
                        else:
                            break
                    current_chunk = [overlap_text]
                    current_length = overlap_tokens
                else:
                    current_chunk = []
                    current_length = 0
                
                current_chunk.append(segment)
                current_length += segment_tokens
            
            # 否则，将段落添加到当前chunk
            else:
                current_chunk.append(segment)
                current_length += segment_tokens
        
        # 添加最后一个chunk（如果有）
        if current_chunk:
            result_chunks.append("".join(current_chunk))
        
        return result_chunks
    
    # 开始递归切分
    chunks = _recursive_split(text)
    logging.debug(f"递归切分完成，共生成 {len(chunks)} 个文本块")
    return chunks


def distances_from_embeddings(
    query_embedding: List[float],
    embeddings: List[List[float]],
    distance_metric: str = "cosine",
    rerank: bool = False,
    query_text: Optional[str] = None,
    node_texts: Optional[List[str]] = None,
    reranker = None,
    reranker_tokenizer = None,
) -> Union[List[float], Tuple[List[float], List[float]]]:
    """
    计算查询嵌入与文档嵌入之间的距离

    Args:
        query_embedding: 查询文本的嵌入向量
        embeddings: 文档嵌入向量列表
        distance_metric: 距离度量方式，支持 'cosine', 'L1', 'L2', 'Linf'
        rerank: 是否使用重排序
        query_text: 查询文本，仅当 rerank=True 时使用
        node_texts: 节点文本列表，仅当 rerank=True 时使用
        reranker: 重排序模型，如果为None且rerank=True，则加载全局配置中的模型
        reranker_tokenizer: 重排序模型的分词器，如果为None且rerank=True，则加载全局配置中的分词器

    Returns:
        距离列表或同时包含距离列表和重排序分数的元组
    """
    try:
        # 检查嵌入向量是否有效
        if query_embedding is None:
            logging.error("查询嵌入向量为None，无法计算距离")
            return [1.0] * len(embeddings)  # 返回最大距离
            
        # 过滤掉None值的嵌入向量，同时记录对应的索引
        valid_embeddings = []
        valid_indices = []
        for i, emb in enumerate(embeddings):
            if emb is not None:
                valid_embeddings.append(emb)
                valid_indices.append(i)
                
        if not valid_embeddings:
            logging.error("没有有效的嵌入向量，无法计算距离")
            return [1.0] * len(embeddings)  # 返回最大距离
        
        logging.debug(f"有效嵌入向量数量: {len(valid_embeddings)}/{len(embeddings)}")
        
        # 计算有效嵌入向量的距离
        distances = np.ones(len(embeddings))  # 预设为最大距离

        if distance_metric == "cosine":
            query_embedding = np.array(query_embedding)
            query_embedding_norm = np.linalg.norm(query_embedding)
            
            for i, idx in enumerate(valid_indices):
                embedding = np.array(valid_embeddings[i])
                embedding_norm = np.linalg.norm(embedding)
                
                # 避免除零错误
                if query_embedding_norm == 0 or embedding_norm == 0:
                    distances[idx] = 1.0
                    continue
                
                # 点积除以模长的乘积，得到余弦相似度
                cosine_similarity = np.dot(query_embedding, embedding) / (query_embedding_norm * embedding_norm)
                
                # 余弦距离 = 1 - 余弦相似度
                distances[idx] = 1 - cosine_similarity

        elif distance_metric == "L1":
            for i, idx in enumerate(valid_indices):
                distances[idx] = np.sum(np.abs(np.array(query_embedding) - np.array(valid_embeddings[i])))

        elif distance_metric == "L2":
            for i, idx in enumerate(valid_indices):
                distances[idx] = np.sqrt(np.sum((np.array(query_embedding) - np.array(valid_embeddings[i])) ** 2))

        elif distance_metric == "Linf":
            for i, idx in enumerate(valid_indices):
                distances[idx] = np.max(np.abs(np.array(query_embedding) - np.array(valid_embeddings[i])))

        else:
            raise ValueError(f"不支持的距离度量方式: {distance_metric}")

        # 如果启用重排序且提供了必要的参数
        if rerank and query_text and node_texts and len(node_texts) > 0:
            rerank_scores = []
            try:
                # 如果没有提供reranker或tokenizer，则加载全局配置中的模型
                if reranker is None or reranker_tokenizer is None:
                    from utils.raptor_config_manager import RaptorConfigManager
                    config_manager = RaptorConfigManager()
                    reranker = config_manager.get_reranker()
                    reranker_tokenizer = config_manager.get_reranker_tokenizer()
                    logging.debug("从全局配置加载重排序模型")
                    
                logging.debug(f"开始对 {len(node_texts)} 个文本段进行重排序")
                
                # 创建查询-文档对
                pairs = [[query_text, text] for text in node_texts]
                
                # 使用分词器处理输入
                logging.debug("使用分词器处理查询-文档对")
                features = reranker_tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                    max_length=512
                )
                
                import torch
                # 使用模型预测得分
                logging.debug("使用重排序模型预测得分")
                with torch.no_grad():
                    scores = reranker(**features).logits.view(-1,).float()
                
                # 将PyTorch张量转换为Python列表
                rerank_scores = scores.tolist()
                
                # 记录重排序前后的排名变化
                original_ranks = list(range(len(distances)))
                sorted_by_distance = sorted(zip(original_ranks, distances), key=lambda x: x[1])
                sorted_by_rerank = sorted(zip(original_ranks, rerank_scores), key=lambda x: x[1], reverse=True)
                
                distance_ranks = [rank for rank, _ in sorted_by_distance]
                rerank_ranks = [rank for rank, _ in sorted_by_rerank]
                
                logging.debug(f"重排序完成。排名变化较大的项 (原始排名 -> 重排序后排名):")
                rank_changes = []
                for i, (orig_rank, new_rank) in enumerate(zip(distance_ranks[:5], rerank_ranks[:5])):
                    rank_change = distance_ranks.index(new_rank) - i
                    if abs(rank_change) > 0:
                        rank_changes.append(f"{distance_ranks.index(new_rank)+1} -> {i+1}")
                
                if rank_changes:
                    logging.debug(", ".join(rank_changes))
                    
                logging.debug(f"重排序完成，获得 {len(rerank_scores)} 个重排序分数")
                
                # 返回原始距离和重排序分数
                return distances.tolist(), rerank_scores
                
            except Exception as e:
                logging.error(f"重排序过程中发生错误: {str(e)}")
                # 发生错误时仅返回原始距离
                return distances.tolist()
        
        # 如果不使用重排序，仅返回原始距离
        return distances.tolist()
        
    except Exception as e:
        logging.error(f"计算距离时发生错误: {str(e)}")
        # 发生错误时返回全为1的距离列表（表示最大距离）
        return [1.0] * len(embeddings)


def get_node_list(node_dict: Dict[int, Node]) -> List[Node]:
    """
    Converts a dictionary of node indices to a sorted list of nodes.

    Args:
        node_dict (Dict[int, Node]): Dictionary of node indices to nodes.

    Returns:
        List[Node]: Sorted list of nodes.
    """
    indices = sorted(node_dict.keys())
    node_list = [node_dict[index] for index in indices]
    return node_list


def get_embeddings(node_list: List[Node], embedding_model: str) -> List:
    """
    Extracts the embeddings of nodes from a list of nodes.

    Args:
        node_list (List[Node]): List of nodes.
        embedding_model (str): The name of the embedding model to be used.

    Returns:
        List: List of node embeddings.
    """
    return [node.embeddings[embedding_model] for node in node_list]


def get_children(node_list: List[Node]) -> List[Set[int]]:
    """
    Extracts the children of nodes from a list of nodes.

    Args:
        node_list (List[Node]): List of nodes.

    Returns:
        List[Set[int]]: List of sets of node children indices.
    """
    return [node.children for node in node_list]


def get_text(node_list: List[Node]) -> str:
    """
    Generates a single text string by concatenating the text from a list of nodes.

    Args:
        node_list (List[Node]): List of nodes.

    Returns:
        str: Concatenated text.
    """
    text = ""
    for node in node_list:
        text += f"{' '.join(node.text.splitlines())}"
        text += "\n\n"
    return text


def indices_of_nearest_neighbors_from_distances(distances: List[float]) -> np.ndarray:
    """
    Returns the indices of nearest neighbors sorted in ascending order of distance.

    Args:
        distances (List[float]): A list of distances between embeddings.

    Returns:
        np.ndarray: An array of indices sorted by ascending distance.
    """
    return np.argsort(distances)
