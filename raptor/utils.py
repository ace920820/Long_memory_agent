import logging
import re
from typing import Dict, List, Set

import numpy as np
import tiktoken
from scipy import spatial

from .tree_structures import Node

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


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
    delimiters = [".", "!", "?", "\n"]
    regex_pattern = "|".join(map(re.escape, delimiters))
    sentences = re.split(regex_pattern, text)
    overlap = 20

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
    logging.info(f"开始递归切分文本，文本长度：{len(text)}，最大token数：{max_tokens}，重叠token数：{overlap}")
    
    # 定义递归切分函数
    def _recursive_split(text_segment, delimiters_idx=0):
        # 定义分隔符列表，按优先级排序
        delimiters = [
            [".", "!", "?", "\n"],  # 主要分隔符（句子边界）
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
            logging.info(f"使用分隔符 {current_delimiters} 无法有效切分文本，尝试下一级分隔符")
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
                logging.info(f"单个段落token数 ({segment_tokens}) 超过最大限制 ({max_tokens})，递归切分此段落")
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
    logging.info(f"递归切分完成，共生成 {len(chunks)} 个文本块")
    return chunks


def distances_from_embeddings(
    query_embedding: List[float],
    embeddings: List[List[float]],
    distance_metric: str = "cosine",
) -> List[float]:
    """
    Calculates the distances between a query embedding and a list of embeddings.

    Args:
        query_embedding (List[float]): The query embedding.
        embeddings (List[List[float]]): A list of embeddings to compare against the query embedding.
        distance_metric (str, optional): The distance metric to use for calculation. Defaults to 'cosine'.

    Returns:
        List[float]: The calculated distances between the query embedding and the list of embeddings.
    """
    distance_metrics = {
        "cosine": spatial.distance.cosine,
        "L1": spatial.distance.cityblock,
        "L2": spatial.distance.euclidean,
        "Linf": spatial.distance.chebyshev,
    }

    if distance_metric not in distance_metrics:
        raise ValueError(
            f"Unsupported distance metric '{distance_metric}'. Supported metrics are: {list(distance_metrics.keys())}"
        )

    distances = [
        distance_metrics[distance_metric](query_embedding, embedding)
        for embedding in embeddings
    ]

    return distances


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
