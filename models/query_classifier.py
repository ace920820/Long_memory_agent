"""
查询分类器模块

该模块提供了用于解析和分类用户查询的功能，以确定合适的检索策略（综合检索或精准检索）。
综合检索适用于需要全面了解某个主题的查询，而精准检索适用于需要具体细节的查询。
"""
from typing import Set, Dict, List
import jieba
import re


class QueryClassifier:
    """
    查询分类器类：用于分类用户查询以确定合适的检索策略。
    主要功能：
    1. 对用户输入的查询进行分类，判断是需要综合检索还是精准检索
    2. 支持动态添加关键词，实现分类策略的灵活调整
    """
    
    def __init__(self):
        """
        初始化查询分类器，预定义不同检索类型的关键词。
        """
        # 综合召回关键词：用于识别需要全面信息的查询
        self.comprehensive_keywords: Set[str] = {
            "综合", "总结", "概述", "概览", "报告", "简介",
            "整体", "全局", "总体", "汇总", "归纳", "概要",
            "总的", "大致", "大概", "大体", "大致上"
        }
        
        # 精准召回关键词：用于识别需要具体细节的查询
        self.precise_keywords: Set[str] = {
            "具体", "详细", "精确", "准确", "详情", "细节",
            "准确地", "精准地", "具体地", "详细地"
        }
        
        # 将所有关键词添加到jieba分词器的用户词典中
        for word in self.comprehensive_keywords | self.precise_keywords:
            jieba.add_word(word)
    
    def _preprocess_query(self, query: str) -> str:
        """
        预处理查询文本，移除不必要的字符并标准化文本。
        
        参数:
            query: 输入的查询字符串
            
        返回:
            预处理后的查询字符串
        """
        # 移除标点符号和特殊字符
        query = re.sub(r'[^\w\s]', '', query)
        # 转换为小写（对中文无影响，主要用于处理混合查询）
        query = query.lower()
        return query
    
    def _extract_keywords(self, query: str) -> List[str]:
        """
        使用jieba分词器从查询中提取关键词。
        
        参数:
            query: 预处理后的查询字符串
            
        返回:
            提取出的关键词列表
        """
        return list(jieba.cut(query))
    
    def classify_query(self, query: str) -> str:
        """
        对查询进行分类，判断是需要综合检索还是精准检索。
        
        参数:
            query: 用户输入的查询字符串
            
        返回:
            "COMPREHENSIVE" 表示需要综合检索
            "PRECISE" 表示需要精准检索
        """
        # 对查询进行预处理
        processed_query = self._preprocess_query(query)
        
        # 提取查询中的关键词
        keywords = self._extract_keywords(processed_query)
        
        # 计算查询中包含的综合检索和精准检索关键词的数量
        comprehensive_score = sum(1 for word in keywords if word in self.comprehensive_keywords)
        precise_score = sum(1 for word in keywords if word in self.precise_keywords)
        
        # 根据关键词匹配分数决定检索类型
        if comprehensive_score > precise_score:
            return "COMPREHENSIVE"
        elif precise_score > comprehensive_score:
            return "PRECISE"
        else:
            # 当无法明确判断时，默认使用精准检索
            return "PRECISE"
    
    def add_comprehensive_keywords(self, keywords: Set[str]) -> None:
        """
        添加新的综合检索关键词。
        
        参数:
            keywords: 要添加的新关键词集合
        """
        self.comprehensive_keywords.update(keywords)
        for word in keywords:
            jieba.add_word(word)
    
    def add_precise_keywords(self, keywords: Set[str]) -> None:
        """
        添加新的精准检索关键词。
        
        参数:
            keywords: 要添加的新关键词集合
        """
        self.precise_keywords.update(keywords)
        for word in keywords:
            jieba.add_word(word)
