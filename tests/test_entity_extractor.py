import pytest
import logging
import spacy

from models.entity_extractor import EntityExtractor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestEntityExtractor:
    """
    EntityExtractor 的单元测试类
    
    测试目标：
    1. 测试实体提取功能
    2. 验证不同语言模型的支持
    3. 处理边界情况
    4. 验证实体类型
    """
    
    @pytest.fixture
    def zh_extractor(self):
        """
        创建中文实体提取器的测试夹具
        """
        return EntityExtractor(language='zh')
    
    @pytest.fixture
    def en_extractor(self):
        """
        创建英文实体提取器的测试夹具
        """
        try:
            return EntityExtractor(language='en')
        except ValueError:
            pytest.skip("英文模型未安装，跳过英文相关测试")
    
    def test_zh_entity_extraction(self, zh_extractor):
        """
        测试中文实体提取
        """
        test_texts = [
            "李明是北京大学的教授，他在人工智能领域有着深入的研究。",
            "阿里巴巴在杭州举办了一次重要的技术峰会。",
            "2025年，中国科学技术大学的张院士获得了重要奖项。"
        ]
        
        for text in test_texts:
            entities = zh_extractor.extract_entities(text)
            logger.info(f"中文文本 '{text}' 提取实体: {entities}")
            
            assert len(entities) > 0, f"未能从文本 '{text}' 中提取实体"
            
            for entity in entities:
                assert all(key in entity for key in ['name', 'type', 'start_pos', 'end_pos']), \
                    f"实体 {entity} 缺少必要的键"
    
    def test_en_entity_extraction(self, en_extractor):
        """
        测试英文实体提取
        """
        test_texts = [
            "Apple Inc. is headquartered in Cupertino, California.",
            "Elon Musk is the CEO of Tesla and SpaceX.",
            "The United Nations is located in New York City."
        ]
        
        for text in test_texts:
            entities = en_extractor.extract_entities(text)
            logger.info(f"英文文本 '{text}' 提取实体: {entities}")
            
            assert len(entities) > 0, f"未能从文本 '{text}' 中提取实体"
            
            for entity in entities:
                assert all(key in entity for key in ['name', 'type', 'start_pos', 'end_pos']), \
                    f"实体 {entity} 缺少必要的键"
    
    def test_empty_text(self, zh_extractor):
        """
        测试空文本处理
        """
        empty_texts = ['', '   ', None]
        
        for text in empty_texts:
            entities = zh_extractor.extract_entities(text)
            assert len(entities) == 0, f"空文本 '{text}' 不应提取实体"
    
    def test_entity_types(self, zh_extractor):
        """
        测试实体类型获取
        """
        zh_types = zh_extractor.get_entity_types()
        
        logger.info(f"中文模型实体类型: {zh_types}")
        
        assert len(zh_types) > 0, "中文模型未找到实体类型"
    
    def test_position_accuracy(self, zh_extractor):
        """
        测试实体位置准确性
        """
        text = "李明是北京大学的教授，他在人工智能领域有着深入的研究。"
        entities = zh_extractor.extract_entities(text)
        
        for entity in entities:
            # 验证实体位置
            extracted_entity = text[entity['start_pos']:entity['end_pos']]
            assert extracted_entity == entity['name'], \
                f"位置不准确：预期 '{entity['name']}'，实际 '{extracted_entity}'"
    
    @pytest.mark.parametrize("language", ['zh'])
    def test_invalid_language(self, language):
        """
        测试语言模型处理
        """
        try:
            extractor = EntityExtractor(language=language)
            assert extractor is not None, f"未能创建 {language} 语言的实体提取器"
        except Exception as e:
            pytest.fail(f"创建 {language} 语言的实体提取器时发生异常: {e}")

def main():
    """
    主函数，用于手动测试
    """
    pytest.main([__file__])

if __name__ == "__main__":
    main()
