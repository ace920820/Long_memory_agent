import spacy
import logging
from typing import List, Dict, Optional

class EntityExtractor:
    """
    实体提取器，使用 spaCy 进行命名实体识别（NER）
    
    支持多种语言的实体识别，可以提取文本中的命名实体及其类型
    """
    
    def __init__(self, language: str = 'zh'):
        """
        初始化 EntityExtractor
        
        :param language: 语言模型，默认为中文 'zh'
        """
        try:
            # 根据语言选择合适的 spaCy 模型
            model_map = {
                'zh': 'zh_core_web_sm',  # 中文模型
                'en': 'en_core_web_sm',  # 英文模型
            }
            
            model_name = model_map.get(language, 'zh_core_web_sm')
            
            try:
                # 尝试加载 spaCy 模型
                self.nlp = spacy.load(model_name)
                logging.info(f"成功加载 spaCy {model_name} 模型")
            except OSError:
                # 如果模型未安装，提供详细的安装指导
                install_command = f"python -m spacy download {model_name}"
                error_msg = (
                    f"未找到 {model_name} 模型。请使用以下命令安装：\n"
                    f"    {install_command}\n"
                    "或者使用默认的中文模型 'zh_core_web_sm'"
                )
                logging.error(error_msg)
                raise ValueError(error_msg)
        
        except Exception as e:
            logging.error(f"初始化 EntityExtractor 时发生错误: {e}")
            raise

    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        """
        从给定文本中提取实体
        
        :param text: 待提取实体的文本
        :return: 实体列表，每个实体包含名称、类型、起始和结束位置
        """
        if not text:
            logging.warning("输入文本为空")
            return []

        try:
            # 处理文本
            doc = self.nlp(text)
            
            # 提取实体
            entities = [
                {
                    "name": ent.text,           # 实体名称
                    "type": ent.label_,         # 实体类型
                    "start_pos": ent.start_char,# 起始位置
                    "end_pos": ent.end_char     # 结束位置
                }
                for ent in doc.ents
            ]
            
            logging.info(f"从文本中提取到 {len(entities)} 个实体")
            return entities
        
        except Exception as e:
            logging.error(f"实体提取过程中发生错误: {e}")
            return []

    def get_entity_types(self) -> List[str]:
        """
        获取当前模型支持的实体类型
        
        :return: 支持的实体类型列表
        """
        try:
            return list(self.nlp.get_pipe("ner").labels)
        except Exception as e:
            logging.error(f"获取实体类型时发生错误: {e}")
            return []

def main():
    """
    用于测试 EntityExtractor 的主函数
    """
    try:
        extractor = EntityExtractor(language='zh')
        
        # 测试文本
        test_texts = [
            "李明是北京大学的教授，他在人工智能领域有着深入的研究。",
            "2025年，阿里巴巴在杭州举办了一次重要的技术峰会。"
        ]
        
        for text in test_texts:
            print(f"\n测试文本: {text}")
            entities = extractor.extract_entities(text)
            print("提取的实体:")
            for entity in entities:
                print(f"- 名称: {entity['name']}, 类型: {entity['type']}, 位置: [{entity['start_pos']}, {entity['end_pos']}]")
    
    except Exception as e:
        print(f"测试过程中发生错误: {e}")

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main()
