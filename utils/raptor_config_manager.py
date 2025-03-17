"""
RAPTOR配置管理器模块
负责加载和管理RAPTOR相关配置，提供嵌入模型选择等功能
"""
import os
import yaml
import logging
from raptor import RetrievalAugmentationConfig
from raptor.EmbeddingModels import SBertEmbeddingModel, OpenAIEmbeddingModel

# 配置日志
logger = logging.getLogger(__name__)

class RaptorConfigManager:
    """RAPTOR配置管理器类，负责加载和管理RAPTOR相关配置"""
    
    def __init__(self, config_path="config/raptor_config.yaml"):
        """
        初始化配置管理器
        
        Args:
            config_path (str): RAPTOR配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config()
        logger.info(f"RAPTOR配置管理器初始化，配置文件: {config_path}")
    
    def _load_config(self):
        """
        加载配置文件
        
        Returns:
            dict: 配置字典
        """
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f).get("raptor", {})
            else:
                logger.warning(f"RAPTOR配置文件不存在: {self.config_path}，使用默认配置")
                return {}
        except Exception as e:
            logger.error(f"加载RAPTOR配置失败: {str(e)}")
            return {}
    
    def get_embedding_model(self):
        """
        根据配置获取嵌入模型实例
        
        Returns:
            BaseEmbeddingModel: 嵌入模型实例
        """
        # 获取模型类型
        model_type = self.config.get("embedding", {}).get("model_type", "local")
        
        try:
            if model_type == "local":
                # 使用本地嵌入模型
                model_config = self.config.get("embedding", {}).get("local", {})
                model_path = model_config.get("model_path")
                model_name = model_config.get("model_name", model_path)
                
                if not model_path:
                    raise ValueError("本地嵌入模型路径未配置")
                
                logger.info(f"使用本地嵌入模型: {model_path}")
                return SBertEmbeddingModel(model_name=model_path)
            else:
                # 使用OpenAI嵌入模型
                model_config = self.config.get("embedding", {}).get("openai", {})
                model = model_config.get("model", "text-embedding-ada-002")
                base_url = model_config.get("base_url")
                
                # 检查环境变量
                if not os.environ.get("OPENAI_API_KEY"):
                    logger.warning("未设置OPENAI_API_KEY环境变量，可能会导致API调用失败")
                
                logger.info(f"使用OpenAI嵌入模型: {model}")
                if base_url:
                    return OpenAIEmbeddingModel(model=model)
                else:
                    return OpenAIEmbeddingModel(model=model)
        except Exception as e:
            logger.error(f"创建嵌入模型失败: {str(e)}，将使用默认OpenAI模型")
            return OpenAIEmbeddingModel()
    
    def get_ra_config(self):
        """
        获取RetrievalAugmentation配置实例
        
        Returns:
            RetrievalAugmentationConfig: 检索增强配置实例
        """
        # 获取配置参数
        chunk_size = self.config.get("chunk_size", 300)
        overlap = self.config.get("overlap", 20)
        num_layers = self.config.get("num_layers", 5)
        threshold = self.config.get("threshold", 0.5)
        top_k = self.config.get("top_k", 5)
        selection_mode = self.config.get("selection_mode", "top_k")
        
        # 获取嵌入模型
        embedding_model = self.get_embedding_model()
        
        logger.info(f"创建RAPTOR配置: chunk_size={chunk_size}, overlap={overlap}, num_layers={num_layers}")
        
        # 创建配置对象
        return RetrievalAugmentationConfig(
            embedding_model=embedding_model,
            # 树构建器配置
            tb_max_tokens=chunk_size,
            tb_num_layers=num_layers,
            tb_threshold=threshold,
            tb_top_k=top_k,
            tb_selection_mode=selection_mode,
            # 树检索器配置
            tr_threshold=threshold,
            tr_top_k=top_k,
            tr_selection_mode=selection_mode
        )
