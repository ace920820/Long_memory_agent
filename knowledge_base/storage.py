"""
存储模块 - 负责文档的存储、检索和树结构管理
"""
import json
import os
import sys
from typing import Dict, List, Optional
import logging
from datetime import datetime

# 添加项目根目录到Python路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from raptor import RetrievalAugmentation

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentStorage:
    """文档存储类 - 负责文档的存储和树结构管理"""
    
    def __init__(self, storage_dir: str = "data/RAtree/files", tree_save_path: str = "data/RAtree/default_tree"):
        """
        初始化文档存储类
        
        Args:
            storage_dir: 文档存储目录
            tree_save_path: RA树结构保存路径
        """
        self.storage_dir = storage_dir
        self.metadata_file = os.path.join(storage_dir, "metadata.json")
        self._ensure_storage_exists()
        self.metadata = self._load_metadata()
        
        # 初始化RA
        self.tree_save_path = tree_save_path
        os.makedirs(os.path.dirname(tree_save_path), exist_ok=True)
        self.RA = RetrievalAugmentation(tree=tree_save_path)
        logger.info(f"初始化文档存储，存储目录: {storage_dir}, RA保存路径: {tree_save_path}")
    
    def _ensure_storage_exists(self):
        """确保存储目录和元数据文件存在"""
        os.makedirs(self.storage_dir, exist_ok=True)
        if not os.path.exists(self.metadata_file):
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump({}, f, ensure_ascii=False, indent=2)
    
    def _load_metadata(self) -> Dict:
        """加载元数据"""
        try:
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载元数据失败: {str(e)}")
            return {}
    
    def _save_metadata(self):
        """保存元数据"""
        try:
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存元数据失败: {str(e)}")
    
    def add_document_in_metadata(self, doc_id: str, content: str) -> bool:
        """
        添加新文档，包括保存文件和元数据,不包含更新树结构
        该方法实现了文档存储的基础功能，将文档内容保存到文件系统，并更新元数据信息
        
        Args:
            doc_id: 文档ID，用于唯一标识文档
            content: 文档内容
            
        Returns:
            bool: 添加成功返回True，失败返回False
        """
        try:
            # 构建文档保存路径
            doc_path = os.path.join(self.storage_dir, f"{doc_id}.txt")
            logger.info(f"准备将文档 {doc_id} 保存到: {doc_path}")
            
            # 保存文档内容到文件
            with open(doc_path, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info(f"文档 {doc_id} 内容已保存到文件")
            
            # 更新元数据
            self.metadata[doc_id] = {
                "file_path": doc_path,
                "created_at": datetime.now().isoformat()
            }
            self._save_metadata()
            logger.info(f"文档 {doc_id} 的元数据已更新")
            
            return True
        except Exception as e:
            logger.error(f"添加文档 {doc_id} 失败: {str(e)}")
            return False

    def add_document_in_tree(self, doc_id: str, content: str) -> bool:
        """
        将文档内容添加到知识树结构中
        该方法负责更新运行时的树结构，但不会保存树结构到文件
        
        Args:
            doc_id: 文档ID，用于日志记录
            content: 要添加到树中的文档内容
            
        Returns:
            bool: 添加成功返回True，失败返回False
        """
        try:
            # 更新RA树结构
            self.RA.add_documents(content)
            logger.info(f"文档 {doc_id} 已添加到运行中的RA树结构")
            return True
        except Exception as e:
            logger.error(f"将文档 {doc_id} 添加到树结构失败: {str(e)}")
            return False

    def save_RA_tree(self, tree_save_path="data/default_tree"):
        """
        保存当前运行中的知识树结构到文件
        该方法将内存中的树结构持久化到指定路径
        
        Args:
            tree_save_path: 树结构的保存路径，默认为'data/default_tree'
            
        Returns:
            None
        """
        try:
            self.RA.save(tree_save_path)
            logger.info(f"已将运行中的RA树结构保存到: {tree_save_path}")
        except Exception as e:
            logger.error(f"保存RA树结构失败: {str(e)}")

    def get_tree_info_summary(self) -> Dict:
        """
        获取当前树的统计信息摘要
        该方法提供树结构的基本统计数据，包括层数、节点数等
        
        Returns:
            Dict: 包含以下字段的字典：
                - num_layers: 树的层数
                - total_nodes: 总节点数
                - leaf_nodes: 叶子节点数
                - summary_nodes: 摘要节点数
        """
        try:
            if not self.RA.tree:
                logger.warning("RA树未初始化，返回空统计信息")
                return {
                    "num_layers": 0,
                    "total_nodes": 0,
                    "leaf_nodes": 0,
                    "summary_nodes": 0
                }

            tree = self.RA.tree
            info = {
                "num_layers": tree.num_layers,
                "total_nodes": len(tree.all_nodes),
                "leaf_nodes": len(tree.leaf_nodes),
                "summary_nodes": len(tree.all_nodes) - len(tree.leaf_nodes)
            }
            logger.info(f"获取树统计信息成功: {info}")
            return info
        except Exception as e:
            logger.error(f"获取树统计信息失败: {str(e)}")
            return {}

    def get_tree_info(self) -> Optional[Dict]:
        """
        获取当前完整的树结构信息
        该方法返回树的详细信息，包括所有节点和它们之间的关系
        
        Returns:
            Optional[Dict]: 包含完整树结构的字典，如果获取失败返回None
                - tree: 包含节点和边信息的树结构数据
        """
        try:
            logger.info("开始获取完整树结构")
            if not self.RA.tree:
                logger.error("RA树未初始化")
                return None

            tree_data = self.RA.get_all_nodes_info()
            logger.info("树结构获取成功")
            return {
                "tree": tree_data
            }
        except Exception as e:
            logger.error(f"获取树结构失败: {str(e)}")
            return None

    def get_document(self, doc_id='') -> Optional[Dict]:
        """
        TODO 待开发完成
        获取文档内容和对应的树结构
        
        Args:
            doc_id: 文档ID
            
        Returns:
            Optional[Dict]: 文档数据，包含content和tree字段
        """
        try:
            if doc_id not in self.metadata:
                logger.warning(f"文档不存在: {doc_id}")
                return None

            doc_info = self.metadata[doc_id]

            # 读取文档内容
            with open(doc_info["file_path"], "r", encoding="utf-8") as f:
                content = f.read()

            # 获取树结构
            logger.info(f"从RA获取文档 {doc_id} 的树结构")
            if not self.RA.tree:
                logger.error("RA树未初始化")
                return None

            # TODO: 实现读取特定树文件结构的功能
            # 需要保存和管理多个树文件
            tree_data = ''
            
            return {
                "id": doc_id,
                "content": content,
                "created_at": doc_info["created_at"],
                "tree": tree_data
            }
        except Exception as e:
            logger.error(f"获取文档 {doc_id} 失败: {str(e)}")
            return None

    def delete_document(self, doc_id: str) -> bool:
        """
        删除文档及其相关数据
        该方法会删除文档文件、更新元数据，但目前不会更新树结构（待开发）
        
        Args:
            doc_id: 要删除的文档ID
            
        Returns:
            bool: 删除成功返回True，失败返回False
        """
        try:
            if doc_id not in self.metadata:
                logger.warning(f"要删除的文档不存在: {doc_id}")
                return False
            
            # 获取文档路径
            doc_path = self.metadata[doc_id]["file_path"]
            logger.info(f"准备删除文档文件: {doc_path}")
            
            # 删除文件
            if os.path.exists(doc_path):
                os.remove(doc_path)
                logger.info(f"文档文件已删除: {doc_path}")
            
            # 更新元数据
            del self.metadata[doc_id]
            self._save_metadata()
            logger.info(f"文档 {doc_id} 的元数据已删除")
            
            # TODO: 实现从树结构中删除文档的功能
            logger.warning("从树结构中删除文档的功能尚未实现")
            
            return True
        except Exception as e:
            logger.error(f"删除文档 {doc_id} 失败: {str(e)}")
            return False

    def list_documents(self) -> List[Dict]:
        """
        列出所有文档
        
        Returns:
            List[Dict]: 文档列表，每个文档包含id和created_at信息
        """
        try:
            documents = []
            for doc_id, info in self.metadata.items():
                documents.append({
                    "id": doc_id,
                    "created_at": info["created_at"]
                })
            return documents
        except Exception as e:
            logger.error(f"列出文档失败: {str(e)}")
            return []
    


if __name__ == "__main__":

    DocumentStorage0 = DocumentStorage()
    print("树载入成功")
    content="""智能长期记忆对话系统商业提案
"""
    DocumentStorage0.add_document_in_tree("",content)
    print("节点添加成功")
    print(DocumentStorage0.get_tree_info())
    print(DocumentStorage0.get_tree_info_summary())