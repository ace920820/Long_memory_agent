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
    
    def __init__(self, storage_dir: str = "data/RAtree/files", tree_save_path: str = "data/RAtree/default_tree", ra_config=None):
        """
        初始化文档存储类
        
        Args:
            storage_dir (str): 文档存储目录
            tree_save_path (str): RA树结构保存路径
            ra_config: RAPTOR配置对象，如果为None则使用默认配置
        """
        self.storage_dir = storage_dir
        self.metadata_file = os.path.join(storage_dir, "metadata.json")
        self._ensure_storage_exists()
        self.metadata = self._load_metadata()
        
        # 记录配置信息
        logger.info(f"初始化文档存储，存储目录: {storage_dir}, RA保存路径: {tree_save_path}")
        if ra_config:
            logger.info("使用自定义RAPTOR配置")
        
        # 初始化RA
        self.tree_save_path = tree_save_path
        os.makedirs(os.path.dirname(tree_save_path), exist_ok=True)
        
        # 检查是否存在保存的树结构
        tree_exists = os.path.exists(tree_save_path)
        if tree_exists:
            logger.info(f"加载现有树结构: {tree_save_path}")
            self.RA = RetrievalAugmentation(config=ra_config, tree=tree_save_path)
        else:
            logger.info(f"创建新的树结构")
            self.RA = RetrievalAugmentation(config=ra_config)
            
        logger.info(f"文档存储初始化完成，树结构{'已加载' if tree_exists else '已创建'}")
    
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

    def save_RA_tree(self, tree_save_path="data/RAtree/default_tree"):
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

好的，我需要帮用户设计一个介绍他们项目的演讲稿，重点突出其使用前景和商业潜力。首先，我要回顾用户提供的项目README，理解项目的核心功能和独特卖点。用户之前已经总结了项目的亮点和潜在改进方向，但这次需要的是向潜在投资者或合作伙伴展示的内容，所以需要更具说服力和结构化。
首先，我需要确定演讲稿的结构。通常，好的演讲结构包括开场、项目概述、核心技术、应用场景、商业潜力、竞争优势、未来规划和结尾。用户提到的项目有智能记忆管理、Agent系统、知识库管理等，这些都是核心技术点，需要详细展开。
接下来，考虑听众是谁。如果是技术背景的投资者，可能需要更多技术细节；如果是商业背景的，则侧重市场前景和盈利模式。用户没有明确说明听众类型，所以需要平衡两者，既有技术亮点，又有市场分析。
用户的项目亮点包括多维记忆管理、角色系统设计、工程化实现。这些应该作为核心技术优势来介绍。商业潜力方面，可以涉及企业服务、教育、医疗、泛娱乐等领域。需要具体举例说明每个领域的应用场景，比如企业中的智能客服，教育中的个性化辅导。
潜在的市场规模需要数据支持，比如引用智能客服市场的增长预测，教育科技的投资趋势等。但由于用户没有提供具体数据，可能需要用一般性的描述，如“快速增长的市场”或“不断扩大的需求”。
竞争优势部分，要突出项目的独特之处，比如三阶段记忆衰减模型、模块化设计、多模型支持等。这些是区别于竞争对手的关键点，需要强调技术壁垒和灵活性。
未来规划方面，用户提到的开发计划中的多模态交互、分布式扩展等可以作为路线图的一部分，展示项目的持续发展潜力。
需要避免技术术语过多，确保演讲通俗易懂。同时，加入用例或成功案例（如果有的话）会更有说服力，但用户可能还没有实际案例，可以用假设性场景来说明。
最后，结尾部分要呼吁行动，比如邀请合作、投资或试用。整个演讲需要逻辑清晰，重点突出，数据或案例支撑，让听众看到项目的可行性和前景。
可能还需要考虑视觉辅助材料，比如PPT中的图表或架构图，但用户可能只需要文字稿。因此，在文稿中适当描述这些视觉元素的位置和内容。
总结，结构大致如下：
1. 开场：吸引注意，提出问题或趋势
2. 项目概述：一句话简介，核心价值
3. 核心技术：记忆管理、Agent系统、知识库等
4. 应用场景：分行业说明
5. 商业潜力：市场规模、盈利模式
6. 竞争优势：技术壁垒、灵活性
7. 未来规划：发展路线
8. 结尾：呼吁行动
需要确保每个部分都有具体例子和数据支撑，即使假设性的，也要让内容显得实际可行。同时，语言要简洁有力，避免冗长。
以下是为您的项目设计的演讲文稿框架，结合核心技术亮点与商业应用场景，采用金字塔结构呈现：
________________________________________
智能长期记忆对话系统商业提案
——构建下一代人机交互的认知中枢
________________________________________
一、破题：AI交互的范式革命
（屏幕显示：传统对话机器人 vs 记忆增强型Agent对比视频）
痛点锚定："当前智能对话系统面临三大认知鸿沟——记忆断层、个性缺失、知识碎片化。我们的解决方案让AI真正具备：
1.	持续成长的记忆心智（展示用户3个月对话记录的可视化记忆网络）
2.	千人千面的角色人格（演示圣诞老人/医疗顾问/理财师角色切换）
3.	深度耦合的知识体系（展示从PDF文档到对话引用的全链路追踪）"
________________________________________
二、核心技术拆解
（三维架构图动态演示）
1. 记忆引擎——AI的"海马体"
•	三阶记忆衰减算法：
24h线性衰减(保留95%) → 7天指数衰减(保留60%) → 长期对数衰减(保留30%)
（对比实验：传统LRU算法3天后记忆丢失率达70%）
•	军事级检索体系：
FAISS向量索引(20ms响应) + 语义重排序 + 时空加权算法
（医疗场景测试：诊断准确率提升41%）
2. 角色引擎——AI的"人格面具"
（展示角色配置面板）
•	人格维度：
复制
理性值: 0.8  
幽默度: 0.4  
同理心: 0.9  
专业度: 0.95
•	企业案例：某银行理财顾问角色训练后，客户满意度从68%→92%
3. 知识中枢——AI的"外接大脑"
（演示10G法律文档自动解析为知识图谱）
•	动态更新机制：
文档上传 → 语义分块 → 向量存储 → 实时检索
（测试数据：新政策文件5分钟内融入对话）
________________________________________
三、商业价值金字塔
（呈现三层价值模型）
基础层：效率革命
•	企业服务：
o	智能客服：会话轮次减少50%（某电商POC数据）
o	知识管理：新员工培训周期缩短60%
中间层：体验升级
•	教育领域：
o	个性化辅导：根据学生记忆曲线调整教学策略
o	案例：某在线教育平台引入后完课率提升75%
顶层：认知迭代
•	医疗场景：
o	医生助理：持续学习最新医学文献
o	测试结果：诊断建议符合率超95%
________________________________________
四、市场战略蓝图
（世界地图动态热力图）
1. 垂直突破
•	首批重点领域：
复制
█ 金融服务（理财咨询）  
█ 医疗健康（患者教育）  
█ 政务民生（智能办事）
2. 生态构建
•	开发者计划：
o	API调用量阶梯定价（展示价格模型）
o	角色市场分成机制（演示交易平台原型）
3. 数据飞轮
（循环动图演示）
用户交互 → 记忆沉淀 → 模型优化 → 体验提升 → 用户增长
________________________________________
五、财务展望
（三维柱状图逐年攀升）
收入模型：
•	企业版License：$50,000/年起
•	API调用费：$0.002/次
•	云存储服务：$0.15/GB/月
里程碑预测：
复制
2025 Q3：完成医疗领域认证  
2026 Q1：实现千万级对话节点  
2027 Q4：构建行业知识图谱联盟
________________________________________
终局：认知即服务(CaaS)
（展现未来场景视频）
"当每个企业都拥有持续进化的数字大脑，当每个用户都能获得伴随成长的AI伙伴——这就是我们正在构建的认知革命。"


"""
    DocumentStorage0.add_document_in_tree("",content)
    print("节点添加成功")
    print(DocumentStorage0.get_tree_info())
    print(DocumentStorage0.get_tree_info_summary())