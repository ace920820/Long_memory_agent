import unittest
import os
import shutil
import tempfile
import logging
from models.rag_module import RAGModule

# 配置日志输出
logging.basicConfig(level=logging.DEBUG,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class TestRAGModule(unittest.TestCase):
    """RAG模块的单元测试类"""
    
    @classmethod
    def setUpClass(cls):
        """在所有测试用例执行前运行一次，初始化模型"""
        # 创建临时目录用于测试
        cls.test_dir = tempfile.mkdtemp()
        cls.index_path = os.path.join(cls.test_dir, 'test_index')
        os.makedirs(cls.index_path, exist_ok=True)
        
        # 创建测试配置文件
        cls.test_config_path = os.path.join(cls.test_dir, 'test_config.yaml')
        
        # 使用正斜杠替换反斜杠
        safe_path = cls.index_path.replace('\\', '/')
        
        config_content = f"""embedding:
  model_path: "BAAI/bge-small-zh"
  similarity_threshold: 0.5
  index_path: "{safe_path}"

rerank:
  model_path: "BAAI/bge-small-zh"
  batch_size: 32
  max_length: 512"""
        
        # 写入测试配置
        with open(cls.test_config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        # 初始化RAG模块（只初始化一次）
        cls.rag = RAGModule(config_path=cls.test_config_path)
        
        # 准备测试数据
        cls.test_docs = [
            {
                'content': '人工智能是计算机科学的一个重要分支',
                'metadata': {'source': 'test1', 'category': 'AI'}
            },
            {
                'content': '机器学习是人工智能的核心技术之一',
                'metadata': {'source': 'test2', 'category': 'ML'}
            },
            {
                'content': '深度学习是机器学习的一个重要方向',
                'metadata': {'source': 'test3', 'category': 'DL'}
            }
        ]

    def setUp(self):
        """每个测试用例执行前的设置"""
        # 清空文档列表
        self.rag.documents = []
        self.rag.document_embeddings = None
        self.rag.hnsw_index = None
        self.rag.exact_index = None
        self.rag.chunk_to_doc = {}

    @classmethod
    def tearDownClass(cls):
        """在所有测试用例执行后清理资源"""
        shutil.rmtree(cls.test_dir)

    def test_add_documents(self):
        """测试添加文档功能"""
        # 添加文档
        self.rag.add_documents(self.test_docs)
        
        # 验证文档数量
        self.assertEqual(len(self.rag.documents), len(self.test_docs))
        
        # 验证文档内容
        for i, doc in enumerate(self.test_docs):
            self.assertEqual(self.rag.documents[i]['content'], doc['content'])
            self.assertEqual(self.rag.documents[i]['metadata'], doc['metadata'])
    
    def test_search(self):
        """测试搜索功能"""
        # 添加文档
        self.rag.add_documents(self.test_docs)
        
        # 执行搜索
        query = "什么是人工智能"
        results = self.rag.search(query)
        
        # 验证搜索结果
        self.assertTrue(len(results) > 0)
        self.assertIn('content', results[0])
        self.assertIn('metadata', results[0])
        self.assertIn('similarity_score', results[0])
        self.assertGreaterEqual(results[0]['similarity_score'], 0.0)
        self.assertLessEqual(results[0]['similarity_score'], 1.0)
    
    def test_index_persistence(self):
        """测试索引持久化功能"""
        # 添加文档
        self.rag.add_documents(self.test_docs)
        
        # 保存索引
        self.rag._save_index()
        
        # 创建新的RAG实例并加载索引
        new_rag = RAGModule(config_path=self.test_config_path)
        
        # 验证文档数量
        self.assertEqual(len(new_rag.documents), len(self.test_docs))
        
        # 验证搜索功能
        query = "什么是人工智能"
        results = new_rag.search(query)
        self.assertTrue(len(results) > 0)

if __name__ == '__main__':
    unittest.main()
