"""
上传模块 - 处理文件上传和临时存储
"""
import os
import uuid
import logging
from typing import Optional
from fastapi import UploadFile

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FileUploader:
    """文件上传处理类"""
    
    def __init__(self, upload_dir: str = "data/files"):
        """
        初始化文件上传器
        
        Args:
            upload_dir: 上传文件存储目录
        """
        self.upload_dir = upload_dir
        os.makedirs(upload_dir, exist_ok=True)
        logger.info(f"初始化文件上传器，上传目录: {upload_dir}")
    
    async def upload_file(self, file: UploadFile) -> Optional[str]:
        """
        上传文件并返回文件ID
        
        Args:
            file: 上传的文件对象
            
        Returns:
            Optional[str]: 文件ID，上传失败则返回None
        """
        try:
            # 生成唯一文件ID
            file_id = str(uuid.uuid4())
            file_path = os.path.join(self.upload_dir, f"{file_id}.txt")
            
            # 保存文件
            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)
            
            logger.info(f"文件上传成功: {file_id}")
            return file_id
        except Exception as e:
            logger.error(f"文件上传失败: {str(e)}")
            return None
    
    def read_file_content(self, file_id: str) -> Optional[str]:
        """
        读取已上传文件的内容
        
        Args:
            file_id: 文件ID
            
        Returns:
            Optional[str]: 文件内容，读取失败则返回None
        """
        try:
            file_path = os.path.join(self.upload_dir, f"{file_id}.txt")
            if not os.path.exists(file_path):
                logger.warning(f"文件不存在: {file_id}")
                return None
            
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            logger.info(f"读取文件成功: {file_id}")
            return content
        except Exception as e:
            logger.error(f"读取文件失败: {str(e)}")
            return None
