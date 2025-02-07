"""
知识库管理系统
提供文档上传、解析、存储和管理功能
"""

from .api import app
from .storage import DocumentStorage
from .upload import FileUploader

__all__ = ['app', 'DocumentManager', 'DocumentStorage', 'FileUploader']
