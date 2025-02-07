"""
API模块 - 提供文档管理的HTTP接口
"""
import os
from typing import Dict, List, Optional
import logging
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from pydantic import BaseModel

from .storage import DocumentStorage
from .upload import FileUploader

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 初始化FastAPI应用
app = FastAPI(title="知识库API")

# 初始化文档存储和文件上传器
storage = DocumentStorage()
uploader = FileUploader()

class NodeUpdate(BaseModel):
    """节点更新请求模型"""
    text: str

@app.post("/documents/upload", response_model=Dict[str, str])
async def upload_document(file: UploadFile = File(...)):
    """
    上传文档
    
    Args:
        file: 文件对象
        
    Returns:
        Dict[str, str]: 包含文档ID的响应
        
    Raises:
        HTTPException: 上传失败时抛出
    """
    try:
        # 上传文件
        file_id = await uploader.upload_file(file)
        if not file_id:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="文件上传失败"
            )
        
        # 读取文件内容
        content = uploader.read_file_content(file_id)
        if not content:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="读取文件内容失败"
            )
        
        # 保存文档
        if storage.add_document(file_id, content):
            logger.info(f"文档上传成功: {file_id}")
            return {"id": file_id}
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="保存文档失败"
        )
    except Exception as e:
        logger.error(f"上传文档失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/documents/{doc_id}")
async def get_document(doc_id: str):
    """
    获取文档
    
    Args:
        doc_id: 文档ID
        
    Returns:
        Dict: 文档信息
        
    Raises:
        HTTPException: 文档不存在或获取失败时抛出
    """
    doc = storage.get_document(doc_id)
    if not doc:
        logger.warning(f"文档不存在: {doc_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="文档不存在"
        )
    return doc

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """
    删除文档
    
    Args:
        doc_id: 文档ID
        
    Returns:
        Dict: 操作结果
        
    Raises:
        HTTPException: 删除失败时抛出
    """
    if storage.delete_document(doc_id):
        return {"message": "删除成功"}
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="删除文档失败"
    )

@app.put("/documents/{doc_id}/nodes/{node_id}")
async def update_document_node(doc_id: str, node_id: str, update: NodeUpdate):
    """
    更新文档节点
    
    Args:
        doc_id: 文档ID
        node_id: 节点ID
        update: 更新内容
        
    Returns:
        Dict: 操作结果
        
    Raises:
        HTTPException: 更新失败时抛出
    """
    if storage.update_document_node(doc_id, node_id, update.text):
        return {"message": "更新成功"}
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="更新节点失败"
    )

@app.get("/documents")
async def list_documents():
    """
    列出所有文档
    
    Returns:
        List[Dict]: 文档列表
    """
    return storage.list_documents()

@app.get("/tree/info")
async def get_tree_info():
    """
    获取树结构信息
    
    Returns:
        Dict: 树的统计信息
    """
    return storage.get_tree_info()
