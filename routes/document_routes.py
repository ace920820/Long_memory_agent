from flask import Blueprint, request, jsonify
import os
from werkzeug.utils import secure_filename
import logging
from typing import List

# 创建蓝图
document_bp = Blueprint('document', __name__)

# 允许的文件类型
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'md'}

def allowed_file(filename: str) -> bool:
    """检查文件类型是否允许"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@document_bp.route('/api/documents', methods=['GET'])
def get_documents():
    """获取所有文档列表"""
    try:
        documents = document_bp.rag_module.get_documents()
        return jsonify({
            "success": True,
            "documents": documents
        })
    except Exception as e:
        logging.error(f"Error getting documents: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@document_bp.route('/api/documents', methods=['POST'])
def upload_document():
    """上传新文档"""
    try:
        if 'files' not in request.files:
            return jsonify({
                "success": False,
                "error": "No file part"
            }), 400
            
        file = request.files['files']
        if file.filename == '':
            return jsonify({
                "success": False,
                "error": "No selected file"
            }), 400
            
        if not allowed_file(file.filename):
            return jsonify({
                "success": False,
                "error": f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
            }), 400
            
        # 保存文件到临时目录
        filename = file.filename
        temp_path = os.path.join('temp', filename)
        os.makedirs('temp', exist_ok=True)
        file.save(temp_path)
        
        # 添加文件到知识库
        result = document_bp.rag_module.add_file(temp_path, filename)
        
        # 删除临时文件
        os.remove(temp_path)
        
        if result["success"]:
            return jsonify(result)
        else:
            return jsonify(result), 400
            
    except Exception as e:
        logging.error(f"Error uploading document: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@document_bp.route('/api/documents/<int:doc_id>', methods=['DELETE'])
def delete_document(doc_id: int):
    """删除文档"""
    try:
        result = document_bp.rag_module.delete_document(doc_id)
        if result["success"]:
            return jsonify(result)
        else:
            return jsonify(result), 404
    except Exception as e:
        logging.error(f"Error deleting document: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

def init_document_routes(rag_module):
    """初始化文档路由"""
    document_bp.rag_module = rag_module
    return document_bp
