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
    """上传新文档并添加到RA树结构中"""
    try:
        # 记录开始处理上传请求
        logging.info("开始处理文档上传请求")
        
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
        filename = secure_filename(file.filename)
        temp_path = os.path.join('temp', filename)
        os.makedirs('temp', exist_ok=True)
        file.save(temp_path)
        
        logging.info(f"文件已保存到临时路径: {temp_path}")
        
        try:
            # 读取文件内容
            with open(temp_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            logging.info(f"成功读取文件内容，长度: {len(content)}")
            
            # 使用add_documents方法添加到RA树
            result = document_bp.rag_module.add_documents_no_saving(content)
            
            if result["status"] == "success":
                logging.info(f"文档 {filename} 已成功添加到RA树")
                return jsonify({
                    "success": True,
                    "message": f"文档 {filename} 上传并添加到知识库成功",
                    "filename": filename
                })
            else:
                logging.error(f"添加文档到RA树失败: {result['message']}")
                return jsonify({
                    "success": False,
                    "error": result["message"]
                }), 400
                
        finally:
            # 删除临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)
                logging.info(f"临时文件已删除: {temp_path}")
            
    except Exception as e:
        error_msg = f"处理文档上传时发生错误: {str(e)}"
        logging.error(error_msg)
        return jsonify({
            "success": False,
            "error": error_msg
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

@document_bp.route('/api/documents/content/<path:file_name>', methods=['GET'])
def get_document_content(file_name):
    """获取文档内容"""
    try:
        # 查找文档
        for doc in document_bp.rag_module.documents:
            if doc['file_name'] == file_name:
                return jsonify({
                    "success": True,
                    "content": doc.get('content', ''),
                    "chunks": doc.get('chunks', [])
                })
        
        return jsonify({
            "success": False,
            "error": "Document not found"
        }), 404
        
    except Exception as e:
        logging.error(f"Error getting document content: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

def init_document_routes(rag_module):
    """初始化文档路由"""
    document_bp.rag_module = rag_module
    return document_bp
