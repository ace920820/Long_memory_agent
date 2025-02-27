from flask import Blueprint, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename
import logging
from typing import List
import fitz  # 导入PyMuPDF库的fitz模块
import tempfile

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
            # 根据文件类型处理文档内容
            file_ext = filename.rsplit('.', 1)[1].lower()
            content = ""
            
            if file_ext == 'pdf':
                # 使用PyMuPDF处理PDF文件
                logging.info(f"检测到PDF文件，开始使用PyMuPDF处理: {filename}")
                content = extract_text_from_pdf(temp_path)
                logging.info(f"PDF文件处理完成，提取文本长度: {len(content)}")
            else:
                # 处理文本文件
                try:
                    with open(temp_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                except UnicodeDecodeError:
                    # 尝试其他编码
                    with open(temp_path, 'r', encoding='latin-1') as f:
                        content = f.read()
            
            logging.info(f"成功读取文件内容，长度: {len(content)}")
            
            if not content.strip():
                return jsonify({
                    "success": False,
                    "error": "文件内容为空或无法提取有效文本"
                }), 400
            
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

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    从PDF文件中提取文本内容
    
    Args:
        pdf_path: PDF文件路径
        
    Returns:
        str: 提取的文本内容
    """
    try:
        # 记录开始处理PDF文件
        logging.info(f"开始从PDF文件提取文本: {pdf_path}")
        
        # 打开PDF文件
        pdf_document = fitz.open(pdf_path)
        text_content = []
        
        # 处理每一页并提取文本
        for page_num in range(len(pdf_document)):
            # 获取当前页
            page = pdf_document[page_num]
            # 提取文本
            text = page.get_text()
            text_content.append(text)
            logging.info(f"已处理第 {page_num+1}/{len(pdf_document)} 页，提取文本长度: {len(text)}")
        
        # 合并所有页面的文本
        full_text = "\n\n".join(text_content)
        logging.info(f"PDF文本提取完成，总长度: {len(full_text)}")
        
        return full_text
    except Exception as e:
        error_msg = f"PDF文本提取失败: {str(e)}"
        logging.error(error_msg)
        raise Exception(error_msg)

@document_bp.route('/api/test_pdf_extract', methods=['POST'], endpoint='pdf_extract_test')
def pdf_extract_test():
    """测试PDF文本提取功能"""
    try:
        # 记录开始处理上传请求
        logging.info("开始处理PDF测试请求")
        
        if 'file' not in request.files:
            return jsonify({
                "success": False,
                "error": "No file part"
            }), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                "success": False,
                "error": "No selected file"
            }), 400
            
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({
                "success": False,
                "error": "Only PDF files are allowed for this test"
            }), 400
            
        # 保存文件到临时目录
        filename = secure_filename(file.filename)
        temp_path = os.path.join('temp', filename)
        os.makedirs('temp', exist_ok=True)
        file.save(temp_path)
        
        logging.info(f"PDF文件已保存到临时路径: {temp_path}")
        
        try:
            # 使用PyMuPDF处理PDF文件
            content = extract_text_from_pdf(temp_path)
            
            # 提取前1000个字符作为预览
            preview = content[:1000] + "..." if len(content) > 1000 else content
            
            return jsonify({
                "success": True,
                "message": "PDF文本提取成功",
                "text_length": len(content),
                "preview": preview
            })
                
        finally:
            # 删除临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)
                logging.info(f"临时文件已删除: {temp_path}")
            
    except Exception as e:
        error_msg = f"处理PDF测试时发生错误: {str(e)}"
        logging.error(error_msg)
        return jsonify({
            "success": False,
            "error": error_msg
        }), 500

@document_bp.route('/pdf_test', endpoint='pdf_test_page_view')
def pdf_test_page():
    """
    渲染PDF测试页面
    """
    logging.info("访问PDF测试页面")
    return render_template('pdf_test.html')

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
