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
        logging.info(f"安全化后的文件名: {filename}")  # 添加日志记录文件名

        # 检查安全化后的文件名是否包含扩展名
        if '.' not in filename:
            # 从原始文件名获取扩展名
            original_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
            if original_ext and original_ext in ALLOWED_EXTENSIONS:
                # 添加原始扩展名
                filename = f"{filename}.{original_ext}"
                logging.info(f"添加扩展名后的文件名: {filename}")
        
        temp_path = os.path.join('temp', filename)
        os.makedirs('temp', exist_ok=True)
        file.save(temp_path)
        
        logging.info(f"文件已保存到临时路径: {temp_path}")
        
        try:
            # 根据文件类型处理文档内容
            if '.' in filename:
                file_ext = filename.rsplit('.', 1)[1].lower()
            else:
                # 如果文件名中没有扩展名，尝试从原始文件名获取
                file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
                if not file_ext:
                    return jsonify({
                        "success": False,
                        "error": "无法确定文件类型"
                    }), 400
            
            logging.info(f"检测到文件类型: {file_ext}")
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

@document_bp.route('/api/test_pdf_extract', methods=['POST'])
def test_pdf_extract():
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

@document_bp.route('/test_pdf_extract', methods=['POST'])
def test_pdf_extract_web():
    """
    用于测试PDF文本提取功能的网页接口
    接收PDF文件上传并返回提取的文本
    """
    # 记录网页端PDF提取请求的开始
    logging.info("收到网页端PDF文本提取测试请求")
    
    # 获取上传的文件
    if 'file' not in request.files:
        logging.error("未接收到文件")
        return jsonify({'error': '未接收到文件'}), 400
    
    file = request.files['file']
    
    # 检查文件名是否为空
    if file.filename == '':
        logging.error("未选择文件")
        return jsonify({'error': '未选择文件'}), 400
    
    # 检查文件是否为PDF
    if not file.filename.endswith('.pdf'):
        logging.error(f"不支持的文件类型: {file.filename}")
        return jsonify({'error': '仅支持PDF文件'}), 400
    
    try:
        # 创建临时文件
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, file.filename)
        
        logging.info(f"保存上传的PDF文件到临时路径: {temp_file_path}")
        file.save(temp_file_path)
        
        # 提取PDF文本
        logging.info(f"开始提取PDF文本: {file.filename}")
        extracted_text = extract_text_from_pdf(temp_file_path)
        
        # 计算文本长度
        text_length = len(extracted_text)
        logging.info(f"PDF文本提取成功，共 {text_length} 个字符")
        
        # 提取文本预览（最多1000个字符）
        preview_length = min(1000, text_length)
        text_preview = extracted_text[:preview_length]
        
        # 删除临时文件
        os.remove(temp_file_path)
        logging.info(f"已删除临时文件: {temp_file_path}")
        
        return jsonify({
            'success': True,
            'text_length': text_length,
            'preview': text_preview
        })
    
    except Exception as e:
        logging.error(f"PDF处理错误: {str(e)}", exc_info=True)
        return jsonify({'error': f'PDF处理错误: {str(e)}'}), 500

@document_bp.route('/pdf_test')
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
