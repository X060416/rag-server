"""
Document Processor
Uses markitdown to convert documents to markdown, then LangChain to split into chunks.
"""

import os
import logging
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from markitdown import MarkItDown
from rapidocr_onnxruntime import RapidOCR
import re
import ftfy


from config import CHUNK_SIZE, CHUNK_OVERLAP, SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)

# Initialize markitdown
md_converter = MarkItDown()


def is_supported(file_path: str) -> bool:
    """Check if the file extension is supported."""
    ext = os.path.splitext(file_path)[1].lower()
    return ext in SUPPORTED_EXTENSIONS


def convert_to_markdown(file_path: str) -> str:
    """
    Convert a document to markdown using markitdown.
    Supports PDF, DOCX, PPTX, XLSX, HTML, and many other formats.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if not is_supported(file_path):
        raise ValueError(f"Unsupported file type: {file_path}")

    logger.info(f"Converting {file_path} to markdown...")
    result = md_converter.convert(file_path) #核心转换
    markdown_text = result.text_content

    if not markdown_text or not markdown_text.strip():
        raise ValueError(f"No content extracted from: {file_path}")

    logger.info(f"Extracted {len(markdown_text)} characters from {file_path}")
    return markdown_text

def clean_text(text: str) -> str:
    """
    清洗 markitdown 提取出的文本，针对 PDF 和 Word 的常见噪声。
    保留 Markdown 结构（标题、列表、表格语法不被破坏）。
    """
    if not text or not text.strip():
        return text

    # 修复编码乱码（ftfy）
    text = ftfy.fix_text(text)

    #去除不可见的控制字符
    # \x00-\x08, \x0b, \x0c, \x0e-\x1f 是 ASCII 控制字符
    # 保留 \t（制表符，Markdown 表格需要）和 \n（换行）
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)

    #去除页码
    text = re.sub(r'\n\d{1,4}\s*\n', '\n', text)

    #压缩连续空行
    # 3 个以上换行压缩为 2 个（Markdown 段落分隔就是 \n\n）
    text = re.sub(r'\n{3,}', '\n\n', text)

    #去除每行行尾多余空格
    # PDF 提取经常每行末尾有多余空格
    text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)

    # 修复 PDF 断词连字符 
    # PDF 换行时会把一个单词从中间断开加连字符：
    #   comput-\ner → computer
    #   资源管-\n理 → 资源管理
    # 只修复纯英文断词（字母+连字符+换行+字母），中文不动
    text = re.sub(r'([a-zA-Z])-\n([a-zA-Z])', r'\1\2', text)

    # 去除多余空格（但保留 Markdown 缩进）
    # 连续 3 个以上空格压缩为 1 个，但不影响行首的 Markdown 缩进
    text = re.sub(r'(?<=\S) {3,}(?=\S)', ' ', text)

    return text.strip()


def split_text(markdown_text: str, source_name: str = "") -> List[dict]:
    """
    Split markdown text into chunks.
    Uses MarkdownHeaderTextSplitter first to preserve document structure,
    then RecursiveCharacterTextSplitter for fine-grained chunking.

    Returns a list of {"text": str, "metadata": dict} dicts.
    """
    chunks = []

    # Step 1: Try markdown header splitting to preserve structure 按照Markdown标题层级切分
    try:
        md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "header_1"),
                ("##", "header_2"),
                ("###", "header_3"),
            ]
        )
        md_chunks = md_splitter.split_text(markdown_text)
    except Exception as e:
        logger.warning(f"Markdown header splitting failed: {e}, falling back to recursive only")
        from langchain_core.documents import Document
        md_chunks = [Document(page_content=markdown_text, metadata={})]

    # Step 2: Recursive character splitting for each markdown section 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )

    for md_chunk in md_chunks:
        sub_chunks = text_splitter.split_text(md_chunk.page_content)

        for i, chunk_text in enumerate(sub_chunks):
            header_info = " > ".join(
                v for v in md_chunk.metadata.values() if v
            )

            # 把文件名和章节名注入到 chunk 文本中参与向量检索
            enriched_text = chunk_text
            if header_info:
                enriched_text = f"Section: {header_info}\n\n{enriched_text}"
            if source_name:
                enriched_text = f"Source: {source_name}\n\n{enriched_text}"

            metadata = {
                "source": source_name,
                "section": header_info or "root",
                "chunk_index": i,
            }
            chunks.append({
                "text": enriched_text,
                "metadata": metadata,
            })

    logger.info(f"Split into {len(chunks)} chunks (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


def process_document(file_path: str, source_name: str = "") -> List[dict]:
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext in {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"}:
        # 图片走 OCR
        ocr_text = ocr_image(file_path)
        if not ocr_text.strip():
            raise ValueError("OCR 未识别到文字")
        ocr_text = clean_text(ocr_text)
        chunks = split_text(ocr_text, source_name)
    else:
        # 其他文档走 markitdown
        markdown_text = convert_to_markdown(file_path)
        markdown_text = clean_text(markdown_text)
        chunks = split_text(markdown_text, source_name)
    
    return chunks


ocr_engine = None  # 懒加载，避免每次启动都加载模型

def get_ocr():
    global ocr_engine
    if ocr_engine is None:
        ocr_engine = RapidOCR()
    return ocr_engine

def ocr_image(file_path: str) -> str:
    """用 RapidOCR 从图片提取文字"""
    result, elapse = get_ocr()(file_path)
    if result is None:
        return ""
    lines = []
    for line in result:
        lines.append(line[1])  # line[1] 是文字
    return "\n".join(lines)
