"""
知識庫文件載入與切分模組。

讀取單一 Markdown 知識庫，並切分為適合檢索的文本塊（Chunks）。
使用 Markdown 標題結構作為優先分隔符，保持每個主題的語意完整性。
"""

import os
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAQ_FILE_PATH = os.path.join(_PROJECT_ROOT, "knowledge_base", "campus_faq.md")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def load_campus_faq() -> List[Document]:
    """
    讀取校園常見問題知識庫，並切分為文本塊。

    Returns:
        List[Document]: 切分後的文件塊列表。
    """
    if not os.path.exists(FAQ_FILE_PATH):
        raise FileNotFoundError(f"找不到知識庫檔案: {FAQ_FILE_PATH}")

    with open(FAQ_FILE_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", "。", "；", "，", " "],
    )

    chunks = splitter.create_documents([content])
    return chunks
