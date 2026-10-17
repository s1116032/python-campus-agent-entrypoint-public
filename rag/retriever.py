"""
混合檢索模組（Hybrid Retriever）。

結合 BM25（稀疏/關鍵字檢索）與 bge-m3（稠密/語義檢索），
並使用 LangChain EnsembleRetriever 內建的 RRF (Reciprocal Rank Fusion) 融合。
"""

from typing import List, Optional

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from rag.document_loader import load_campus_faq

import jieba
from opencc import OpenCC


TOP_K = 4
EMBEDDING_MODEL = "bge-m3"
RRF_WEIGHTS = [0.5, 0.5]

_hybrid_retriever: Optional[EnsembleRetriever] = None

_t2s_converter = OpenCC("t2s")


def chinese_tokenize(text: str) -> list:
    """
    中文分詞，供 BM25 使用。

    BM25 預設以空白切詞，但中文沒有空白，整句會變成單一 token。
    因此先轉簡體（jieba 詞典以簡體為主，分詞較準確），再用 jieba 切詞。
    索引與查詢使用同一函數，確保匹配一致。
    """
    simplified = _t2s_converter.convert(text)
    return [token for token in jieba.cut(simplified) if token.strip()]


def build_hybrid_retriever() -> EnsembleRetriever:
    """
    建立混合檢索器（BM25 + bge-m3 向量檢索，以 RRF 融合）。
    """
    documents = load_campus_faq()

    bm25_retriever = BM25Retriever.from_documents(
        documents,
        preprocess_func=chinese_tokenize,
    )
    bm25_retriever.k = TOP_K

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    chroma_vectorstore = Chroma.from_documents(
        documents,
        embeddings,
        collection_name="campus_faq",
    )
    dense_retriever = chroma_vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=RRF_WEIGHTS,
    )

    return ensemble_retriever


def get_hybrid_retriever() -> EnsembleRetriever:
    """
    取得混合檢索器（單例快取，僅首次呼叫時建立索引）。
    """
    global _hybrid_retriever
    if _hybrid_retriever is None:
        _hybrid_retriever = build_hybrid_retriever()
    return _hybrid_retriever


def retrieve_campus_info(query: str, top_k: int = TOP_K) -> List[Document]:
    """
    根據查詢檢索校園知識庫。

    Args:
        query: 使用者查詢（建議傳入改寫後的查詢）。
        top_k: 回傳的文件數量。

    Returns:
        List[Document]: 檢索到的相關文件。
    """
    retriever = get_hybrid_retriever()
    results = retriever.invoke(query)
    return results[:top_k]
