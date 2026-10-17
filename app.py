"""
Campus Agent Entrypoint。

提供基礎本地聊天介面，並在使用者輸入送入 LLM 前執行前處理流程：

1. Regex 白名單 + 文長限制。
2. Unicode 正規化。
3. Regex 白名單 + 文長限制。
4. 語言檢測。
5. 提示詞注入防護。
6. 意圖路由。
"""

import os
import sys
import logging

import streamlit as st
from dotenv import load_dotenv
from fastmcp import Client
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from input_guard import check_allowed_chars, unicode_normalize
from language_guard import check_language_zh_en
from output_guard import convert_to_traditional
from prompt_guard import guard_against_jailbreak_fast, load_qwen3guard
from query_rewriter import rewrite_query
from rag.retriever import get_hybrid_retriever, retrieve_campus_info
from router import route_intent


def _setup_logging():
    qr_logger = logging.getLogger("query_rewriter")
    if not qr_logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
        )
        qr_logger.addHandler(handler)
    # INFO 會印出 SKIP / NO_CHANGE / REWRITTEN，DEBUG 會額外印出完整 prompt
    qr_logger.setLevel(logging.INFO)
    qr_logger.propagate = False


_setup_logging()

load_dotenv()

# =========================================================
# 頁面設定
# =========================================================

st.set_page_config(page_title="Campus Agent Entrypoint", page_icon="✨", layout="wide")

INPUT_MAX_LENGTH = 1024

# 提示詞注入防護開關，可在 .env 設定 PROMPT_GUARD_ENABLED=false 停用以節省載入時間
PROMPT_GUARD_ENABLED = os.getenv("PROMPT_GUARD_ENABLED", "true").strip().lower() in (
    "true",
    "1",
    "yes",
    "on",
)

# 提示詞注入防護策略：False 為保守拒絕，True 為模型異常時放行
GUARD_FAIL_OPEN = False


# =========================================================
# 資源初始化
# =========================================================


@st.cache_resource
def get_mcp_client():
    """
    初始化 MCP Client。

    目前僅保留 MCP 連線，作為後續路由層或工具呼叫的擴充入口。
    """
    return Client("mcp_server.py")


@st.cache_resource
def get_llm():
    """
    初始化本地 LLM。
    """
    return ChatOllama(model="qwen2.5:7b", temperature=0.7)


@st.cache_resource
def get_rewriter_llm():
    """
    Query Rewriting 使用低溫度，避免改寫結果漂移。
    """
    return ChatOllama(model="qwen2.5:7b", temperature=0.0)


@st.cache_resource
def get_prompt_guard():
    """
    載入提示詞注入防護模型。
    """
    return load_qwen3guard()


# =========================================================
# 側邊欄：系統狀態
# =========================================================

st.sidebar.title("📊 系統狀態")

try:
    _client = get_mcp_client()
    llm = get_llm()

    st.sidebar.success("MCP Server: 已連線")
    st.sidebar.success("本地 LLM (qwen2.5:7b): 就緒")
    st.sidebar.success("語言檢測: 就緒")

    with st.sidebar.spinner("載入校園知識庫檢索器..."):
        get_hybrid_retriever()
    st.sidebar.success("校園知識庫檢索器: 就緒")

    if PROMPT_GUARD_ENABLED:
        with st.sidebar.spinner("載入提示詞注入防護模型..."):
            get_prompt_guard()
        st.sidebar.success("提示詞注入防護: 就緒")
    else:
        st.sidebar.warning("提示詞注入防護: 已停用（.env）")

    st.sidebar.info("輸入前處理：白名單 → 正規化 → 白名單 → 語言檢測 → 注入防護")

except Exception as e:
    st.sidebar.error(f"系統連線失敗: {str(e)}")
    st.error("系統初始化失敗，請確認相關模型與服務是否可用。")
    st.stop()

# =========================================================
# 主介面
# =========================================================

st.title("Campus Agent Entrypoint")
st.markdown("""
本專案為基礎本地聊天 Agent 的輸入前處理與輸出後處理入口。  
目前聚焦於輸入進入模型前的檢查流程與輸出前的格式統一。
""")

if "messages" not in st.session_state:
    system_content = (
        "你是一位友善、專業且親切的台灣的星光大學校園助手。"
        "請一律使用繁體中文與台灣用語回答使用者的問題。"
    )
    st.session_state.messages = [
        {
            "role": "system",
            "content": system_content,
        }
    ]

for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input("請輸入您的問題..."):
    # 第一層：原始輸入白名單與文長檢查。
    if (
        check_allowed_chars(prompt, allow_empty=False, max_length=INPUT_MAX_LENGTH)
        == "refuse"
    ):
        st.warning(
            f"輸入格式不符合限制：長度不可超過 {INPUT_MAX_LENGTH} 字，或包含不允許的字元。"
        )
        st.stop()

    # 第二層：Unicode 正規化。
    normalized_prompt = unicode_normalize(prompt)

    # 第三層：正規化後再次執行白名單與文長檢查。
    if (
        check_allowed_chars(
            normalized_prompt, allow_empty=False, max_length=INPUT_MAX_LENGTH
        )
        == "refuse"
    ):
        st.warning("輸入經正規化後仍不符合格式限制，請重新輸入。")
        st.stop()

    prompt = normalized_prompt

    # 第四層：語言檢測。
    if check_language_zh_en(prompt) == "refuse":
        st.warning("輸入語言不符合限制，目前僅支援中文與英文。")
        st.stop()

    # 第五層：提示詞注入防護。
    if PROMPT_GUARD_ENABLED:
        with st.spinner("正在進行提示詞注入防護檢查..."):
            guard_decision = guard_against_jailbreak_fast(
                prompt, fail_open=GUARD_FAIL_OPEN, use_cache=True, max_new_tokens=24
            )

        if guard_decision == "refuse":
            st.warning("輸入未通過提示詞注入防護檢查，已拒絕處理。")
            st.stop()
    # 第六層：Query Rewriting。
    with st.spinner("正在進行指代還原..."):
        rewrite_result = rewrite_query(
            get_rewriter_llm(),
            prompt,
            st.session_state.messages,
        )

    rewritten_query = rewrite_result["query"]

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
            "rewritten": rewritten_query,
        }
    )
    with st.chat_message("user"):
        st.markdown(prompt)

        # 第七層：意圖路由，使用改寫後的查詢。
    with st.spinner("正在判斷意圖..."):
        routing_result = route_intent(rewritten_query)

    intent = routing_result["intent"]
    score = routing_result.get("score")
    reason = routing_result.get("reason")

    score_str = f"{score:.3f}" if score is not None else "N/A"

    # 根據意圖決定後續處理。
    with st.chat_message("assistant"):
        if intent == "campus_info":
            with st.spinner("正在檢索校園知識庫並生成回答..."):
                try:
                    retrieved_docs = retrieve_campus_info(rewritten_query)

                    if not retrieved_docs:
                        response_text = "抱歉，我目前在校園知識庫中找不到相關的資訊。"
                    else:
                        context = "\n\n".join(
                            [doc.page_content for doc in retrieved_docs]
                        )

                        system_content = st.session_state.messages[0]["content"]

                        rag_user_prompt = f"""<context>
{context}
</context>

<question>
{rewritten_query}
</question>

請根據 <context> 中的資訊回答 <question>。
若 <context> 中沒有足夠資訊，請誠實說明目前沒有相關資料，不要編造答案。"""

                        lc_messages = [
                            SystemMessage(content=system_content),
                            HumanMessage(content=rag_user_prompt),
                        ]

                        response = llm.invoke(lc_messages)
                        response_text = convert_to_traditional(response.content)

                    st.markdown(response_text)
                    st.caption(
                        f"(意圖: `{intent}`, 分數: {score_str} | 原因: {reason})"
                    )

                    if retrieved_docs:
                        with st.expander("🔍 查看檢索來源", expanded=False):
                            for i, doc in enumerate(retrieved_docs, 1):
                                st.markdown(f"**片段 {i}**")
                                st.text(doc.page_content)

                    st.session_state.messages.append(
                        {"role": "assistant", "content": response_text}
                    )

                except Exception as e:
                    error_msg = f"校園資訊檢索時發生錯誤: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg}
                    )
        else:
            # chitchat 或 fallback
            with st.spinner("AI 正在思考中..."):
                try:
                    lc_messages = []
                    for msg in st.session_state.messages:
                        if msg["role"] == "system":
                            lc_messages.append(SystemMessage(content=msg["content"]))
                        elif msg["role"] == "user":
                            content = msg.get("rewritten") or msg["content"]
                            lc_messages.append(HumanMessage(content=content))
                        elif msg["role"] == "assistant":
                            lc_messages.append(AIMessage(content=msg["content"]))

                    response = llm.invoke(lc_messages)
                    final_answer = response.content

                    final_answer = convert_to_traditional(final_answer)

                    st.markdown(final_answer)
                    st.caption(
                        f"(意圖: `{intent}`, 分數: {score_str} | 原因: {reason})"
                    )

                    st.session_state.messages.append(
                        {"role": "assistant", "content": final_answer}
                    )

                except Exception as e:
                    error_msg = f"處理問題時發生錯誤: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg}
                    )
