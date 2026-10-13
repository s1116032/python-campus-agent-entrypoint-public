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

import streamlit as st
from fastmcp import Client
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from input_guard import check_allowed_chars, unicode_normalize
from language_guard import check_language_zh_en
from prompt_guard import load_qwen3guard, guard_against_jailbreak_fast
from router import route_intent

# =========================================================
# 頁面設定
# =========================================================

st.set_page_config(page_title="Campus Agent Entrypoint", page_icon="✨", layout="wide")

# 輸入長度上限。
INPUT_MAX_LENGTH = 1024

# 提示詞注入防護策略。
# False: 模型異常或無法判斷時，拒絕輸入。
# True: 模型異常或無法判斷時，放行輸入。
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
    # 保留 MCP 連線；目前未呼叫工具。
    _client = get_mcp_client()
    llm = get_llm()

    st.sidebar.success("MCP Server: 已連線")
    st.sidebar.success("本地 LLM (qwen2.5:7b): 就緒")
    st.sidebar.success("語言檢測: 就緒")

    with st.sidebar.spinner("載入提示詞注入防護模型..."):
        get_prompt_guard()

    st.sidebar.success("提示詞注入防護: 就緒")
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
本專案為基礎本地聊天 Agent 的輸入前處理入口。  
目前不處理 RAG 或工具呼叫，主要聚焦於輸入進入模型前的檢查流程與意圖路由。
""")

# 初始化對話歷史。
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": (
                "你是一位友善、專業且親切的星光大學校園助手。"
                "請一律使用繁體中文回答使用者的問題。"
            ),
        }
    ]

# 顯示歷史對話，過濾 system message。
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# 接收使用者輸入。
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
    with st.spinner("正在進行提示詞注入防護檢查..."):
        guard_decision = guard_against_jailbreak_fast(
            prompt, fail_open=GUARD_FAIL_OPEN, use_cache=True, max_new_tokens=24
        )

    if guard_decision == "refuse":
        st.warning("輸入未通過提示詞注入防護檢查，已拒絕處理。")
        st.stop()

    # 第六層：意圖路由。
    with st.spinner("正在判斷意圖..."):
        routing_result = route_intent(prompt)

    intent = routing_result["intent"]
    score = routing_result.get("score")
    reason = routing_result.get("reason")

    # 通過前處理與路由後，加入對話歷史。
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 根據意圖決定後續處理。
    with st.chat_message("assistant"):
        if intent == "campus_info":
            response_text = "🚧 校園資訊問答功能 (RAG) 施工中，敬請期待！"
            st.markdown(response_text)
            st.caption(f"(意圖: `{intent}`, 分數: {score:.3f} | 原因: {reason})")
            st.session_state.messages.append(
                {"role": "assistant", "content": response_text}
            )
        else:
            # chitchat 或 fallback，送入 LLM。
            with st.spinner("AI 正在思考中..."):
                try:
                    lc_messages = []
                    for msg in st.session_state.messages:
                        if msg["role"] == "system":
                            lc_messages.append(SystemMessage(content=msg["content"]))
                        elif msg["role"] == "user":
                            lc_messages.append(HumanMessage(content=msg["content"]))
                        elif msg["role"] == "assistant":
                            lc_messages.append(AIMessage(content=msg["content"]))

                    response = llm.invoke(lc_messages)
                    final_answer = response.content

                    st.markdown(final_answer)
                    st.caption(
                        f"(意圖: `{intent}`, 分數: {score:.3f} | 原因: {reason})"
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
