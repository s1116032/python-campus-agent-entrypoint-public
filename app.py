"""
Campus Agent Entrypoint。

提供本地聊天介面，並在使用者輸入送入 LLM 前執行前處理流程：

1. Regex 白名單 + 文長限制
2. Unicode 正規化
3. Regex 白名單 + 文長限制
4. 語言檢測
5. 提示詞注入防護
6. Query Rewriting（指代還原）
7. 意圖路由
8. 輸出後處理（簡轉繁與慣用語）

另提供「預約討論室」功能（新增／查看／修改／取消），
與一般聊天共用同一套輸入前處理。
"""

import logging
import os
import sys
from typing import Optional

import streamlit as st
from dotenv import load_dotenv
from fastmcp import Client
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from booking.db import (
    get_reservation,
    init_db,
    list_available_rooms,
    list_user_reservations,
)
from booking.extractor import (
    FIELD_LABELS,
    empty_draft,
    extract_booking_fields,
    extract_manage_intent,
    missing_fields,
)
from booking.service import (
    cancel_booking,
    get_current_user_id,
    modify_booking,
    submit_booking,
)
from input_guard import check_allowed_chars, unicode_normalize
from language_guard import check_language_zh_en
from output_guard import convert_to_traditional
from prompt_guard import guard_against_jailbreak_fast, load_qwen3guard
from query_rewriter import rewrite_query
from rag.retriever import get_hybrid_retriever, retrieve_campus_info
from router import route_intent


# =========================================================
# 日誌與環境變數
# =========================================================


def _setup_logging():
    for name in ("query_rewriter", "booking.extractor"):
        target = logging.getLogger(name)
        if not target.handlers:
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
                )
            )
            target.addHandler(handler)
        target.setLevel(logging.INFO)
        target.propagate = False


_setup_logging()
load_dotenv()


# =========================================================
# 頁面設定與全域參數
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
    """初始化 MCP Client，作為後續路由層或工具呼叫的擴充入口。"""
    return Client("mcp_server.py")


@st.cache_resource
def get_llm():
    """初始化本地 LLM。"""
    return ChatOllama(model="qwen2.5:7b", temperature=0.7)


@st.cache_resource
def get_rewriter_llm():
    """Query Rewriting 使用低溫度，避免改寫結果漂移。"""
    return ChatOllama(model="qwen2.5:7b", temperature=0.0)


@st.cache_resource
def get_prompt_guard():
    """載入提示詞注入防護模型。"""
    return load_qwen3guard()


@st.cache_resource
def get_booking_db():
    """初始化預約討論室資料庫。"""
    init_db()
    return True


# =========================================================
# 共用輸入前處理
# =========================================================


def preprocess_user_input(text: str) -> Optional[str]:
    """
    一般聊天與預約對話共用的輸入前處理。

    白名單 → 正規化 → 白名單 → 語言檢測 → 提示詞注入防護。
    兩個入口共用同一套檢查，避免安全落差。
    """
    if (
        check_allowed_chars(text, allow_empty=False, max_length=INPUT_MAX_LENGTH)
        == "refuse"
    ):
        st.warning(
            f"輸入格式不符合限制：長度不可超過 {INPUT_MAX_LENGTH} 字，或包含不允許的字元。"
        )
        st.stop()

    normalized = unicode_normalize(text)

    if (
        check_allowed_chars(normalized, allow_empty=False, max_length=INPUT_MAX_LENGTH)
        == "refuse"
    ):
        st.warning("輸入經正規化後仍不符合格式限制，請重新輸入。")
        st.stop()

    if check_language_zh_en(normalized) == "refuse":
        st.warning("輸入語言不符合限制，目前僅支援中文與英文。")
        st.stop()

    if PROMPT_GUARD_ENABLED:
        with st.spinner("正在進行提示詞注入防護檢查..."):
            decision = guard_against_jailbreak_fast(
                normalized, fail_open=GUARD_FAIL_OPEN, use_cache=True, max_new_tokens=24
            )
        if decision == "refuse":
            st.warning("輸入未通過提示詞注入防護檢查，已拒絕處理。")
            st.stop()

    return normalized


# =========================================================
# 預約討論室：狀態管理
# =========================================================


def _start_create():
    st.session_state.booking_stage = "collect"
    st.session_state.booking_draft = empty_draft()
    st.session_state.booking_result = None
    st.session_state.booking_messages = [
        {
            "role": "assistant",
            "content": (
                "您好，我們來預約討論室。\n\n"
                "請告訴我：**討論室編號、人數、日期與起訖時間、事由**。\n"
                "可以一次說完，例如：「明天下午兩點到三點，B01，五個人，專題報告討論。」"
            ),
        }
    ]


def _start_manage():
    st.session_state.booking_stage = "manage_list"
    st.session_state.manage_messages = []
    st.session_state.pending_modify = None
    st.session_state.pending_cancel = None


def _exit_booking():
    st.session_state.booking_stage = None
    st.session_state.booking_draft = None
    st.session_state.booking_result = None
    st.session_state.booking_messages = []
    st.session_state.manage_messages = []
    st.session_state.pending_modify = None
    st.session_state.pending_cancel = None


# =========================================================
# 預約討論室：新增流程
# =========================================================


def render_booking_chat():
    """對話式收集：自由輸入，欄位不齊時補問。"""
    st.subheader("📅 預約討論室")
    st.caption("此對話僅用於預約討論室，不會影響一般聊天紀錄。")

    for message in st.session_state.get("booking_messages", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("請說明討論室、人數、時間與事由...")
    if not user_input:
        return

    normalized = preprocess_user_input(user_input)

    # 預約流程不做 Query Rewriting 與意圖路由：
    # 這裡是專屬工作流，上下文由提取層自行處理。
    st.session_state.booking_messages.append({"role": "user", "content": normalized})

    valid_room_ids = [room["room_id"] for room in list_available_rooms()]

    with st.spinner("正在整理預約資訊..."):
        result = extract_booking_fields(
            get_rewriter_llm(),
            normalized,
            st.session_state.booking_draft,
            valid_room_ids,
        )

    if not result["success"]:
        st.session_state.booking_messages.append(
            {
                "role": "assistant",
                "content": "我沒有順利理解剛才的說明，請再說一次。",
            }
        )
        st.rerun()

    updated = result["draft"]
    if updated.get("reason"):
        updated["reason"] = convert_to_traditional(updated["reason"])
    st.session_state.booking_draft = updated

    missing = missing_fields(updated)
    if missing:
        labels = "、".join(FIELD_LABELS[key] for key in missing)
        st.session_state.booking_messages.append(
            {"role": "assistant", "content": f"還缺 {labels}，請補充。"}
        )
        st.rerun()

    st.session_state.booking_stage = "confirm"
    st.rerun()


def render_booking_confirmation():
    """確認卡片：只接受按鈕操作，不開放自由對話。"""
    st.subheader("📅 預約確認")

    draft = st.session_state.get("booking_draft")
    if not draft or missing_fields(draft):
        st.session_state.booking_stage = "collect"
        st.rerun()

    st.markdown(f"**人數**：{draft['people_count']} 人")
    st.markdown(f"**時間**：{draft['start_time']} ～ {draft['end_time']}")
    st.markdown(f"**討論室**：{draft['room_id']}")
    st.markdown(f"**事由**：{draft['reason']}")
    st.caption("如需修改，請選擇「我要調整」後用對話修改。")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("✅ 確認送出", type="primary", use_container_width=True):
            result = submit_booking(draft)
            if result["success"]:
                st.session_state.booking_result = {
                    "message": result["message"],
                    "summary": (
                        f"- 討論室：{draft['room_id']}\n"
                        f"- 人數：{draft['people_count']} 人\n"
                        f"- 時間：{draft['start_time']} ～ {draft['end_time']}\n"
                        f"- 事由：{draft['reason']}"
                    ),
                }
                st.session_state.booking_stage = "done"
                st.session_state.booking_draft = None
            else:
                # 保留草稿，讓使用者可以修正後再送出
                st.session_state.booking_messages.append(
                    {
                        "role": "assistant",
                        "content": f"無法送出：{result['message']}\n\n請修正後再試一次。",
                    }
                )
                st.session_state.booking_stage = "collect"
            st.rerun()

    with col2:
        if st.button("✏️ 我要調整", use_container_width=True):
            st.session_state.booking_stage = "collect"
            st.session_state.booking_messages.append(
                {
                    "role": "assistant",
                    "content": "請告訴我要修改的欄位與新內容，例如「人數改成 4 人」。",
                }
            )
            st.rerun()

    with col3:
        if st.button("❌ 取消", use_container_width=True):
            _exit_booking()
            st.rerun()


def render_booking_done():
    """預約完成：罐頭訊息，確保編號與時間準確。"""
    st.subheader("📅 預約完成")

    result = st.session_state.get("booking_result")
    if not result:
        _exit_booking()
        st.rerun()

    st.success(result["message"])
    st.markdown(result["summary"])

    if st.button("返回聊天"):
        _exit_booking()
        st.rerun()


# =========================================================
# 預約討論室：管理流程
# =========================================================


def _append_manage_message(content: str, role: str = "assistant"):
    st.session_state.manage_messages.append({"role": role, "content": content})


def _render_reservation_table(reservations):
    rows = []
    for r in reservations:
        rows.append(
            {
                "編號": f"#{r['id']}",
                "討論室": r["room_id"],
                "時間": f"{r['start_time']} ～ {r['end_time']}",
                "人數": f"{r['people_count']} 人",
                "事由": r["reason"],
            }
        )
    st.table(rows)


def _handle_manage_input(user_input: str, reservations):
    normalized = preprocess_user_input(user_input)
    _append_manage_message(normalized, role="user")

    available_ids = [r["id"] for r in reservations]

    with st.spinner("正在理解您的需求..."):
        result = extract_manage_intent(get_rewriter_llm(), normalized, available_ids)

    if not result["success"]:
        _append_manage_message("我沒有順利理解，請再說一次。")
        st.rerun()

    intent = result["intent"]
    reservation_id = result["reservation_id"]

    if intent == "cancel":
        if reservation_id is None:
            _append_manage_message("請告訴我要取消哪一筆，例如「取消 #2」。")
            st.rerun()
        st.session_state.pending_cancel = {"reservation_id": reservation_id}
        st.session_state.booking_stage = "manage_cancel_confirm"
        st.rerun()

    elif intent == "modify":
        if reservation_id is None:
            _append_manage_message(
                "請告訴我要修改哪一筆，例如「把 #1 的時間改成下午三點」。"
            )
            st.rerun()

        target = get_reservation(reservation_id)
        target_draft = {
            "room_id": target["room_id"],
            "people_count": target["people_count"],
            "start_time": target["start_time"],
            "end_time": target["end_time"],
            "reason": target["reason"],
        }
        valid_room_ids = [room["room_id"] for room in list_available_rooms()]

        with st.spinner("正在整理修改內容..."):
            extract_result = extract_booking_fields(
                get_rewriter_llm(), normalized, target_draft, valid_room_ids
            )

        if not extract_result["success"] or not extract_result["changed"]:
            _append_manage_message(
                "我沒有辨識到要修改的內容，請再說一次，例如「把 #1 的時間改成下午三點」。"
            )
            st.rerun()

        new_fields = extract_result["draft"]
        if new_fields.get("reason"):
            new_fields["reason"] = convert_to_traditional(new_fields["reason"])

        st.session_state.pending_modify = {
            "reservation_id": reservation_id,
            "old_fields": target_draft,
            "new_fields": new_fields,
        }
        st.session_state.booking_stage = "manage_modify_confirm"
        st.rerun()

    else:
        _append_manage_message("請告訴我您想「修改」還是「取消」預約。")
        st.rerun()


def render_manage_list():
    st.subheader("📋 我的預約")

    for msg in st.session_state.get("manage_messages", []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    student_id = get_current_user_id()
    reservations = list_user_reservations(student_id, status="confirmed")

    if not reservations:
        st.info("您目前沒有有效的預約。")
        return

    _render_reservation_table(reservations)
    st.caption(
        "💡 想修改或取消，直接告訴我，例如「把 #1 的時間改成下午三點」或「取消 #2」。"
    )

    user_input = st.chat_input("想修改或取消哪一筆預約？")
    if not user_input:
        return

    _handle_manage_input(user_input, reservations)


def render_manage_cancel_confirm():
    st.subheader("📋 取消預約確認")

    pending = st.session_state.get("pending_cancel")
    if not pending:
        st.session_state.booking_stage = "manage_list"
        st.rerun()

    reservation = get_reservation(pending["reservation_id"])
    if not reservation:
        st.error("找不到該預約。")
        st.session_state.pending_cancel = None
        st.session_state.booking_stage = "manage_list"
        st.rerun()

    st.markdown("確定要取消以下預約嗎？")
    st.markdown(f"- **編號**：#{reservation['id']}")
    st.markdown(f"- **討論室**：{reservation['room_id']}")
    st.markdown(f"- **時間**：{reservation['start_time']} ～ {reservation['end_time']}")
    st.markdown(f"- **人數**：{reservation['people_count']} 人")
    st.markdown(f"- **事由**：{reservation['reason']}")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ 確認取消", type="primary", use_container_width=True):
            result = cancel_booking(reservation["id"])
            prefix = "" if result["success"] else "無法取消："
            _append_manage_message(prefix + result["message"])
            st.session_state.pending_cancel = None
            st.session_state.booking_stage = "manage_list"
            st.rerun()
    with col2:
        if st.button("❌ 再想想", use_container_width=True):
            st.session_state.pending_cancel = None
            st.session_state.booking_stage = "manage_list"
            st.rerun()


def render_manage_modify_confirm():
    st.subheader("📋 修改預約確認")

    pending = st.session_state.get("pending_modify")
    if not pending:
        st.session_state.booking_stage = "manage_list"
        st.rerun()

    reservation_id = pending["reservation_id"]
    old = pending["old_fields"]
    new = pending["new_fields"]

    st.markdown(f"**修改預約 #{reservation_id}**，請確認以下變動：")
    st.markdown("| 欄位 | 修改前 | 修改後 |")
    st.markdown("|---|---|---|")
    st.markdown(f"| 討論室 | {old['room_id']} | {new['room_id']} |")
    st.markdown(f"| 人數 | {old['people_count']} | {new['people_count']} |")
    st.markdown(f"| 開始時間 | {old['start_time']} | {new['start_time']} |")
    st.markdown(f"| 結束時間 | {old['end_time']} | {new['end_time']} |")
    st.markdown(f"| 事由 | {old['reason']} | {new['reason']} |")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ 確認修改", type="primary", use_container_width=True):
            result = modify_booking(reservation_id, new)
            prefix = "" if result["success"] else "無法修改："
            _append_manage_message(prefix + result["message"])
            st.session_state.pending_modify = None
            st.session_state.booking_stage = "manage_list"
            st.rerun()
    with col2:
        if st.button("❌ 取消修改", use_container_width=True):
            st.session_state.pending_modify = None
            st.session_state.booking_stage = "manage_list"
            st.rerun()


# =========================================================
# 側邊欄：系統狀態與功能入口
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

    get_booking_db()
    st.sidebar.success("預約討論室系統: 就緒")

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

st.sidebar.divider()
st.sidebar.subheader("📅 預約討論室")

if st.sidebar.button("📅 新增預約", use_container_width=True):
    _start_create()
    st.rerun()

if st.sidebar.button("📋 我的預約", use_container_width=True):
    _start_manage()
    st.rerun()

if st.session_state.get("booking_stage"):
    if st.sidebar.button("結束預約", use_container_width=True):
        _exit_booking()
        st.rerun()


# =========================================================
# 主介面
# =========================================================

st.title("Campus Agent Entrypoint")

booking_stage = st.session_state.get("booking_stage")

if booking_stage == "collect":
    render_booking_chat()
    st.stop()
elif booking_stage == "confirm":
    render_booking_confirmation()
    st.stop()
elif booking_stage == "done":
    render_booking_done()
    st.stop()
elif booking_stage == "manage_list":
    render_manage_list()
    st.stop()
elif booking_stage == "manage_modify_confirm":
    render_manage_modify_confirm()
    st.stop()
elif booking_stage == "manage_cancel_confirm":
    render_manage_cancel_confirm()
    st.stop()

st.markdown("""
本專案為基礎本地聊天 Agent 的輸入前處理與輸出後處理入口。  
目前聚焦於輸入進入模型前的檢查流程與輸出前的格式統一。
""")

if "messages" not in st.session_state:
    system_content = (
        "你是一位友善、專業且親切的台灣的星光大學校園助手。"
        "請一律使用繁體中文與台灣用語回答使用者的問題。"
    )
    st.session_state.messages = [{"role": "system", "content": system_content}]

for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input("請輸入您的問題..."):
    prompt = preprocess_user_input(prompt)

    # 第六層：Query Rewriting
    with st.spinner("正在進行指代還原..."):
        rewrite_result = rewrite_query(
            get_rewriter_llm(), prompt, st.session_state.messages
        )
    rewritten_query = rewrite_result["query"]

    st.session_state.messages.append(
        {"role": "user", "content": prompt, "rewritten": rewritten_query}
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    # 第七層：意圖路由，使用改寫後的查詢
    with st.spinner("正在判斷意圖..."):
        routing_result = route_intent(rewritten_query)

    intent = routing_result["intent"]
    score = routing_result.get("score")
    reason = routing_result.get("reason")
    score_str = f"{score:.3f}" if score is not None else "N/A"

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
                    final_answer = convert_to_traditional(response.content)

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
