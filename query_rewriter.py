"""
Query Rewriting 層。

目的：在進入意圖路由之前，把帶有指代的問題還原為可獨立理解的查詢。

設計原則：有信心才跳過。
目前只在以下情況跳過 LLM 改寫：
1. 對話首輪，沒有可參考的上下文。
2. 寒暄、感謝、道別，或以「你」詢問 Agent 本身，明顯沒有需要解析的內容。

其餘一律呼叫 LLM 改寫，並強制使用 JSON 結構化輸出與思維鏈 (CoT) 確保改寫品質。

注意：日曆型時間的自動解析（如今天、下週三）目前未實作，
留待後續改用專門的時間解析套件處理，避免規則寫死或依賴 LLM 判斷。
"""

import re
import json
import logging
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

# =========================================================
# 設定
# =========================================================

MAX_HISTORY_TURNS = 3

MAX_REWRITTEN_LENGTH = 512

SMALL_TALK_PATTERN = re.compile(
    r"^(哈囉|嗨|你好|您好|早安|午安|晚安|"
    r"hi|hello|hey|good morning|good afternoon|good evening|good night|"
    r"謝謝|謝謝你|感謝|多謝|thanks|thank you|thx|"
    r"再見|掰掰|bye|byebye|see you)"
    r"[\s，。！？!?、,\.…~]*$",
    re.IGNORECASE,
)

AGENT_QUESTION_PATTERN = re.compile(
    r"(你是誰|你叫什麼|你叫啥|你是什麼|你是機器人|你是ai|"
    r"你能做什麼|你會做什麼|你可以做什麼|你會什麼|你有哪些功能|"
    r"你來自哪裡|你用的是什麼模型|你是什麼模型)",
    re.IGNORECASE,
)


# =========================================================
# 跳過判斷 (系統層級的快速攔截)
# =========================================================


def count_user_turns(messages: List[Dict[str, Any]]) -> int:
    return sum(1 for m in messages if m.get("role") == "user")


def should_skip_rewrite(
    text: str,
    messages: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if count_user_turns(messages) == 0:
        return {"skip": True, "reason": "first_turn"}

    normalized = (text or "").strip()
    if not normalized:
        return {"skip": True, "reason": "empty_query"}

    if SMALL_TALK_PATTERN.match(normalized):
        return {"skip": True, "reason": "small_talk"}

    if AGENT_QUESTION_PATTERN.search(normalized):
        return {"skip": True, "reason": "agent_question"}

    return {"skip": False, "reason": "needs_rewrite"}


# =========================================================
# Few-shot 範例
# =========================================================

FEW_SHOT_EXAMPLES = """<example>
<history>
使用者：校園餐廳有兩個區域吧？
助手：是的，第一餐廳和第二餐廳。
</history>
<user_query>
距離有點遠耶
</user_query>
<output>
{
  "reason": "『距離』省略了主詞，是評論句中的零指代，指第一餐廳與第二餐廳之間的距離，需補齊。",
  "needs_rewrite": true,
  "rewritten_query": "第一餐廳和第二餐廳的距離有點遠耶"
}
</output>
</example>

<example>
<history>
使用者：我昨天去圖書館借書，結果被罰款了。
助手：為什麼會被罰款呢？
</history>
<user_query>
就是這件事怎麼申訴？
</user_query>
<output>
{
  "reason": "『這件事』是模糊指代，指涉上一句提到的『去圖書館借書被罰款』的事件。",
  "needs_rewrite": true,
  "rewritten_query": "去圖書館借書被罰款這件事怎麼申訴？"
}
</output>
</example>

<example>
<history>
使用者：上次說的那本課外書叫什麼？
助手：是《三體》。
</history>
<user_query>
還有續集嗎？
</user_query>
<output>
{
  "reason": "『還有續集嗎』省略主詞，指涉上一句提到的《三體》是否有續集。",
  "needs_rewrite": true,
  "rewritten_query": "《三體》還有續集嗎？"
}
</output>
</example>

<example>
<history>
使用者：我正在學餐吃晚餐，Wi-Fi訊號忽強忽弱。
助手：請問是學餐哪個區域呢？需要幫您報修嗎？
</history>
<user_query>
這邊的超商是24小時嗎
</user_query>
<output>
{
  "reason": "「這邊」是空間指代，指涉上一句提到的「學餐」。雖然句子有主詞「超商」，但必須將「這邊」還原為具體地點才能精準回答。",
  "needs_rewrite": true,
  "rewritten_query": "學餐的超商是24小時嗎？"
}
</output>
</example>

<example>
<history>
使用者：圖書館幾點開？
助手：早上八點。
</history>
<user_query>
圖書館週日有開嗎？
</user_query>
<output>
{
  "reason": "這句話語意已完整，包含主詞與疑問對象，且沒有空間或時間代詞，沒有需要還原的指代。",
  "needs_rewrite": false,
  "rewritten_query": "圖書館週日有開嗎？"
}
</output>
</example>"""


# =========================================================
# prompt 組裝
# =========================================================


def _format_history(messages: List[Dict[str, Any]]) -> str:
    turns = [m for m in messages if m.get("role") in ("user", "assistant")]
    turns = turns[-(MAX_HISTORY_TURNS * 2) :]

    if not turns:
        return "（無）"

    lines = []
    for m in turns:
        role = "使用者" if m["role"] == "user" else "助手"
        content = m.get("rewritten") or m.get("content", "")
        lines.append(f"{role}：{content}")

    return "\n".join(lines)


def build_rewrite_prompt(
    text: str,
    messages: List[Dict[str, Any]],
) -> str:
    history_text = _format_history(messages)

    return f"""<system>
你是 Query Rewriting 模組，負責指代還原。

任務：根據對話歷史，把使用者最新的一句話改寫成「不依賴上下文也能理解」的完整句子。

請遵守以下規則：
1. 只做指代還原，不要回答問題，不要新增原始輸入沒有的資訊。
2. 如果指涉對象已經出現在同一則訊息內，不需要改寫。
3. 需要改寫的對象通常包含：
   - 時間與空間指代：如「上次、這邊、那邊、這裡、那裡、樓上、樓下」。即使句子結構看似完整（例如有主詞），若包含這些代詞且上下文有明確線索，請盡量將其還原為具體的時間或地點。
   - 省略主詞或賓語的零指代：包括評論句（例如「距離有點遠耶」）。
   - 模糊指代：如「這件事」、「這樣」。
   - 部分與整體或同義替換的指代：如上一句提到「行李箱」，這句提到「輪子」。
4. 如果無法確定指代對象，保留原句，不要猜測。

輸出格式：請嚴格輸出合法的 JSON 格式，不要包含任何 Markdown 程式碼框（如 ```json）或額外說明：
{{
  "reason": "簡短分析是否有指代，以及指代對象是什麼。若無指代請說明為何不需要改寫。",
  "needs_rewrite": true,
  "rewritten_query": "改寫後的完整句子。若不需要改寫，請原封不動輸出原始輸入。"
}}
</system>

<examples>
{FEW_SHOT_EXAMPLES}
</examples>

<history>
{history_text}
</history>

<user_query>
{text}
</user_query>
"""


# =========================================================
# 輸出解析 (防禦性 JSON 解析)
# =========================================================


def _parse_json_output(text: str) -> dict:
    """
    嘗試解析 LLM 輸出的 JSON。
    包含防禦機制，處理 LLM 偶爾會加上 Markdown 標籤或前後廢話的問題。
    """
    text = (text or "").strip()

    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return {}


# =========================================================
# 主入口
# =========================================================


def rewrite_query(
    llm: Any,
    text: str,
    messages: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Query Rewriting 主入口。
    """
    decision = should_skip_rewrite(text, messages)
    if decision["skip"]:
        logger.info(
            "[QueryRewrite] SKIP | reason=%s | input=%r",
            decision["reason"],
            text,
        )
        return {
            "query": text,
            "rewritten": False,
            "original": text,
            "reason": decision["reason"],
        }

    prompt = build_rewrite_prompt(text, messages)
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        raw_output = response.content
    except Exception as e:
        logger.warning("[QueryRewrite] LLM_ERROR | input=%r | error=%s", text, e)
        return {
            "query": text,
            "rewritten": False,
            "original": text,
            "reason": "rewrite_error",
        }

    parsed = _parse_json_output(raw_output)

    if not parsed:
        logger.warning(
            "[QueryRewrite] JSON_PARSE_ERROR | input=%r | raw_output=%r",
            text,
            raw_output,
        )
        return {
            "query": text,
            "rewritten": False,
            "original": text,
            "reason": "json_parse_error",
        }

    needs_rewrite = parsed.get("needs_rewrite", False)
    rewritten_query = parsed.get("rewritten_query", text)
    llm_reason = parsed.get("reason", "no_reason")

    invalid = (
        not isinstance(rewritten_query, str)
        or not rewritten_query.strip()
        or len(rewritten_query) > MAX_REWRITTEN_LENGTH
    )

    if not needs_rewrite or rewritten_query == text or invalid:
        logger.info(
            "[QueryRewrite] NO_CHANGE | input=%r | llm_reason=%r",
            text,
            llm_reason,
        )
        return {
            "query": text,
            "rewritten": False,
            "original": text,
            "reason": f"no_change ({llm_reason})",
        }

    logger.info(
        "[QueryRewrite] REWRITTEN | input=%r | output=%r | llm_reason=%r",
        text,
        rewritten_query,
        llm_reason,
    )
    return {
        "query": rewritten_query,
        "rewritten": True,
        "original": text,
        "reason": f"rewritten ({llm_reason})",
    }
