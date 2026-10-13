"""
意圖路由層。

使用 Semantic Router 與 OllamaEncoder (bge-m3) 判斷使用者輸入的意圖。
目前定義的意圖包含：
- campus_info: 校園資訊問答 (需 RAG，目前施工中)。
- chitchat: 日常閒聊。
- fallback: 無法明確分類或分數低於門檻的輸入。
"""

import unicodedata
from typing import Any, Dict, Optional
from semantic_router import Route, SemanticRouter
from semantic_router.encoders import OllamaEncoder
from semantic_router.index import LocalIndex

# =========================================================
# 基本設定
# =========================================================

MODEL_NAME = "bge-m3"
FALLBACK_NAME = "fallback"

# 只針對有明確語意的意圖設定門檻
THRESHOLD_CAMPUS_INFO = 0.5
THRESHOLD_CHITCHAT = 0.2


# =========================================================
# 文字正規化
# =========================================================


def unicode_normalize(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u3000", " ")  # 全形空格
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text


def normalize_text(text: str) -> str:
    text = unicode_normalize(str(text))
    return text.lower().strip()


# =========================================================
# 定義原始語料 (只保留有明確語意邊界的意圖)
# =========================================================

campus_info_utterances_raw = [
    # 圖書館與討論室
    "圖書館幾點開？",
    "圖書館週末有開嗎？",
    "圖書館有24小時自習室嗎？",
    "怎麼預約圖書館討論室？",
    "討論室有幾人間的？",
    "M中間討論室有螢幕嗎？",
    "我可以一個人在討論室讀書嗎？",
    "可以在討論室出聲背書嗎？",
    # 網路與宿舍
    "學校Wi-Fi帳號密碼是多少？",
    "Campus-Student 怎麼連？",
    "宿舍 Dorm-WiFi 連不上怎麼辦？",
    "宿舍有門禁嗎？",
    "可以帶朋友回宿舍嗎？",
    "怎麼申請退宿？",
    # 餐廳與交通
    "學餐幾點開？",
    "美食街週末有開嗎？",
    "學校有全家便利商店嗎？",
    "怎麼申請機車停車證？",
    "學生可以開車進校園嗎？",
    "5566公車有到學校嗎？",
    # 英文與中英夾雜
    "What time does the library close?",
    "How to book the discussion room?",
    "請問 library 討論室怎麼預約？",
    "Campus Wi-Fi 密碼是多少？",
    "Dorm 訪客規定是什麼？",
]

chitchat_utterances_raw = [
    # 中文問候
    "你好",
    "哈囉",
    "早安",
    "午安",
    "晚安",
    "嗨，有人在嗎？",
    "你好嗎？",
    "謝謝你。",
    "再見。",
    # 英文問候
    "Hello",
    "Hi there",
    "Good morning",
    "Good evening",
    "How are you?",
    "Thank you.",
    "Bye.",
    # 中英夾雜
    "Hello 你好",
    "Hi 早安",
    "Good morning，今天天氣不錯。",
    "Bye bye，再見。",
]


# =========================================================
# 建立路由（正規化語料 + 設定各自門檻）
# =========================================================

campus_info = Route(
    name="campus_info",
    utterances=[normalize_text(u) for u in campus_info_utterances_raw],
)
campus_info.score_threshold = THRESHOLD_CAMPUS_INFO

chitchat = Route(
    name="chitchat",
    utterances=[normalize_text(u) for u in chitchat_utterances_raw],
)
chitchat.score_threshold = THRESHOLD_CHITCHAT

# SemanticRouter 只接收有明確定義的意圖
encoder = OllamaEncoder(name=MODEL_NAME)
router = SemanticRouter(
    encoder=encoder,
    routes=[campus_info, chitchat],
    index=LocalIndex(),
    auto_sync="local",
)


# =========================================================
# 解析 SemanticRouter 回傳物件
# =========================================================


def extract_score(response: Any) -> Optional[float]:
    val = getattr(response, "similarity_score", None)
    if val is not None:
        try:
            return float(val)
        except Exception:
            pass
    return None


def extract_route_name(response: Any) -> Optional[str]:
    val = getattr(response, "name", None)
    if val not in (None, ""):
        return str(val)
    return None


# =========================================================
# 核心：意圖路由函式
# =========================================================


def route_intent(query: str) -> Dict[str, Any]:
    """
    判斷使用者輸入的意圖。

    Returns:
        {
            "intent":     "campus_info" | "chitchat" | "fallback",
            "route_name": semantic-router 原始判定 (可能為 None),
            "score":      similarity_score,
            "reason":     判定原因
        }
    """
    q = normalize_text(query)

    if not q:
        return {
            "intent": FALLBACK_NAME,
            "route_name": None,
            "score": None,
            "reason": "empty_query",
        }

    response = router(q)
    route_name = extract_route_name(response)
    score = extract_score(response)

    # 1. 完全沒有匹配到任何 Route
    if route_name is None:
        return {
            "intent": FALLBACK_NAME,
            "route_name": None,
            "score": score,
            "reason": "no_match",
        }

    # 2. 手動比對分數與該 Route 的門檻
    if score is not None:
        route_threshold_map = {
            "campus_info": THRESHOLD_CAMPUS_INFO,
            "chitchat": THRESHOLD_CHITCHAT,
        }
        route_threshold = route_threshold_map.get(route_name)

        if route_threshold is not None and score < route_threshold:
            return {
                "intent": FALLBACK_NAME,
                "route_name": route_name,
                "score": score,
                "reason": "below_threshold",
            }
    else:
        return {
            "intent": FALLBACK_NAME,
            "route_name": route_name,
            "score": score,
            "reason": "missing_score",
        }

    # 3. 正常回傳 (分數高於門檻)
    return {
        "intent": route_name,
        "route_name": route_name,
        "score": score,
        "reason": "matched",
    }


def route_intent_name(query: str) -> str:
    return route_intent(query)["intent"]
