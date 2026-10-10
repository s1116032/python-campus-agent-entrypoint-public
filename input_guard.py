"""
輸入前處理層。

包含：
1. Regex 白名單檢查。
2. 輸入長度限制。
3. Unicode 正規化。

白名單檢查會在正規化前後各執行一次，
確保正規化後的文字仍符合預期格式。
"""

import re
import unicodedata

# 允許字元集合。
# 以白名單方式限制輸入字元，較符合輸入內容可預期的場景。
ALLOWED_CHARS = "".join(
    (
        "\u4e00-\u9fa5",  # 常用中文 / 繁體中文 CJK 範圍
        "\u3400-\u4dbf",  # CJK 擴充 A
        "A-Za-z0-9",  # 英文與數字
        "\uff21-\uff3a",  # 全形大寫 Ａ-Ｚ
        "\uff41-\uff5a",  # 全形小寫 ａ-ｚ
        "\uff10-\uff19",  # 全形數字 ０-９
        " \r\n",  # 空白與換行
        "，。！？；：「」『』（）—…　",  # 常見全形標點與全形空格
        "‘’“”",  # 彎引號
        "《》〈〉",  # 書名號
        "【】〖〗",  # 方頭括號
        "·・",  # 譯名常用間隔號
        "／～＋＝",  # 全形斜線、浪號、加號、等號
        ",.!?:;\"'()@$%&+=#~_/-",  # 半形標點
    )
)

# 預先編譯，避免每次請求重複編譯。
ALLOWED_PATTERN = re.compile(f"[{ALLOWED_CHARS}]+")


def check_allowed_chars(
    text: str, allow_empty: bool = False, max_length: int = 1024
) -> str:
    """
    檢查輸入是否符合白名單與長度限制。

    Args:
        text: 待檢查字串。
        allow_empty: 是否允許空字串。
        max_length: 允許的最大字元長度。

    Returns:
        "accept" 或 "refuse"。
    """
    if not isinstance(text, str):
        return "refuse"

    if len(text) > max_length:
        return "refuse"

    if text == "":
        return "accept" if allow_empty else "refuse"

    return "accept" if ALLOWED_PATTERN.fullmatch(text) is not None else "refuse"


def unicode_normalize(text: str) -> str:
    """
    將輸入正規化為一致形式。

    使用 NFKC 正規化，例如：
    - 全形英數轉半形。
    - 相容字元統一。
    - 全形空格轉一般空格。

    正規化後可能改變原始字元，因此需要再次執行白名單檢查。
    """
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFKC", text)

    # 補強處理 NFKC 後仍可能殘留的全形空格。
    text = text.replace("\u3000", " ")

    # 統一換行格式。
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    return text
