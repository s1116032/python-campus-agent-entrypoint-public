"""
輸出後處理層。

使用 OpenCC 將模型輸出統一轉換為繁體中文，並處理地區慣用詞。
避免模型因使用者輸入簡體字而跟著回覆簡體字。
"""

from opencc import OpenCC

# 模組級單例，避免每次請求重新載入字典。's2twp' 包含簡轉繁與短語轉換（如：软件 -> 軟體）。
_converter = OpenCC("s2twp")


def convert_to_traditional(text: str) -> str:
    """
    將簡體中文或混合文本轉換為繁體中文（含台灣慣用語）。

    Args:
        text: 模型輸出的原始文字。

    Returns:
        轉換後的繁體中文。若輸入無效或轉換失敗，則回傳原文。
    """
    if not isinstance(text, str) or not text.strip():
        return text

    try:
        return _converter.convert(text)
    except Exception:
        return text
