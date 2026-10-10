"""
語言檢測層。

僅允許中文與英文輸入。
短文本採取較寬鬆的判斷策略，長文本則切分後逐段檢查。
"""

import re
from lingua import Language, LanguageDetectorBuilder

# =========================================================
# 規則參數
# =========================================================

# 目前允許的語言。
ACCEPTED_LANGUAGES = {
    Language.CHINESE,
    Language.ENGLISH,
}

# 英文信心值高於此門檻時，視為英文輸入。
ENGLISH_CONFIDENCE_THRESHOLD = 0.05

# 小於等於此單位數的文本視為短文本。
SHORT_UNIT_THRESHOLD = 4

# 短文本中，若最高語言信心值高於此門檻，則接受該判斷。
HIGH_CONFIDENCE_THRESHOLD = 0.5

# 超過此單位數的文本會切分後逐段檢查。
LONG_TEXT_THRESHOLD = 20

# 長文本切分時，每段目標單位數。
TARGET_SEGMENT_UNITS = 10


# =========================================================
# lingua detector
# =========================================================

# 使用所有語言建立檢測器。
detector = LanguageDetectorBuilder.from_all_languages().build()


# =========================================================
# 文字清洗
# =========================================================

# 語言檢測只依賴中文、英文與空白。
# 標點、符號、數字等非語言特徵會先被替換為空白。
LANGUAGE_KEEP_PATTERN = re.compile(
    "[^\u4e00-\u9fa5\u3400-\u4dbfA-Za-z\uff21-\uff3a\uff41-\uff5a\\s]+"
)


def normalize_for_language_detection(text: str) -> str:
    """
    清洗文字，只保留語言判斷需要的字元。

    因為前一層已經做過 NFKC 正規化，這裡不再重複執行。
    """
    if not isinstance(text, str):
        return ""

    text = LANGUAGE_KEEP_PATTERN.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# =========================================================
# 單位計算與切分
# =========================================================

# 語言單位定義：
# - 單一中文字算 1 個單位。
# - 連續英文字母算 1 個單位。
UNIT_PATTERN = re.compile(
    "(?:[\u4e00-\u9fa5\u3400-\u4dbf])|(?:[A-Za-z\uff21-\uff3a\uff41-\uff5a]+)"
)


def count_units_from_cleaned(cleaned_text: str) -> int:
    """
    計算已清洗文本中的語言單位數。
    """
    if not cleaned_text:
        return 0

    return len(UNIT_PATTERN.findall(cleaned_text))


def count_zh_en_units(text: str) -> int:
    """
    計算原始文本中的中英文語言單位數。
    """
    cleaned_text = normalize_for_language_detection(text)
    return count_units_from_cleaned(cleaned_text)


def split_cleaned_text_into_segments(
    cleaned_text: str, target_units: int = TARGET_SEGMENT_UNITS
):
    """
    將已清洗文本切成多段。

    每段盡量接近 target_units 個語言單位，
    用於長文本逐段語言檢測。
    """
    units = list(UNIT_PATTERN.finditer(cleaned_text))
    unit_count = len(units)

    if unit_count == 0:
        return []

    num_segments = max(1, int(unit_count / target_units + 0.5))
    num_segments = min(num_segments, unit_count)

    base_size = unit_count // num_segments
    remainder = unit_count % num_segments

    segments = []
    idx = 0

    for seg_idx in range(num_segments):
        size = base_size + (1 if seg_idx < remainder else 0)

        if size <= 0:
            continue

        seg_units = units[idx : idx + size]

        if not seg_units:
            idx += size
            continue

        start = seg_units[0].start()
        end = seg_units[-1].end()

        segment = cleaned_text[start:end].strip()

        if segment:
            segments.append(segment)

        idx += size

    return segments


# =========================================================
# lingua 信心值輔助函式
# =========================================================


def _to_language_confidence_tuple(item):
    """
    將 lingua 的信心值結果轉為 (language, confidence)。

    相容 ConfidenceValue 物件或 (language, value) tuple。
    """
    lang = getattr(item, "language", None)
    value = getattr(item, "value", None)

    if lang is None and isinstance(item, tuple) and len(item) >= 1:
        lang = item[0]

    if value is None and isinstance(item, tuple) and len(item) >= 2:
        value = item[1]

    try:
        value = float(value)
    except Exception:
        value = 0.0

    return lang, value


def get_top_and_english_confidence(text: str):
    """
    取得最高語言信心值與英文信心值。

    Returns:
        top_lang, top_conf, en_conf
    """
    try:
        raw_values = list(detector.compute_language_confidence_values(text))
        values = [_to_language_confidence_tuple(item) for item in raw_values]
    except Exception:
        values = []

    if values:
        values = sorted(values, key=lambda item: item[1], reverse=True)
        top_lang, top_conf = values[0]

        en_conf = next(
            (conf for lang, conf in values if lang == Language.ENGLISH), None
        )
    else:
        top_lang, top_conf = None, 0.0
        en_conf = None

    if en_conf is None:
        try:
            en_conf = detector.compute_language_confidence(text, Language.ENGLISH)
        except Exception:
            en_conf = 0.0

    return (top_lang, float(top_conf or 0.0), float(en_conf or 0.0))


# =========================================================
# 單段語言檢查
# =========================================================


def _check_language_zh_en_core(cleaned_text: str) -> str:
    """
    單段文本語言檢查。

    規則：
    1. 英文信心值達門檻，直接接受。
    2. 短文本若高信心語言為中文，接受；非中英文則拒絕。
    3. 長文本只檢查最高語言是否為中文或英文。
    """
    if not cleaned_text.strip():
        return "accept"

    unit_count = count_units_from_cleaned(cleaned_text)

    if unit_count == 0:
        return "accept"

    try:
        top_lang, top_conf, en_conf = get_top_and_english_confidence(cleaned_text)

        if en_conf >= ENGLISH_CONFIDENCE_THRESHOLD:
            return "accept"

        if unit_count <= SHORT_UNIT_THRESHOLD:
            if (
                top_lang is not None
                and top_lang != Language.ENGLISH
                and top_conf > HIGH_CONFIDENCE_THRESHOLD
            ):
                if top_lang == Language.CHINESE:
                    return "accept"
                return "refuse"

            return "accept"

        if top_lang is None:
            return "accept"

        if top_lang in ACCEPTED_LANGUAGES:
            return "accept"

        return "refuse"

    except Exception:
        return "refuse"


# =========================================================
# 主入口
# =========================================================


def check_language_zh_en(text: str) -> str:
    """
    語言檢測主入口。

    流程：
    1. 清洗文字。
    2. 計算語言單位數。
    3. 短文本直接檢查。
    4. 長文本切分後逐段檢查。
    """
    if not isinstance(text, str):
        return "refuse"

    cleaned_text = normalize_for_language_detection(text)

    if not cleaned_text:
        return "accept"

    unit_count = count_units_from_cleaned(cleaned_text)

    if unit_count == 0:
        return "accept"

    if unit_count <= LONG_TEXT_THRESHOLD:
        return _check_language_zh_en_core(cleaned_text)

    segments = split_cleaned_text_into_segments(
        cleaned_text, target_units=TARGET_SEGMENT_UNITS
    )

    if not segments:
        return "accept"

    for segment in segments:
        segment_result = _check_language_zh_en_core(segment)

        if segment_result == "refuse":
            return "refuse"

    return "accept"
