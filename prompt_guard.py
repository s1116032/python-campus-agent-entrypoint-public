"""
提示詞注入防護層。

使用 Qwen3Guard 模型檢查輸入是否可能包含提示詞注入。
目前僅使用 4-bit 模式。
"""

from typing import Optional, Tuple, List
import re
import hashlib
from collections import OrderedDict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None


MODEL_NAME = "Qwen/Qwen3Guard-Gen-8B"

# 輸入前端已有文長限制，這裡不再做字元級截斷。
# 內部仍保留 token 安全上限，避免 tokenizer 吃到過長序列。
GUARD_MAX_INPUT_TOKENS = 2048

SAFE_RE = re.compile(r"Safety:\s*(Safe|Unsafe|Controversial)", re.IGNORECASE)

# 匹配 Category: 或 Categories:
CATEGORY_SECTION_RE = re.compile(
    r"Categor(?:y|ies)\s*:\s*(.*)", re.IGNORECASE | re.DOTALL
)

CATEGORY_RE = re.compile(
    r"(Violent|Non-violent Illegal Acts|Sexual Content or Sexual Acts|PII|Suicide & Self-Harm|Unethical Acts|Politically Sensitive Topics|Copyright Violation|Jailbreak|None)",
    re.IGNORECASE,
)


# 簡單 LRU cache，避免同樣輸入重複跑模型。
GUARD_CACHE = OrderedDict()
GUARD_CACHE_MAX_SIZE = 10000


def _get_compute_dtype():
    """
    取得適合的計算精度。
    若支援 bf16，優先使用 bf16；否則使用 fp16。
    """
    try:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
    except Exception:
        pass
    return torch.float16


def load_qwen3guard():
    """
    載入 Qwen3Guard。

    目前只使用 4-bit 模式。
    """
    tokenizer = globals().get("qwen3guard_tokenizer")
    model = globals().get("qwen3guard_model")

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        globals()["qwen3guard_tokenizer"] = tokenizer

    if model is None:
        if BitsAndBytesConfig is None:
            raise RuntimeError(
                "無法使用 BitsAndBytesConfig，請確認 transformers 版本支援 4-bit 量化。"
            )

        try:
            import bitsandbytes  # noqa: F401
        except ImportError as e:
            raise RuntimeError("請安裝 bitsandbytes 才能使用 4-bit Qwen3Guard。") from e

        compute_dtype = _get_compute_dtype()

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            quantization_config=bnb_config,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )

        model.eval()
        globals()["qwen3guard_model"] = model
        print("[Qwen3Guard] Loaded with 4-bit quantization.")

    return tokenizer, model


def _extract_label_and_categories(content: str) -> Tuple[Optional[str], List[str]]:
    """
    從 Qwen3Guard 輸出中解析安全標籤與類別。
    """
    content = content or ""

    safe_match = SAFE_RE.search(content)
    label = safe_match.group(1).capitalize() if safe_match else None

    section_match = CATEGORY_SECTION_RE.search(content)
    category_text = section_match.group(1) if section_match else content

    categories = [c.strip() for c in CATEGORY_RE.findall(category_text) if c.strip()]

    return label, categories


def guard_against_jailbreak_fast(
    user_text: str,
    *,
    fail_open: bool = False,
    use_cache: bool = True,
    max_new_tokens: int = 24,
) -> str:
    """
    使用 Qwen3Guard 檢查輸入是否可能包含提示詞注入。

    Args:
        user_text: 使用者輸入。
        fail_open:
            False: 模型異常或無法判斷時拒絕。
            True: 模型異常或無法判斷時放行。
        use_cache: 是否使用快取。
        max_new_tokens: 模型生成上限。

    Returns:
        "accept" 或 "refuse"。
    """
    try:
        if user_text is None:
            return "accept"

        text = str(user_text).strip()

        if not text:
            return "accept"

        cache_key = hashlib.sha256(text.encode("utf-8")).hexdigest()

        if use_cache and cache_key in GUARD_CACHE:
            GUARD_CACHE.move_to_end(cache_key)
            return GUARD_CACHE[cache_key]

        tokenizer, model = load_qwen3guard()

        messages = [{"role": "user", "content": text}]

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        model_inputs = tokenizer(
            [prompt],
            return_tensors="pt",
            truncation=True,
            max_length=GUARD_MAX_INPUT_TOKENS,
        ).to(model.device)

        with torch.inference_mode():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )

        output_ids = generated_ids[0][model_inputs.input_ids.shape[1] :].tolist()
        content = tokenizer.decode(output_ids, skip_special_tokens=True)

        label, categories = _extract_label_and_categories(content)

        # 只要出現 Jailbreak，就拒絕。
        if any(c.lower() == "jailbreak" for c in categories):
            decision = "refuse"

        # 輸出無法解析：保守拒絕，除非 fail_open。
        elif label is None:
            decision = "accept" if fail_open else "refuse"

        # Unsafe 但沒有類別，可能是輸出被截斷或格式異常。
        elif label.lower() == "unsafe" and not categories:
            decision = "accept" if fail_open else "refuse"

        else:
            decision = "accept"

        if use_cache:
            GUARD_CACHE[cache_key] = decision
            if len(GUARD_CACHE) > GUARD_CACHE_MAX_SIZE:
                GUARD_CACHE.popitem(last=False)

        return decision

    except Exception:
        return "accept" if fail_open else "refuse"
