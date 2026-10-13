# campus-agent-entrypoint

本專案是從 `python-local-campus-ai-assistant-public` 獨立出來的重構專案。

前一專案包含 RAG 校園知識問答與預約討論室功能。本專案先限縮至基礎本地聊天 Agent，專注整理使用者輸入進入模型前的前處理流程與意圖路由。

---

## 架構規劃

本專案作為 Agent 的入口層，目前已實作：

- 輸入前處理（白名單、正規化、語言檢測）。
- 提示詞注入防護層。
- 意圖路由層。

---

## 目前輸入處理流程

目前使用者輸入會經過以下流程：

```text
使用者輸入
    ↓
第一次 Regex 白名單 + 文長限制
    ↓
Unicode 正規化
    ↓
第二次 Regex 白名單 + 文長限制
    ↓
語言檢測，僅允許中文與英文
    ↓
提示詞注入防護
    ↓
意圖路由 (Semantic Router)
    ↓
送入本地 LLM 或 回傳施工中訊息
```

---

## 各層說明

### 1. 第一次 Regex 白名單 + 文長限制

檢查輸入是否只包含允許的字元，並限制長度。

這一步在 Unicode 正規化之前執行，主要先防範畸形輸入與原始碼攻擊。

目前允許：

- 中文與英文。
- 半形與全形英數。
- 常見中英文標點。
- 空白與換行。
- 彎引號、書名號、方頭括號、間隔號等常見輸入字元。

### 2. Unicode 正規化

使用 `NFKC` 做基本正規化，例如：

- 全形英數轉半形。
- 相容字元統一。
- 全形空格轉一般空格。
- 統一換行符號。

此步驟參考 OWASP Canonicalization 原則，先將輸入統一為較標準的形式，再進入後續檢查，降低因字元表示不同造成的判斷落差。

### 3. 第二次 Regex 白名單 + 文長限制

正規化後，文字內容可能改變，因此再檢查一次。

例如：

- 全形字元轉半形後是否仍符合白名單。
- 正規化後長度是否仍符合限制。
- 是否產生原本未預期的字元。

### 4. 語言檢測

使用 `lingua-language-detector` 判斷輸入語言。

目前僅允許：

- 中文。
- 英文。

此層目的：

- 避免不支援語言進入後續流程。
- 降低模型處理未知語言時的成本與不確定性。
- 在較重的安全模型之前先做輕量篩選。

### 5. 提示詞注入防護

使用 `Qwen3Guard-Gen-8B` 模型檢查輸入是否可能包含提示詞注入。目前僅使用 4-bit 模式。

### 6. 意圖路由

使用 `semantic-router` 與 `bge-m3` 模型判斷使用者意圖。
目前定義的意圖包含：

- `campus_info`: 校園資訊問答。因需 RAG，目前先回傳「施工中」訊息。
- `chitchat`: 日常閒聊，交由 LLM 處理。
- `fallback`: 無法明確分類，交由 LLM 處理。

---

## 技術選型

| 用途 | 使用技術 |
|---|---|
| 前端介面 | Streamlit |
| Agent 通訊架構 | FastMCP |
| 本地 LLM | Ollama |
| LLM 模型 | `qwen2.5:7b` |
| LangChain 整合 | `langchain-ollama`、`langchain-core` |
| 語言檢測 | `lingua-language-detector` |
| 提示詞注入防護 | `Qwen3Guard-Gen-8B`、`transformers`、`bitsandbytes` |
| 意圖路由 | `semantic-router`、`bge-m3` |
| 單元測試 | `pytest` |
| 依賴管理 | `pip-tools` |

---

## 專案結構

```text
campus-agent-entrypoint/
├── app.py                  # Streamlit 主程式與聊天流程
├── mcp_server.py           # 最小 MCP Server
├── input_guard.py          # 白名單、文長限制、Unicode 正規化
├── language_guard.py       # 中英文語言檢測
├── prompt_guard.py         # 提示詞注入防護
├── router.py               # 意圖路由
├── requirements.in         # 手動維護的正式依賴
├── requirements.txt        # 由 pip-compile 自動產生
├── requirements-dev.in     # 手動維護的開發依賴
├── requirements-dev.txt    # 由 pip-compile 自動產生
└── tests/
    ├── conftest.py
    ├── test_input_guard.py
    └── manual_test_cases.txt
```

---

## 安裝與執行

安裝依賴：

```bash
pip install -r requirements-dev.txt
```

下載本地模型：

```bash
ollama pull qwen2.5:7b
ollama pull bge-m3
```

啟動應用：

```bash
streamlit run app.py --server.headless true
```

---

## 測試

執行自動化測試：

```bash
pytest
```

專案中也提供手動測試案例：

```text
tests/manual_test_cases.txt
```

---

## 已知限制

- 語言檢測目前使用較小規模資料與規則調整參數，短文本、混合語言或特殊用語仍可能誤判。
- 意圖路由目前使用較小規模語料與規則調整參數，對於語意相近但意圖不同的輸入，或較複雜的長句，仍可能誤判。
- 編碼後資料的種類較多，若要處理可能需要多層檢查，且可能產生誤報副作用，因此目前先不納入。

---

## 未來規劃

- 校園資訊問答 (RAG) 實際對話功能。
- 輸出端檢查。
- 指代消解。

---

## License

Copyright © 2026 hanwu910514.

詳情請參閱[Apache License 2.0](LICENSE)檔案