# campus-agent-entrypoint

本專案是從 `python-local-campus-ai-assistant-public` 獨立出來的重構專案。

前一專案包含 RAG 校園知識問答與預約討論室功能。本專案重新整理架構，把重點放在**穩健的輸入處理流程**、**明確的意圖分流**、**可維護的 RAG 檢索**，以及**對話式的預約討論室功能**，讓每一層都有清楚的職責與可測試的邊界。

---

## 架構規劃

本專案作為校園 Agent 的入口層，目前已實作：

- 輸入前處理（白名單、正規化、語言檢測）。
- 提示詞注入防護層。
- Query Rewriting（指代還原）。
- 意圖路由層。
- 校園資訊 RAG 問答（BM25 + bge-m3 混合檢索）。
- **預約討論室（新增、查看、修改、取消）**。
- 輸出後處理（簡轉繁與慣用語）。

---

## 核心處理流程

### 1. 校園資訊問答與閒聊

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
提示詞注入防護（Qwen3Guard）
    ↓
Query Rewriting（指代還原）
    ↓
意圖路由 (Semantic Router)
    ↓
    ├─ campus_info  → RAG 檢索（BM25 + bge-m3 + RRF）→ LLM 生成
    └─ chitchat / fallback → LLM 生成
    ↓
輸出後處理（簡轉繁與慣用語）
    ↓
回傳給使用者
```

### 2. 預約討論室 (Booking)

預約功能獨立於一般聊天，透過側邊欄切換模式（`新增預約` / `我的預約`），並**與一般聊天共用同一套輸入前處理**以確保安全性。

- **新增預約 (Create)**：透過對話自由收集欄位，欄位不齊全時由程式自動補問。收集完畢後顯示「確認卡片」，使用者點擊確認後才寫入資料庫。
- **我的預約 (Manage)**：列表顯示目前有效的預約（附帶穩定編號如 `#1`），使用者可透過對話表達「把 #1 的時間改成三點」或「取消 #2」。
- **修改機制**：採用「軟刪除舊預約（狀態設為 `superseded`）+ 新增新預約」，並透過 `superseded_by_id` 建立版本鏈。此設計避免了修改中途放棄時的回滾問題，並保留完整的修改歷史。
- **取消機制**：將預約狀態設為 `cancelled`。
- **後端驗證**：嚴格檢查事由黑名單、開放時段、時段衝突（修改時自動排除自己）、討論室容量與人數下限。

---

## 各層說明

### 1. 第一次 Regex 白名單 + 文長限制
檢查輸入是否只包含允許的字元，並限制長度。這一步在正規化之前執行，主要先防範畸形輸入與原始碼攻擊。

### 2. Unicode 正規化
使用 `NFKC` 將輸入統一為標準形式（全形轉半形、相容字元統一、統一換行符號）。此步驟參考 OWASP Canonicalization 原則，降低因字元表示不同造成的判斷落差。

### 3. 第二次 Regex 白名單 + 文長限制
正規化後文字內容可能改變，因此再檢查一次，確保仍符合白名單與長度限制。

### 4. 語言檢測
使用 `lingua-language-detector` 判斷輸入語言，僅允許中文與英文。目的是在進入較重的安全模型之前先做輕量篩選。

### 5. 提示詞注入防護
使用 `Qwen3Guard-Gen-8B`（4-bit）檢查輸入是否可能包含提示詞注入。此層可透過 `.env` 開關停用，以節省開發時的載入時間。

### 6. Query Rewriting（指代還原）
在進入意圖路由之前，把帶有指代的問題還原為可獨立理解的查詢。原則是有信心才跳過，其餘一律呼叫 LLM 改寫，並採用 **JSON 結構化輸出**與 **Few-shot 範例**提升改寫品質。Prompt 使用 XML 標籤隔離系統指令、歷史與輸入。

### 7. 意圖路由
使用 `semantic-router` 與 `bge-m3` 判斷使用者意圖，分為 `campus_info`（走 RAG）與 `chitchat / fallback`（直接送 LLM）。路由使用改寫後的查詢，因為指代已還原。

### 8. 校園資訊 RAG 問答
針對 `campus_info` 意圖，使用混合檢索從校園知識庫取得相關內容：
- **稀疏檢索**：BM25（以 `jieba` 中文分詞，解決中文無空白導致切詞失效的問題）。
- **稠密檢索**：bge-m3 語意向量，存於 Chroma。
- **融合**：以 `EnsembleRetriever` 內建的 RRF 融合兩路結果。

### 9. 輸出後處理
使用 OpenCC（`s2twp`）將模型輸出統一轉換為繁體中文與台灣慣用詞，避免模型因使用者輸入簡體字而跟著回覆簡體字。

### 10. 預約討論室 (Booking)
- **`booking/db.py`**：SQLite 資料庫層。管理 `rooms` 與 `reservations` 表，支援時段衝突檢查（修改時可排除自己），並啟用外鍵約束。
- **`booking/extractor.py`**：LLM 提取層。將自然語言轉為結構化草稿（Create 模式），或解析修改/取消意圖與目標編號（Manage 模式）。
- **`booking/service.py`**：業務邏輯層。處理身分取得、事由黑名單、開放時段檢查、時段衝突驗證，以及送出/修改/取消的商業邏輯。

---

## 技術選型

| 用途 | 使用技術 |
|---|---|
| 前端介面 | Streamlit |
| Agent 通訊架構 | FastMCP |
| 本地 LLM | Ollama + `qwen2.5:7b` |
| 語言檢測 | `lingua-language-detector` |
| 提示詞注入防護 | `Qwen3Guard-Gen-8B`、`transformers`、`bitsandbytes` |
| Query Rewriting | LangChain + XML 標籤 + JSON 結構化輸出 |
| 意圖路由 | `semantic-router`、`bge-m3` |
| RAG 稀疏檢索 | BM25（`rank_bm25`）、`jieba` 中文分詞 |
| RAG 稠密檢索 | `bge-m3`、Chroma |
| RAG 結果融合 | `EnsembleRetriever`（RRF，`langchain-classic`） |
| 輸出後處理 | OpenCC |
| 預約資料儲存 | SQLite（Python 標準函式庫） |
| 依賴管理 | `pip-tools` |

---

## 專案結構

```text
campus-agent-entrypoint/
├── knowledge_base/
│   └── campus_faq.md          # 校園知識庫（單一檔案）
├── booking/                   # 預約討論室模組
│   ├── __init__.py
│   ├── db.py                  # SQLite 資料庫操作
│   ├── extractor.py           # LLM 欄位與意圖提取
│   └── service.py             # 業務邏輯與驗證
├── data/                      # SQLite 資料庫檔案 (自動產生)
├── rag/                       # RAG 混合檢索模組
│   ├── __init__.py
│   ├── document_loader.py     # 讀取與切分知識庫
│   └── retriever.py           # BM25 + bge-m3 + RRF 混合檢索
├── app.py                     # Streamlit 主程式與流程編排
├── mcp_server.py              # 最小 MCP Server
├── input_guard.py             # 白名單、文長限制、Unicode 正規化
├── language_guard.py          # 語言檢測
├── prompt_guard.py            # 提示詞注入防護
├── query_rewriter.py          # Query Rewriting（指代還原）
├── router.py                  # 意圖路由
├── output_guard.py            # 輸出簡轉繁
├── .env                       # 環境變數開關
├── requirements.in            # 手動維護的正式依賴
└── tests/
    ├── conftest.py
    ├── test_input_guard.py
    └── manual_test_cases.txt  # 手動測試案例（含邊界與指代情境）
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

### 環境變數（`.env`）

```env
# 提示詞注入防護開關
PROMPT_GUARD_ENABLED=false

# Query Rewriting 除錯日誌開關
REWRITE_DEBUG=false
```

啟動應用：

```bash
streamlit run app.py --server.headless true
```

---

## 測試

專案提供手動測試案例，涵蓋輸入防護、簡體字誘導、以及各類指代消解情境：

```text
tests/manual_test_cases.txt
```

---

## 已知限制

- 語言檢測目前使用較小規模資料與規則調整參數，短文本或混合語言仍可能誤判。
- 意圖路由對於語意相近但意圖不同的輸入，仍可能誤判。
- 預約管理（新增、修改、取消）等確定性高的業務流程，目前不走 Semantic Router，而是透過 UI 狀態機與專屬 Extractor 處理。這是因為 Semantic Router 每新增一個細粒度意圖，都需要擴充大量語料並重新調整相似度閾值，維護成本與誤判風險較高。
- 預約系統目前使用預設的固定 `student_id`，尚未串接真實登入 API。
- 預約系統目前不支援透過對話查詢歷史預約（如「顯示我已取消的預約」）。

---

## 未來規劃

- **預約系統：真實身分驗證**：串接登入 API 取得真實 `student_id`，取代目前的預設值。
- **預約系統：對話式查詢過濾**：支援透過對話查詢歷史預約與過濾特定狀態。
- **意圖路由 (Semantic Router) 擴充**：未來若需將更多工具 (Tools) 接入全域路由，規劃引入 LLM 輔助生成語料與自動化閾值校準機制，以降低擴充語料與調參的成本。
- **RAG 專用 Query Rewriting**：目前 RAG 復用意圖路由前的改寫結果，未來可針對檢索場景做專屬改寫，提升檢索命中率。
- **校園資訊檢索的 metadata 過濾**：已改用 Chroma 作為向量庫，未來可加上章節、類別等 metadata，支援過濾式檢索。
- **對話操作指令**：處理「再說一次」、「第二點展開講」等變體較多的指令，預計走專用 handler 以節省 LLM 成本。
- **日曆型時間解析**：改用 Duckling 將相對時間自動替換為絕對時間。
- **輸出安全過濾**：對模型輸出進行毒性、幻覺等進階檢查。

---

## License

Copyright © 2026 hanwu910514.

詳情請參閱[Apache License 2.0](LICENSE)檔案