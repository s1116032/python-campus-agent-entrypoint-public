# campus-agent-entrypoint

本專案是從 `python-local-campus-ai-assistant-public` 獨立出來的重構專案。

前一專案包含 RAG 校園知識問答與預約討論室功能。本專案重新整理架構，把重點放在**穩健的輸入處理流程**、**明確的意圖分流**，以及**可維護的 RAG 檢索**，讓每一層都有清楚的職責與可測試的邊界。

---

## 架構規劃

本專案作為校園 Agent 的入口層，目前已實作：

- 輸入前處理（白名單、正規化、語言檢測）。
- 提示詞注入防護層。
- Query Rewriting（指代還原）。
- 意圖路由層。
- 校園資訊 RAG 問答（BM25 + bge-m3 混合檢索）。
- 輸出後處理（簡轉繁與慣用語）。

---

## 目前處理流程

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

在進入意圖路由之前，把帶有指代的問題還原為可獨立理解的查詢（例如把「那裡可以停機車嗎」補齊為「星光大學哪裡可以停機車」）。

原則是有信心才跳過，目前只在兩種情況跳過：

- 對話首輪，沒有可參考的上下文。
- 寒暄、感謝、道別，或以「你」詢問 Agent 本身。

其餘一律呼叫 LLM 改寫，並採用 **JSON 結構化輸出**（先給理由再決策）與 **Few-shot 範例** 提升改寫品質。Prompt 使用 XML 標籤隔離系統指令、歷史與輸入。

### 7. 意圖路由

使用 `semantic-router` 與 `bge-m3` 判斷使用者意圖，分為：

- `campus_info`：校園資訊，走 RAG 檢索。
- `chitchat`：日常閒聊，直接送 LLM。
- `fallback`：無法明確分類，送 LLM。

路由使用改寫後的查詢（`rewritten_query`），因為指代已還原，查詢已自含上下文。

### 8. 校園資訊 RAG 問答

針對 `campus_info` 意圖，使用混合檢索從校園知識庫（`knowledge_base/campus_faq.md`）取得相關內容，再交由 LLM 生成回答：

- **稀疏檢索**：BM25（以 `jieba` 中文分詞，解決中文無空白導致切詞失效的問題）。
- **稠密檢索**：bge-m3 語意向量，存於 Chroma。
- **融合**：以 `EnsembleRetriever` 內建的 RRF（Reciprocal Rank Fusion）融合兩路結果。

檢索輸入直接復用意圖路由前的改寫結果，不另做 RAG 專用改寫。

### 9. 輸出後處理

使用 OpenCC（`s2twp`）將模型輸出統一轉換為繁體中文與台灣慣用詞，避免模型因使用者輸入簡體字而跟著回覆簡體字。

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
| 單元測試 | `pytest` |
| 依賴管理 | `pip-tools` |

---

## 專案結構

```text
campus-agent-entrypoint/
├── knowledge_base/
│   └── campus_faq.md          # 校園知識庫（單一檔案）
├── rag/
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
├── requirements.txt           # 由 pip-compile 自動產生
├── requirements-dev.in        # 手動維護的開發依賴
├── requirements-dev.txt       # 由 pip-compile 自動產生
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

### 環境變數（`.env`）

```env
# 提示詞注入防護開關
# true  : 載入並執行 Qwen3Guard（首次執行較慢）
# false : 停用，跳過模型載入，加速開發測試
PROMPT_GUARD_ENABLED=false

# Query Rewriting 除錯日誌開關
# true  : 在終端機印出指代還原的詳細過程
# false : 關閉，避免日誌洗版
REWRITE_DEBUG=false
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

涵蓋輸入防護、提示詞注入、以及意圖路由的驗證情境。

---

## 已知限制

- 語言檢測目前使用較小規模資料與規則調整參數，短文本、混合語言或特殊用語仍可能誤判。
- 意圖路由目前使用較小規模語料與規則調整參數，對於語意相近但意圖不同的輸入，或較複雜的長句，仍可能誤判。
- 編碼後資料的種類較多，若要處理可能需要多層檢查，且可能產生誤報副作用，因此目前先不納入。

---

## 未來規劃

- **RAG 專用 Query Rewriting**：目前 RAG 復用意圖路由前的改寫結果，未來可針對檢索場景做專屬改寫，提升檢索命中率。
- **校園資訊檢索的 metadata 過濾**：已改用 Chroma 作為向量庫，未來可加上章節、類別等 metadata，支援過濾式檢索。
- **向量庫持久化**：目前採記憶體模式，每次啟動重新建立索引，未來可加持久化避免重複 embedding。
- **對話操作指令**：處理「再說一次」、「第二點展開講」等變體較多的指令，預計走專用 handler 以節省 LLM 成本。
- **日曆型時間解析**：改用 Duckling 將相對時間自動替換為絕對時間。目前未實作，因為整合較費工。
- **輸出安全過濾**：對模型輸出進行毒性、幻覺等進階檢查。

---

## License

Copyright © 2026 hanwu910514.

詳情請參閱[Apache License 2.0](LICENSE)檔案