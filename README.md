# RAG 電影字幕測試說明

用 IMDB 排名前 5 名電影（刺激1995、教父、黑暗騎士、教父II、十二怒漢）的中文字幕作為知識庫，搭配 15 題黃金問答，比較「最傳統的標準 RAG」與「本專案設計的 RAG」效果差異。

---

## 啟動服務

### Step 1：啟動 LLM + Embedding Server

```bash
cd km
./start_llm.sh
```

一鍵完成：編譯 llama.cpp → 下載模型 → 啟動服務。

| 服務 | 模型 | Port | 用途 |
|------|------|------|------|
| LLM | Qwen2.5-7B-Instruct-Q4_K_M.gguf | 13141 | 文字生成 |
| Embedding | bge-m3-f16.gguf | 13142 | 向量嵌入 |

總 VRAM 約 ~7GB，適合 16GB GPU。也可分步執行：

```bash
./start_llm.sh install    # 只編譯 llama.cpp
./start_llm.sh download   # 只下載模型
./start_llm.sh start      # 只啟動服務
./start_llm.sh stop       # 停止所有服務
```

### Step 2：安裝 Python 依賴

```bash
cd km
uv sync
```

### Step 3：啟動 KM API 服務（僅 test_rag_movie.py 需要）

```bash
cd km
uv run uvicorn api:app --host 0.0.0.0 --port 18299
```

> 標準 RAG 測試（test_rag_standard_movie.py）**不需要**這個服務，它直接操作 Chroma + LLM。

---

## 執行測試

```bash
cd km

# 標準 RAG（只需 LLM + Embedding server）
uv run python test_rag_standard_movie.py

# 專案 RAG（需要 LLM + Embedding + KM API 服務）
uv run python test_rag_movie.py
```

---

## 標準 RAG 到底多簡單？（test_rag_standard_movie.py）

**這是教科書級的 naive RAG，整個查詢流程只有 3 行核心程式碼：**

```python
# 1. 向量搜尋 — 用 Chroma 找最相似的 5 個 chunk
search_results = chroma.similarity_search_with_score(question, k=5)

# 2. 拼接 — 把 chunk 直接拼起來當上下文
chunks = "\n-----------------\n".join(doc.page_content for doc, _score in search_results)

# 3. 丟給 LLM — 組 prompt 然後 POST
user_content = prompt_template.format(chunk=chunks, query=question)
r = requests.post(LLM_API_URL, json=payload)
```

就這樣，沒了。

### 全部使用系統預設，零調參

| 參數 | 值 | 來源 |
|------|------|------|
| Embedding 模型 | bge-m3-f16 | config.py 預設 |
| LLM 模型 | Qwen2.5-7B-Instruct-Q4_K_M | config.py 預設 |
| 搜尋結果數 k | 5 | 預設值 |
| temperature | 0.7 | 預設值 |
| max_tokens | 2048 | 預設值 |
| Chunk 切分 | 系統預設 | 匯入時由 KM API 處理 |
| Prompt 模板 | config.get_user_prompt_template() | 系統內建 |

### 沒有做的事

- 沒有 Rerank
- 沒有 Query Rewriting
- 沒有 HyDE
- 沒有 Multi-Query
- 沒有 Fusion Retrieval
- 沒有任何上下文後處理

**就是「搜 → 拼 → 問」三步走，連工具函式都直接從 test_rag_movie.py import 複用**（load_txt_files、parse_golden_qa、import_documents、check_answer），自己只寫了 init_chroma 和 run_qa_tests_standard 兩個函式。

---

## 專案 RAG 做了什麼不同？（test_rag_movie.py）

透過 KM 系統的 `/api/v1/query/execute` API 進行查詢：

```python
payload = {
    "collection_name": COLLECTION_NAME,
    "query": question,
    "k": 5,
    "language": "zh-TW",
}
r = requests.post(f"{API_BASE}/api/v1/query/execute", json=payload)
```

系統內部會使用 **merged file（整份文件）** 作為上下文，而非零散的 chunk 拼接，經過完整的文件處理管線。

### 架構對照

```
標準 RAG:
  問題 → Chroma similarity_search → k 個 chunk 拼接 → LLM
                                     ↑
                              零散片段，可能斷裂

專案 RAG:
  問題 → KM API → merged file（整份文件）→ LLM
                   ↑
            完整上下文，前後連貫
```

| | 標準 RAG | 專案 RAG |
|---|---|---|
| 上下文來源 | similarity_search 的 k 個 **chunk 拼接** | **merged file**（整份文件） |
| 呼叫方式 | 直接 Chroma + LLM API | `/api/v1/query/execute` |
| 前置服務 | LLM + Embedding | LLM + Embedding + KM API |
| 程式複雜度 | 3 行核心邏輯 | 系統內部管線 |

---

## 測試結果

把兩份測試輸出丟給 ChatGPT 評分（15 題）：

| | 專案 RAG（第一份） | 標準 RAG（第二份） |
|---|---|---|
| 正確 | 6 | 8 |
| 部分正確 | 3 | 3 |
| 錯誤 | 6 | 4 |
| **換算分數** | **7.5 / 15** | **9.5 / 15** |



### 關鍵發現

標準 RAG 在本測試中反而得分較高（9.5 vs 7.5），原因可能是：

- 電影字幕本身就是短句結構，chunk 切分後仍保有局部語境
- 標準 RAG 的 similarity_search 更精準地定位到答案所在的 chunk
- merged file 策略在長篇文件（如政策文件、技術手冊）中優勢更明顯，但在字幕這類短句場景中差異較小

### 詳細比較表

| 題號 | 專案 RAG | 標準 RAG | 差異說明 |
|---|---|---|---|
| Q1 | 正確 | 部分正確 | 專案精準說出「銀行副總裁」，標準只說「銀行家」 |
| Q2 | 錯誤 | 錯誤 | 兩份都答錯，正解是鶴嘴槌 |
| Q3 | 正確 | 正確 | 都抓到「兩週禁閉＋值得」 |
| Q4 | 錯誤 | 錯誤 | 都答非所問 |
| Q5 | 部分正確 | 部分正確 | 都說對軍師身份，但漏掉「被收養」 |
| Q6 | 正確 | 正確 | 都命中「槍藏在馬桶水箱」 |
| Q7 | 部分正確 | 部分正確 | 都提到父兄被殺，缺完整動機鏈 |
| Q8 | 正確 | 正確 | 都抓到核心 |
| Q9 | 錯誤 | 正確 | 標準命中「25萬＋5%」，專案未提金額 |
| Q10 | 錯誤 | 錯誤 | 兩份都編出不存在劇情 |
| Q11 | 錯誤 | 正確 | 標準命中「父親幸運幣＋約會」 |
| Q12 | 正確 | 錯誤 | 專案說對「裝甲太重需要更快」 |
| Q13 | 錯誤 | 正確 | 標準票數正確 |
| Q14 | 正確 | 正確 | 都抓到「買同款刀」 |
| Q15 | 部分正確 | 正確 | 標準完整命中質疑點 |

---

## 測試資料

```
movie/
├── chinese/                              # 純文字字幕（測試用）
│   ├── 12_Angry_Men_1957.txt
│   ├── The_Dark_Knight_2008.txt
│   ├── The_Godfather_1972.txt
│   ├── The_Godfather_Part_II_1974.txt
│   └── The_Shawshank_Redemption_1994.txt
├── Movie_QA.txt                          # 黃金問答（15 題）
├── 12_Angry_Men_1957.zh-TW.srt          # 原始 SRT 字幕
├── The_Dark_Knight_2008.zh-TW.srt
├── The_Godfather_1972.zh-TW.srt
├── The_Godfather_Part_II_1974.zh-TW.srt
└── The_Shawshank_Redemption_1994.zh-TW.srt
```


# 測試程式

## 標準 RAG
km/test_rag_standard_movie.py

## 專案 RAG
km/test_rag_movie.py