# KM - RAG 知識管理系統完整指南

## 目錄

- [專案概覽](#專案概覽)
- [技術架構](#技術架構)
- [環境需求與安裝](#環境需求與安裝)
- [資料準備與放置](#資料準備與放置)
- [系統運作流程](#系統運作流程)
- [API 端點參考](#api-端點參考)
- [設定說明](#設定說明)
- [目錄結構](#目錄結構)

---

## 專案概覽

KM 是一套基於 **RAG (Retrieval-Augmented Generation)** 的知識管理系統，核心功能是：

1. 將文件匯入並切分為語義片段
2. 透過向量資料庫 (ChromaDB) 儲存與檢索
3. 結合 LLM 生成精準回答

整體採用 **本地部署** 架構，LLM 與 Embedding 模型皆由 llama.cpp 在本機 GPU 上運行，無需依賴雲端 API。

---

## 技術架構

```
┌─────────────────────────────────────────────────────┐
│                    FastAPI (Port 18299)              │
│                      km/api.py                      │
├──────────┬──────────┬───────────┬───────────────────┤
│ 文件處理  │ RAG 查詢  │ 集合管理   │ KV Cache 生成     │
├──────────┴──────────┴───────────┴───────────────────┤
│                   服務層 (km/services/)              │
│  document_processor / rag_query_service / ...       │
├─────────────────────┬───────────────────────────────┤
│  ChromaDB (向量DB)   │  BM25 Index (關鍵字搜尋)      │
├─────────────────────┴───────────────────────────────┤
│              llama.cpp (本地推論引擎)                  │
│  LLM Server :13141  │  Embedding Server :13142      │
│  Qwen2.5-7B-Q4_K_M  │  BGE-M3-F16                  │
└─────────────────────┴───────────────────────────────┘
```

### 技術清單

| 類別 | 技術 | 說明 |
|------|------|------|
| **LLM 推論** | llama.cpp + CUDA | 本地 GPU 推論，OpenAI 相容 API |
| **LLM 模型** | Qwen2.5-7B-Instruct-Q4_K_M | 量化後約 3.5GB，支援中英日 |
| **Embedding 模型** | BGE-M3-F16 | 多語言嵌入模型，約 3.5GB |
| **向量資料庫** | ChromaDB 1.0.15 | 持久化儲存，檔案系統式管理 |
| **關鍵字搜尋** | BM25 Okapi | 中文 jieba / 英文 / 日文 janome 分詞 |
| **文件處理** | LangChain 0.3.26 | 文件載入、切分、嵌入整合 |
| **API 框架** | FastAPI 0.116.1 | 非同步 REST API |
| **文件解析** | pdfplumber / camelot-py / PyPDF2 | PDF、表格萃取 |
| **容器化** | Docker (多階段建構) | 可選部署方式 |
| **語言** | Python 3.11+ | 主要開發語言 |

### GPU 需求

- 兩個 llama.cpp 服務合計約需 **~7GB VRAM**
- 建議 **16GB 以上 GPU**（如 RTX 4060 Ti 16GB 或更好）
- 若無 NVIDIA GPU，`start_llm.sh` 會自動回退至 CPU 編譯

---

## 環境需求與安裝

### 前置需求

- Python 3.11+
- NVIDIA GPU + CUDA toolkit（建議，非必須）
- cmake、gcc/g++（編譯 llama.cpp）

### 安裝步驟

```bash
# 1. 安裝 Python 依賴
cd km/
pip install -r requirements.txt

# 2. 一鍵安裝 llama.cpp + 下載模型 + 啟動服務
./start_llm.sh

# 也可分步執行：
./start_llm.sh install   # 編譯 llama.cpp（自動偵測 CUDA）
./start_llm.sh download  # 從 HuggingFace 下載模型
./start_llm.sh start     # 啟動 LLM + Embedding 服務

# 3. 啟動 KM API 服務
python -m uvicorn api:app --host 0.0.0.0 --port 18299
```

### Docker 部署（可選）

```bash
cd km/
docker build -t km-app .
docker run -p 18299:18299 \
  -e LLM_URL=http://host.docker.internal:13141/v1 \
  -e EMBEDDING_URL=http://host.docker.internal:13142/v1 \
  km-app
```

> 注意：Docker 內只跑 KM API，llama.cpp 服務需在宿主機上另外啟動。

---

## 資料準備與放置

### 支援的檔案格式

| 來源 | 格式 | 說明 |
|------|------|------|
| **本地檔案** | `.txt` | 直接讀取，UTF-8 編碼（自動回退 GBK） |
| **遠端檔案** | PDF, DOCX, XLSX, CSV, PPTX, HTML, MD | 需搭配外部解析服務 |

### 資料準備方式

#### 方式一：本地 .txt 檔案（最簡單）

將你的知識文件轉為 `.txt` 格式，放在機器上任何可存取的路徑即可：

```
/home/sk/data/
├── 產品說明書.txt
├── FAQ常見問題.txt
├── 技術文件_v2.txt
└── 操作手冊.txt
```

**準備建議：**
- 每個 `.txt` 檔案代表一個獨立主題或文件
- 使用 UTF-8 編碼儲存
- 段落之間用空行分隔，有助於切分品質
- 檔名使用有意義的名稱（會作為 metadata 保留）

#### 方式二：遠端檔案（支援多格式）

透過 S3/MinIO/HTTP URL 提供檔案路徑，系統會呼叫外部解析服務處理：

```
s3://my-bucket/documents/report.pdf
minio://endpoint/bucket/manual.docx
https://example.com/files/data.xlsx
```

> 需要設定 `DOCUMENT_ANALYSIS_URL` 指向外部解析服務。

### 匯入資料的 API 呼叫

```bash
curl -X POST http://localhost:18299/api/v1/process \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "my_knowledge_base",
    "file_list": [
      {"path": "/home/sk/data/產品說明書.txt", "filename": "產品說明書.txt"},
      {"path": "/home/sk/data/FAQ常見問題.txt", "filename": "FAQ常見問題.txt"}
    ],
    "language": "zh-TW"
  }'
```

回應會返回 `task_id`，可輪詢進度：

```bash
curl http://localhost:18299/api/v1/status/{task_id}
```

### 匯入後的資料儲存位置

所有處理後的資料存放在 `BASE_FOLDER`（預設 `./tmp/`）：

```
tmp/
└── my_knowledge_base/           # collection_name
    ├── chroma_db/               # ChromaDB 向量資料庫
    │   ├── chroma.sqlite3
    │   └── *.parquet
    ├── merged_files/            # 合併後的文件內容
    │   ├── 產品說明書.txt
    │   ├── FAQ常見問題.txt
    │   └── 大文件_part1.txt     # 大檔案會自動分割
    └── BM25_indices/            # BM25 關鍵字索引
        └── my_knowledge_base_bm25.pkl
```

### 資料準備最佳實踐

1. **每個知識領域建一個 collection** — 例如 `product_docs`、`customer_faq`、`tech_specs`
2. **文件粒度適中** — 每份 .txt 聚焦一個主題，避免數百頁合成一個檔案
3. **Token 限制注意** — 系統預設每組上限 13,000 tokens，超過會自動分割
4. **結構化內容** — 使用標題、段落、列表等格式，有助提升切分與檢索品質
5. **去除雜訊** — 移除頁首頁尾、頁碼、浮水印文字等無用內容

---

## 系統運作流程

### 文件匯入流程（Ingestion Pipeline）

```
原始文件 (.txt / PDF / DOCX ...)
        │
        ▼
┌─────────────────────┐
│  1. 文件載入         │  SimpleTxtLoader (本地)
│                     │  ExternalParserService (遠端)
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  2. 文件切分         │  RecursiveCharacterTextSplitter
│  chunk_size=512     │  chunk_overlap=102
│  智慧斷句           │  依段落→句子→子句→短語切分
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  3. 向量化 & 儲存    │  BGE-M3 Embedding
│                     │  存入 ChromaDB
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  4. 內容合併         │  小檔案合併 / 大檔案分割
│  Token 管理         │  上限 13,000 tokens/組
│  輸出 merged_files  │
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  5. KV Cache 生成    │  呼叫 LLM API cache_prompt=true
│  (預熱快取)         │  加速後續查詢
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  6. BM25 索引建立    │  jieba 中文分詞
│                     │  儲存 .pkl 索引檔
└─────────────────────┘
```

### RAG 查詢流程（Query Pipeline）

```
使用者提問
    │
    ▼
┌───────────────────────┐
│  1. 載入 ChromaDB      │  從 tmp/{collection}/chroma_db/
└────────┬──────────────┘
         ▼
┌───────────────────────┐
│  2. 相似度搜尋         │  Semantic (預設) 或 BM25
│  取回 Top-K 片段      │  預設 k=5
└────────┬──────────────┘
         ▼
┌───────────────────────┐
│  3. 載入完整內容       │  根據 group_id 讀取 merged_file
│                       │  提供完整上下文給 LLM
└────────┬──────────────┘
         ▼
┌───────────────────────┐
│  4. 組裝 Prompt        │  System Prompt + 文件內容 + 問題
│                       │  語言感知模板
└────────┬──────────────┘
         ▼
┌───────────────────────┐
│  5. LLM 推論          │  Qwen2.5-7B 生成回答
│  (利用 KV Cache)      │  使用預熱的快取加速
└────────┬──────────────┘
         ▼
    回傳答案
```

### 搜尋演算法

| 演算法 | 原理 | 適用場景 |
|--------|------|----------|
| **Semantic (語義搜尋)** | BGE-M3 Embedding 向量相似度 | 預設，理解語義相近的查詢 |
| **BM25 (關鍵字搜尋)** | TF-IDF 變體，精確關鍵字匹配 | Embedding 不可用時的備援，或精確搜尋 |

系統會自動降級：若 Embedding 服務不可用，自動切換至 BM25。

---

## API 端點參考

### 健康檢查與資訊

| 方法 | 路徑 | 說明 |
|------|------|------|
| GET | `/health` | 健康檢查 |
| GET | `/api/v1/info` | API 資訊與支援格式 |
| GET | `/api/v1/gpu/status` | 處理狀態 |

### 文件處理

| 方法 | 路徑 | 說明 |
|------|------|------|
| POST | `/api/v1/process` | 匯入文件（非同步，回傳 task_id） |
| GET | `/api/v1/status/{task_id}` | 查詢匯入進度 |
| POST | `/api/v1/tokens` | 計算檔案 token 數 |
| GET | `/api/v1/tokens/status/{task_id}` | 查詢 token 計算結果 |

### 集合管理

| 方法 | 路徑 | 說明 |
|------|------|------|
| GET | `/api/v1/collections` | 列出所有已建立的集合 |

### RAG 查詢

| 方法 | 路徑 | 說明 |
|------|------|------|
| POST | `/api/v1/query` | 檢索相關內容（不呼叫 LLM） |
| POST | `/api/v1/query/openai` | 產生 OpenAI 格式的 payload |
| POST | `/api/v1/query/execute` | 完整 RAG + LLM 推論，直接回傳答案 |

### 查詢範例

```bash
# 取得相關文件片段（不生成回答）
curl -X POST http://localhost:18299/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "my_knowledge_base",
    "question": "產品的退貨政策是什麼？",
    "k": 5
  }'

# 完整 RAG 問答（直接拿到 LLM 生成的回答）
curl -X POST http://localhost:18299/api/v1/query/execute \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "my_knowledge_base",
    "query": "產品的退貨政策是什麼？",
    "k": 5,
    "language": "zh-TW"
  }'
```

---

## 設定說明

設定檔位於 `km/config.py`，支援環境變數與 `.env` 檔案覆蓋。

### 主要設定項

```bash
# 語言設定（影響 Prompt 模板與分詞）
KM_LANG=zh-TW              # zh-TW | en | ja-JP

# API 服務
API_HOST=0.0.0.0
API_PORT=18299

# LLM 設定
LLM_URL=http://127.0.0.1:13141/v1
LLM_MODEL_NAME=Qwen2.5-7B-Instruct-Q4_K_M
LLM_TYPE=llamacpp           # llamacpp | vllm | openai

# Embedding 設定
EMBEDDING_URL=http://127.0.0.1:13142/v1
EMBEDDING_MODEL_NAME=bge-m3-f16
EMBEDDING_TYPE=llamacpp      # llamacpp | vllm | tei | openai

# 搜尋演算法
SEARCH_ALGORITHM=semantic    # semantic | bm25

# Token 限制（每組合併檔案的上限）
MAX_TOKENS_PER_GROUP=13000

# 資料儲存根目錄
BASE_FOLDER=./tmp

# 外部文件解析服務（處理 PDF/DOCX 等格式）
DOCUMENT_ANALYSIS_URL=http://localhost:8778/api/v2/document_processing/doc_analysis
DOCUMENT_ANALYSIS_API_KEY=

# 自訂 System Prompt（可選）
SYSTEM_PROMPT=
```

### 使用 .env 檔案

在 `km/` 目錄下建立 `.env` 檔案即可覆蓋預設值：

```env
KM_LANG=zh-TW
LLM_URL=http://127.0.0.1:13141/v1
SEARCH_ALGORITHM=semantic
BASE_FOLDER=./tmp
```

### llama.cpp 服務設定

在 `start_llm.sh` 中可調整：

```bash
LLM_PORT=13141          # LLM 服務埠
EMBED_PORT=13142        # Embedding 服務埠
LLM_CTX=8192            # LLM context 長度
LLM_GPU_LAYERS=99       # GPU offload 層數（99 = 全部）
EMBED_GPU_LAYERS=99     # Embedding GPU offload 層數
```

---

## 目錄結構

```
km/
├── KM_PROJECT_GUIDE.md          # 本文件
├── aiDAPTIV_Files/              # 使用者指南與範例檔案
│   ├── Example Files/
│   ├── User_Guide/
│   └── installer/
└── km/                          # 主程式目錄
    ├── api.py                   # FastAPI 主入口 (Port 18299)
    ├── config.py                # Pydantic 設定檔
    ├── Dockerfile               # Docker 多階段建構
    ├── pyproject.toml           # 專案 metadata 與依賴
    ├── requirements.txt         # Python 依賴清單
    ├── start_llm.sh             # llama.cpp 一鍵安裝/啟動腳本
    ├── models/                  # 模型快取目錄
    │   ├── Qwen2.5-7B-Instruct-Q4_K_M.gguf
    │   └── bge-m3-f16.gguf
    ├── llama.cpp/               # llama.cpp 原始碼（編譯用）
    ├── tmp/                     # 資料儲存根目錄 (BASE_FOLDER)
    │   └── {collection_name}/
    │       ├── chroma_db/       # ChromaDB 向量資料庫
    │       ├── merged_files/    # 合併後的文件
    │       └── BM25_indices/    # BM25 索引
    └── services/                # 服務模組
        ├── api.py               # 服務層 API
        ├── task_manager.py      # 工作流程管理
        ├── rag_query_service.py # RAG 查詢服務
        ├── document_processor.py # 文件處理
        ├── external_parser.py   # 外部解析服務
        ├── simple_txt_loader.py # 本地 .txt 載入器
        ├── kvcache_generator.py # KV Cache 生成
        ├── kv_cache_content.py  # ChromaDB 內容處理
        ├── bm25_index_manager.py # BM25 索引管理
        └── lib/
            ├── chunk_manager.py     # 文件切分邏輯
            └── tokenizer_manager.py # Token 計數
```

---

## 快速開始總結

```bash
# Step 1: 安裝依賴
cd km/ && pip install -r requirements.txt

# Step 2: 一鍵啟動 LLM 服務（編譯 + 下載模型 + 啟動）
./start_llm.sh

# Step 3: 啟動 KM API
python -m uvicorn api:app --host 0.0.0.0 --port 18299

# Step 4: 準備 .txt 資料並匯入
curl -X POST http://localhost:18299/api/v1/process \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "my_docs",
    "file_list": [
      {"path": "/path/to/your/document.txt", "filename": "document.txt"}
    ],
    "language": "zh-TW"
  }'

# Step 5: 開始問答
curl -X POST http://localhost:18299/api/v1/query/execute \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "my_docs",
    "query": "你的問題",
    "language": "zh-TW"
  }'
```
