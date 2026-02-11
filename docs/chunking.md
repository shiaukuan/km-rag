# 文件切塊策略 (_chunk_documents)

## 概覽

系統使用 LangChain 的 `RecursiveCharacterTextSplitter`，以 **token 數**（非字元數）為單位進行切塊，搭配多層級的正則分隔符，確保切割點盡量落在語意完整的位置。

## 預設參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `chunk_size` | 512 tokens | 每個 chunk 的最大 token 數 |
| `chunk_overlap` | 102 tokens | 相鄰 chunk 之間的重疊 token 數 |
| `max_tokens_per_group` | 13000 tokens | KV Cache 處理時每組的最大 token 數 |
| `is_separator_regex` | True | 分隔符使用正則表達式 |

設定位於 `km/services/document_processor.py` 的 `ProcessingConfig` dataclass。

## Token 計算方式

使用 `count_tokens_embedding()` 函式，透過 Embedding server 的 `/tokenize` API 實際計算 token 數量（非估算），最後加 2（BOS & EOS token）。

支援的 Embedding 後端：
- `llamacpp`：payload 使用 `content` 欄位
- `vllm`：payload 使用 `prompt` 欄位

## 分隔符層級

切割時按以下優先順序嘗試分隔符，優先在高層級（語意完整性高）的位置切割：

### Level 1：段落與結構（最優先）

| 分隔符 | 說明 |
|--------|------|
| `\n\n` | 雙換行（段落分隔） |
| `\n` | 單換行 |

### Level 2：句子結束（語意完整）

| 分隔符 | 說明 |
|--------|------|
| `。` | 中文句號 |
| `\.(?!\d)` | 英文句號（排除小數點如 3.14） |
| `！` | 中文驚嘆號 |
| `\!` | 英文驚嘆號 |
| `？` | 中文問號 |
| `\?` | 英文問號 |

### Level 3：子句與語氣

| 分隔符 | 說明 |
|--------|------|
| `；` | 中文分號 |
| `;` | 英文分號 |
| `：` | 中文冒號 |
| `:` | 英文冒號 |

### Level 4：短語與列表

| 分隔符 | 說明 |
|--------|------|
| `\|` | 直線符號 |
| `，` | 中文逗號 |
| `,(?!\d)` | 英文逗號（排除千分位如 1,000） |
| `、` | 頓號 |

### Level 5：單字邊界（最後手段）

| 分隔符 | 說明 |
|--------|------|
| `\s+` | 所有空白字元 |
| `""` | 逐字切割（fallback） |

## Chunk ID 命名規則

每個 chunk 的 ID 格式為：`{filename}_chunk_{index}`

- 檔名中的 `.` 會被替換為 `_`
- index 從 1 開始，按檔案內順序遞增

## Metadata

每個 chunk 保留原始文件的所有 metadata，並額外加入：

| 欄位 | 說明 |
|------|------|
| `chunk_id` | chunk 唯一識別碼 |
| `chunk_index` | 該檔案內的 chunk 序號 |
