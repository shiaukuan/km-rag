# KM RAG 測試指南

## 前置條件

- 已編譯 llama.cpp（`./start_llm.sh install`）
- 已下載模型（`./start_llm.sh download`）
- 已安裝 Python 依賴（`uv sync --python 3.12`）

## 1. 啟動服務

### 方式一：使用 start_llm.sh 一鍵啟動 LLM + Embedding

```bash
cd km/
./start_llm.sh start
```

這會啟動：
- LLM 服務：`http://127.0.0.1:13141`（Qwen2.5-7B, ctx=16384）
- Embedding 服務：`http://127.0.0.1:13142`（bge-m3-f16, ctx=8192）

### 方式二：手動分別啟動

```bash
cd km/

# LLM 服務（port 13141, context size 16384）
./llama.cpp/build/bin/llama-server \
  -m models/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
  --port 13141 -ngl 99 -c 16384 &

# Embedding 服務（port 13142）
./llama.cpp/build/bin/llama-server \
  -m models/bge-m3-f16.gguf \
  --port 13142 -ngl 99 -c 8192 --embedding &
```

### 啟動 KM API

```bash
cd km/
uv run uvicorn api:app --host 0.0.0.0 --port 18299
```

### 驗證服務狀態

```bash
curl http://127.0.0.1:13141/health   # LLM
curl http://127.0.0.1:13142/health   # Embedding
curl http://127.0.0.1:18299/health   # KM API
```

三個都應回傳 `{"status":"ok"}` 或 `{"status":"healthy"}`。

## 2. 執行 RAG 端對端測試

```bash
cd km/
uv run python test_rag.py
```

### 測試流程

1. 檢查 KM API 服務連線
2. 將 `aiDAPTIV_Files/Example Files/Chinense/` 下 5 份 PDF 轉為 TXT
3. 解析 `Golden_Ques_0904 1.txt` 中 18 題黃金問答
4. 透過 API 匯入文件至 collection `test_chinese`
5. 逐題呼叫 RAG 問答 API 並比對結果

### 預期輸出

```
=== KM RAG 端對端測試 ===

[檢查] KM API 服務... OK
[Step 1] PDF 轉 TXT... 5/5 完成
[Step 2] 解析黃金問答... 18 題
[Step 3] 匯入文件至 collection "test_chinese"... 完成
[Step 4] 開始測試問答...

Q1: 公司級獎勵的提名時間是哪一季？
   期望: 第四季，為期一個月。
   回答: ...
   結果: ✓ PASS

...

========================================
=== 測試結果 ===
通過: 17/18 (94.4%)
========================================
```

## 3. 停止服務

```bash
# 停止 LLM + Embedding
./start_llm.sh stop

# 停止 KM API（如在前台執行則 Ctrl+C）
```

## 常見問題

| 問題 | 原因 | 解決方式 |
|------|------|---------|
| `request exceeds the available context size` | LLM context size 不夠 | 啟動 LLM 時使用 `-c 16384` 以上 |
| `Connection refused :13142` | Embedding 服務未啟動 | 重新執行 `./start_llm.sh start` |
| `System is processing another request` | 前一次匯入尚未完成 | 等待或重啟 KM API |
