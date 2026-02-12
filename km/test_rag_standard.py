# -*- coding: utf-8 -*-
"""
標準 RAG 端對端測試腳本

與 test_rag.py 的差異：
- test_rag.py: 透過 /api/v1/query/execute API，上下文為 merged file（整份文件）
- 本腳本: 直接操作 Chroma + 呼叫 LLM，上下文為 similarity_search 的 k 個 chunk 拼接

流程：
1. 用 pdfplumber 將 PDF 轉為 TXT
2. 解析黃金問答
3. 透過 API 匯入文件至 collection
4. 直接用 Chroma similarity_search + LLM API 逐題測試
5. 比對結果並輸出報告
"""
import os
import sys
import requests

from config import settings, get_user_prompt_template
from test_rag import (
    API_BASE,
    COLLECTION_NAME,
    QUERY_TIMEOUT,
    convert_pdfs_to_txt,
    parse_golden_qa,
    import_documents,
    check_answer,
    check_services,
)

# Embedding / Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# ---------------------------------------------------------------------------
# 初始化 Embedding + Chroma
# ---------------------------------------------------------------------------

def init_chroma():
    """建立 OpenAIEmbeddings 並載入 Chroma collection"""
    embedding_model = OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL_NAME,
        base_url=settings.EMBEDDING_API_URL,
        api_key="EMPTY",
        tiktoken_enabled=False,
        check_embedding_ctx_length=False,
        encoding_format="float",
    )

    chroma_path = os.path.join(settings.BASE_FOLDER, COLLECTION_NAME, "chroma_db")
    chroma = Chroma(
        persist_directory=chroma_path,
        embedding_function=embedding_model,
        collection_name=COLLECTION_NAME,
    )
    return chroma


# ---------------------------------------------------------------------------
# Step 0: 前置檢查（LLM + Embedding）
# ---------------------------------------------------------------------------

def check_llm_and_embedding():
    """檢查 LLM server 與 Embedding server 是否可用"""
    # LLM server
    print("[檢查] LLM server...", end=" ", flush=True)
    try:
        r = requests.get(f"{settings.LLM_URL}/models", timeout=5)
        r.raise_for_status()
        print("OK")
    except Exception as e:
        print("FAIL")
        print(f"  無法連線 LLM server ({settings.LLM_URL}): {e}")
        sys.exit(1)

    # Embedding server
    print("[檢查] Embedding server...", end=" ", flush=True)
    try:
        r = requests.get(f"{settings.EMBEDDING_URL}/models", timeout=5)
        r.raise_for_status()
        print("OK")
    except Exception as e:
        print("FAIL")
        print(f"  無法連線 Embedding server ({settings.EMBEDDING_URL}): {e}")
        sys.exit(1)



# ---------------------------------------------------------------------------
# Step 4 & 5: 逐題測試（標準 RAG：chunk 拼接）
# ---------------------------------------------------------------------------

def run_qa_tests_standard(qa_list, k=5):
    """直接用 Chroma similarity_search + LLM API 逐題測試"""
    print("[Step 4] 初始化 Chroma...", end=" ", flush=True)
    chroma = init_chroma()
    print("OK\n")

    prompt_template = get_user_prompt_template(km_lang=settings.KM_LANG, include_query=True)

    print("[Step 5] 開始標準 RAG 測試問答...\n")

    passed = 0
    results = []

    for idx, qa in enumerate(qa_list, 1):
        question = qa["question"]
        expected = qa["expected_answer"]

        print(f"Q{idx}: {question}")
        print(f"   期望: {expected}")

        try:
            # --- 標準 RAG：直接搜尋 chunk ---
            search_results = chroma.similarity_search_with_score(question, k=k)
            chunks = "\n-----------------\n".join(doc.page_content for doc, _score in search_results)
            
            # 組合 prompt
            user_content = prompt_template.format(chunk=chunks, query=question)

            # 呼叫 LLM API
            llm_payload = {
                "model": settings.LLM_MODEL_NAME,
                "messages": [{"role": "user", "content": user_content}],
                "max_tokens": 2048,
                "temperature": 0.7,
                "top_p": 1.0,
                "stream": False,
            }

            headers = {"Content-Type": "application/json"}
            if settings.LLM_API_KEY:
                headers["Authorization"] = f"Bearer {settings.LLM_API_KEY}"

            r = requests.post(
                settings.LLM_API_URL,
                json=llm_payload,
                headers=headers,
                timeout=QUERY_TIMEOUT,
            )
            r.raise_for_status()
            resp = r.json()

            if "choices" in resp and len(resp["choices"]) > 0:
                answer = resp["choices"][0].get("message", {}).get("content", "")
            else:
                answer = f"[異常回應格式] {resp}"

            is_pass = check_answer(answer, expected)

        except Exception as e:
            answer = f"[錯誤] {e}"
            is_pass = False

        if is_pass:
            passed += 1
            mark = "\u2713 PASS"
        else:
            mark = "\u2717 FAIL"

        display_answer = answer[:120].replace("\n", " ")
        if len(answer) > 120:
            display_answer += "..."
        print(f"   回答: {display_answer}")
        print(f"   結果: {mark}\n")

        results.append({
            "index": idx,
            "question": question,
            "expected": expected,
            "answer": answer,
            "passed": is_pass,
        })

    # 總結
    total = len(qa_list)
    rate = (passed / total * 100) if total else 0
    print("=" * 40)
    print("=== 標準 RAG 測試結果 ===")
    print(f"通過: {passed}/{total} ({rate:.1f}%)")
    print("=" * 40)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== 標準 RAG 端對端測試（chunk 拼接） ===\n")

    check_services()
    check_llm_and_embedding()

    txt_paths = convert_pdfs_to_txt()
    qa_list = parse_golden_qa()
    import_documents(txt_paths)
    run_qa_tests_standard(qa_list)


if __name__ == "__main__":
    main()
