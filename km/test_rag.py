# -*- coding: utf-8 -*-
"""
KM RAG 端對端測試腳本

流程：
1. 用 pdfplumber 將 Chinense/ 下 5 份 PDF 轉為 TXT
2. 解析 Golden_Ques_0904 1.txt 黃金問答
3. 透過 API 匯入文件至 collection
4. 逐題測試 RAG 問答
5. 比對結果並輸出報告
"""
import os
import re
import sys
import time
import unicodedata
import requests
import pdfplumber

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------
API_BASE = "http://127.0.0.1:18299"
COLLECTION_NAME = "test_chinese"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

PDF_DIR = os.path.join(PROJECT_ROOT, "aiDAPTIV_Files", "Example Files", "Chinense")
GOLDEN_QA_FILE = os.path.join(PROJECT_ROOT, "aiDAPTIV_Files", "Example Files", "Golden_Ques_0904 1.txt")
TXT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "tmp", "_test_txt")

POLL_INTERVAL = 3        # 輪詢間隔（秒）
POLL_TIMEOUT = 600       # 輪詢逾時（秒）
QUERY_TIMEOUT = 300      # 單題查詢逾時（秒）


# ---------------------------------------------------------------------------
# Step 0: 前置檢查
# ---------------------------------------------------------------------------
def check_services():
    """檢查 KM API 服務是否可用"""
    print("[檢查] KM API 服務...", end=" ", flush=True)
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        r.raise_for_status()
        print("OK")
    except Exception as e:
        print("FAIL")
        print(f"  無法連線 KM API ({API_BASE}/health): {e}")
        print("  請確認已啟動：python -m uvicorn api:app --host 0.0.0.0 --port 18299")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Step 1: PDF → TXT
# ---------------------------------------------------------------------------
def convert_pdfs_to_txt():
    """將 PDF 轉為 TXT，回傳 TXT 檔案路徑列表"""
    print("[Step 1] PDF 轉 TXT...", end=" ", flush=True)

    os.makedirs(TXT_OUTPUT_DIR, exist_ok=True)

    pdf_files = sorted(f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf"))
    if not pdf_files:
        print("FAIL — 找不到 PDF 檔案")
        sys.exit(1)

    txt_paths = []
    for pdf_name in pdf_files:
        pdf_path = os.path.join(PDF_DIR, pdf_name)
        txt_name = os.path.splitext(pdf_name)[0] + ".txt"
        txt_path = os.path.join(TXT_OUTPUT_DIR, txt_name)

        with pdfplumber.open(pdf_path) as pdf:
            pages_text = []
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages_text.append(text)
            content = "\n\n".join(pages_text)

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(content)

        txt_paths.append(txt_path)

    print(f"{len(txt_paths)}/{len(pdf_files)} 完成")
    return txt_paths


# ---------------------------------------------------------------------------
# Step 2: 解析黃金問答
# ---------------------------------------------------------------------------
def parse_golden_qa():
    """
    解析黃金問答檔案，回傳:
    [{"question": str, "expected_answer": str, "source_pdf": str}, ...]
    """
    print("[Step 2] 解析黃金問答...", end=" ", flush=True)

    with open(GOLDEN_QA_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    qa_list = []
    current_source = ""
    i = 0
    while i < len(lines):
        line = lines[i].rstrip("\n").strip()

        # 來源文件標頭
        m = re.match(r"^=====\s*(.+?)\s*=====$", line)
        if m:
            current_source = m.group(1).strip()
            i += 1
            continue

        # 空行跳過
        if not line:
            i += 1
            continue

        # 問題行（非 → 開頭）
        if not line.startswith("→"):
            question = line
            # 下一行應為答案（→ 開頭）
            i += 1
            while i < len(lines):
                ans_line = lines[i].rstrip("\n").strip()
                if ans_line.startswith("→"):
                    expected_answer = ans_line.lstrip("→").strip()
                    qa_list.append({
                        "question": question,
                        "expected_answer": expected_answer,
                        "source_pdf": current_source,
                    })
                    i += 1
                    break
                elif ans_line == "":
                    i += 1
                    continue
                else:
                    # 沒有找到答案行，跳過
                    break
            continue

        i += 1

    print(f"{len(qa_list)} 題")
    return qa_list


# ---------------------------------------------------------------------------
# Step 3: 匯入文件至 collection
# ---------------------------------------------------------------------------
def import_documents(txt_paths):
    """透過 API 匯入 TXT 文件"""
    print(f'[Step 3] 匯入文件至 collection "{COLLECTION_NAME}"...', end=" ", flush=True)

    file_list = []
    for txt_path in txt_paths:
        abs_path = os.path.abspath(txt_path)
        filename = os.path.basename(abs_path)
        file_list.append({"path": abs_path, "filename": filename})

    payload = {
        "collection_name": COLLECTION_NAME,
        "file_list": file_list,
        "language": "zh-TW",
    }

    r = requests.post(f"{API_BASE}/api/v1/process", json=payload, timeout=30)
    if r.status_code != 200:
        print(f"FAIL — HTTP {r.status_code}: {r.text}")
        sys.exit(1)

    task_id = r.json()["task_id"]

    # 輪詢等待完成
    elapsed = 0
    while elapsed < POLL_TIMEOUT:
        time.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL
        sr = requests.get(f"{API_BASE}/api/v1/status/{task_id}", timeout=10)
        status_data = sr.json()
        status = status_data["status"]
        if status == "completed":
            print("完成")
            return
        elif status == "failed":
            print(f"FAIL — {status_data.get('message', 'unknown error')}")
            sys.exit(1)

    print(f"FAIL — 逾時 ({POLL_TIMEOUT}s)")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Step 4 & 5: 逐題測試 + 結果比對
# ---------------------------------------------------------------------------
def run_qa_tests(qa_list):
    """逐題呼叫 RAG 問答 API 並比對結果"""
    print("[Step 4] 開始測試問答...\n")

    passed = 0
    results = []

    for idx, qa in enumerate(qa_list, 1):
        question = qa["question"]
        expected = qa["expected_answer"]

        print(f"Q{idx}: {question}")
        print(f"   期望: {expected}")

        payload = {
            "collection_name": COLLECTION_NAME,
            "query": question,
            "k": 5,
            "language": "zh-TW",
        }

        try:
            r = requests.post(
                f"{API_BASE}/api/v1/query/execute",
                json=payload,
                timeout=QUERY_TIMEOUT,
            )
            r.raise_for_status()
            resp = r.json()

            if not resp.get("success"):
                answer = f"[API 失敗] {resp.get('message', '')}"
                is_pass = False
            else:
                answer = resp.get("model_response", "")
                # 關鍵字比對：將期望答案拆成片段，檢查 model_response 是否包含
                is_pass = check_answer(answer, expected)

        except Exception as e:
            answer = f"[錯誤] {e}"
            is_pass = False

        if is_pass:
            passed += 1
            mark = "\u2713 PASS"
        else:
            mark = "\u2717 FAIL"

        # 截短顯示回答
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
    print(f"=== 測試結果 ===")
    print(f"通過: {passed}/{total} ({rate:.1f}%)")
    print("=" * 40)

    return results


# ---------------------------------------------------------------------------
# 模糊比對：簡→繁正規化 + 字元 bigram 重疊
# ---------------------------------------------------------------------------

# 常用簡體→繁體對照（涵蓋 HR / 商務常見用字）
_S2T_MAP = {
    '个': '個', '为': '為', '义': '義', '习': '習', '书': '書',
    '买': '買', '亲': '親', '产': '產', '亿': '億', '从': '從',
    '众': '眾', '优': '優', '传': '傳', '体': '體', '关': '關',
    '兴': '興', '决': '決', '准': '準', '则': '則', '创': '創',
    '别': '別', '办': '辦', '动': '動', '务': '務', '劳': '勞',
    '区': '區', '华': '華', '单': '單', '卖': '賣', '厂': '廠',
    '压': '壓', '发': '發', '变': '變', '号': '號', '听': '聽',
    '启': '啟', '员': '員', '团': '團', '围': '圍', '国': '國',
    '图': '圖', '场': '場', '处': '處', '复': '復', '够': '夠',
    '头': '頭', '奖': '獎', '奋': '奮', '学': '學', '宝': '寶',
    '实': '實', '对': '對', '导': '導', '层': '層', '岁': '歲',
    '岗': '崗', '属': '屬', '带': '帶', '帮': '幫', '广': '廣',
    '庆': '慶', '应': '應', '开': '開', '异': '異', '张': '張',
    '当': '當', '录': '錄', '总': '總', '态': '態', '战': '戰',
    '护': '護', '报': '報', '择': '擇', '损': '損', '换': '換',
    '据': '據', '断': '斷', '无': '無', '时': '時', '显': '顯',
    '晋': '晉', '术': '術', '机': '機', '杂': '雜', '权': '權',
    '条': '條', '来': '來', '杰': '傑', '极': '極', '构': '構',
    '档': '檔', '检': '檢', '欢': '歡', '毕': '畢', '没': '沒',
    '测': '測', '灭': '滅', '灵': '靈', '热': '熱', '爱': '愛',
    '状': '狀', '独': '獨', '环': '環', '现': '現', '确': '確',
    '离': '離', '种': '種', '积': '積', '称': '稱', '稳': '穩',
    '竞': '競', '笔': '筆', '签': '簽', '简': '簡', '纪': '紀',
    '纯': '純', '纳': '納', '纷': '紛', '练': '練', '组': '組',
    '细': '細', '终': '終', '经': '經', '结': '結', '给': '給',
    '绍': '紹', '续': '續', '维': '維', '绩': '績', '综': '綜',
    '编': '編', '织': '織', '罚': '罰', '网': '網', '见': '見',
    '规': '規', '观': '觀', '计': '計', '认': '認', '讨': '討',
    '训': '訓', '设': '設', '许': '許', '词': '詞', '该': '該',
    '详': '詳', '说': '說', '调': '調', '谈': '談', '识': '識',
    '资': '資', '质': '質', '费': '費', '车': '車', '转': '轉',
    '软': '軟', '边': '邊', '较': '較', '输': '輸', '达': '達',
    '过': '過', '远': '遠', '还': '還', '连': '連', '运': '運',
    '进': '進', '选': '選', '释': '釋', '错': '錯', '长': '長',
    '门': '門', '间': '間', '闻': '聞', '阅': '閱', '阶': '階',
    '际': '際', '险': '險', '随': '隨', '难': '難', '须': '須',
    '预': '預', '题': '題', '额': '額', '飞': '飛', '验': '驗',
    '龙': '龍', '举': '舉', '仅': '僅', '历': '歷', '归': '歸',
    '满': '滿', '营': '營', '补': '補', '课': '課', '谁': '誰',
    '将': '將', '药': '藥', '节': '節', '范': '範', '读': '讀',
    '误': '誤', '让': '讓', '议': '議', '证': '證', '评': '評',
    '试': '試', '请': '請', '负': '負', '财': '財', '贵': '貴',
    '龄': '齡', '参': '參', '双': '雙', '协': '協', '卫': '衛',
    '币': '幣', '师': '師', '户': '戶', '挥': '揮', '统': '統',
    '尽': '盡', '汇': '匯', '沟': '溝', '残': '殘', '涌': '湧',
    '园': '園', '厅': '廳', '寿': '壽', '尝': '嘗',
}
_S2T_TRANS = str.maketrans(_S2T_MAP)


def _normalize(text: str) -> str:
    """去除標點、空白，並將簡體轉為繁體。"""
    out = []
    for ch in text:
        if unicodedata.category(ch)[0] in ('P', 'Z') or ch in ' \t\n\r':
            continue
        out.append(ch)
    return ''.join(out).translate(_S2T_TRANS)


def check_answer(model_response: str, expected_answer: str) -> bool:
    """
    模糊比對：將期望答案與模型回答正規化後，
    計算期望答案的字元 bigram 在回答中出現的比例。
    比例 >= 40% 即視為通過。
    """
    resp = _normalize(model_response)
    exp = _normalize(expected_answer)

    if len(exp) < 2:
        return exp in resp

    exp_bigrams = set(exp[i:i+2] for i in range(len(exp) - 1))
    if not exp_bigrams:
        return False

    matched = sum(1 for bg in exp_bigrams if bg in resp)
    return matched / len(exp_bigrams) >= 0.4


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=== KM RAG 端對端測試 ===\n")

    check_services()

    txt_paths = convert_pdfs_to_txt()
    qa_list = parse_golden_qa()
    import_documents(txt_paths)
    run_qa_tests(qa_list)


if __name__ == "__main__":
    main()
