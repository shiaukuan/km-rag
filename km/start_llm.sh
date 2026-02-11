#!/bin/bash
#   # 一鍵完成：編譯 llama.cpp + 下載模型 + 啟動服務
#   ./start_llm.sh                                                                                                                                                                                                                           
                                                                                                                                                                                                                                         
#   # 或分步執行
#   ./start_llm.sh install    # 只編譯 llama.cpp
#   ./start_llm.sh download   # 只下載模型
#   ./start_llm.sh start      # 只啟動服務
#   ./start_llm.sh stop       # 停止所有服務

set -e

# ============================================================
#  LLM + Embedding 服務啟動腳本 (llama.cpp)
#  總 VRAM 用量約 ~7GB (適合 16GB GPU)
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="${SCRIPT_DIR}/models"
LLAMA_CPP_DIR="${SCRIPT_DIR}/llama.cpp"
LLAMA_SERVER="${LLAMA_CPP_DIR}/build/bin/llama-server"

# --- 模型設定 ---
LLM_MODEL_REPO="bartowski/Qwen2.5-7B-Instruct-GGUF"
LLM_MODEL_FILE="Qwen2.5-7B-Instruct-Q4_K_M.gguf"
LLM_PORT=13141
LLM_CTX=16384        # context length
LLM_GPU_LAYERS=99   # 全部放 GPU

EMBED_MODEL_REPO="CompendiumLabs/bge-m3-gguf"
EMBED_MODEL_FILE="bge-m3-f16.gguf"
EMBED_PORT=13142
EMBED_CTX=8192
EMBED_GPU_LAYERS=99

# ============================================================
#  1. 安裝 llama.cpp (如果尚未編譯)
# ============================================================
install_llama_cpp() {
    if [ -f "$LLAMA_SERVER" ]; then
        echo "[OK] llama-server 已存在: $LLAMA_SERVER"
        return
    fi

    echo "[INFO] 開始編譯 llama.cpp (CUDA) ..."

    # 確認基本編譯工具
    if ! command -v cmake &>/dev/null; then
        echo "[ERROR] 請先安裝 cmake: sudo apt install cmake"
        exit 1
    fi
    if ! command -v nvcc &>/dev/null; then
        echo "[WARN] 未偵測到 nvcc，將嘗試以 CPU 模式編譯"
        echo "       如需 GPU 加速請安裝 CUDA toolkit: sudo apt install nvidia-cuda-toolkit"
    fi

    git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$LLAMA_CPP_DIR" 2>/dev/null || true
    cd "$LLAMA_CPP_DIR"

    cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release 2>/dev/null || \
    cmake -B build -DCMAKE_BUILD_TYPE=Release  # fallback: CPU only

    cmake --build build --config Release -j "$(nproc)" --target llama-server

    cd "$SCRIPT_DIR"
    echo "[OK] llama-server 編譯完成"
}

# ============================================================
#  2. 下載模型 (使用 huggingface-cli)
# ============================================================
download_models() {
    mkdir -p "$MODELS_DIR"

    # 確認 hf CLI
    if ! command -v hf &>/dev/null; then
        echo "[INFO] 安裝 huggingface_hub CLI ..."
        uv tool install huggingface_hub
    fi

    # 下載 LLM 模型
    if [ ! -f "${MODELS_DIR}/${LLM_MODEL_FILE}" ]; then
        echo "[INFO] 下載 LLM 模型: ${LLM_MODEL_REPO} / ${LLM_MODEL_FILE} ..."
        hf download "$LLM_MODEL_REPO" "$LLM_MODEL_FILE" \
            --local-dir "$MODELS_DIR"
    else
        echo "[OK] LLM 模型已存在: ${MODELS_DIR}/${LLM_MODEL_FILE}"
    fi

    # 下載 Embedding 模型
    if [ ! -f "${MODELS_DIR}/${EMBED_MODEL_FILE}" ]; then
        echo "[INFO] 下載 Embedding 模型: ${EMBED_MODEL_REPO} / ${EMBED_MODEL_FILE} ..."
        hf download "$EMBED_MODEL_REPO" "$EMBED_MODEL_FILE" \
            --local-dir "$MODELS_DIR"
    else
        echo "[OK] Embedding 模型已存在: ${MODELS_DIR}/${EMBED_MODEL_FILE}"
    fi
}

# ============================================================
#  3. 啟動服務
# ============================================================
start_services() {
    echo ""
    echo "========================================"
    echo "  啟動 LLM 服務 (port ${LLM_PORT})"
    echo "  模型: ${LLM_MODEL_FILE}"
    echo "========================================"
    $LLAMA_SERVER \
        -m "${MODELS_DIR}/${LLM_MODEL_FILE}" \
        --port "$LLM_PORT" \
        -ngl "$LLM_GPU_LAYERS" \
        -c "$LLM_CTX" \
        &
    LLM_PID=$!

    echo ""
    echo "========================================"
    echo "  啟動 Embedding 服務 (port ${EMBED_PORT})"
    echo "  模型: ${EMBED_MODEL_FILE}"
    echo "========================================"
    $LLAMA_SERVER \
        -m "${MODELS_DIR}/${EMBED_MODEL_FILE}" \
        --port "$EMBED_PORT" \
        -ngl "$EMBED_GPU_LAYERS" \
        -c "$EMBED_CTX" \
        --embedding \
        &
    EMBED_PID=$!

    echo ""
    echo "========================================"
    echo "  服務已啟動"
    echo "  LLM:       http://127.0.0.1:${LLM_PORT}/v1  (PID: ${LLM_PID})"
    echo "  Embedding:  http://127.0.0.1:${EMBED_PORT}/v1  (PID: ${EMBED_PID})"
    echo ""
    echo "  按 Ctrl+C 停止所有服務"
    echo "========================================"

    # 捕捉 Ctrl+C 停止所有背景程序
    trap "echo '正在停止服務...'; kill $LLM_PID $EMBED_PID 2>/dev/null; exit 0" INT TERM
    wait
}

# ============================================================
#  主流程
# ============================================================
case "${1:-}" in
    install)
        install_llama_cpp
        ;;
    download)
        download_models
        ;;
    start)
        start_services
        ;;
    stop)
        echo "停止所有 llama-server 程序..."
        pkill -f llama-server || echo "沒有正在運行的 llama-server"
        ;;
    *)
        install_llama_cpp
        download_models
        start_services
        ;;
esac
