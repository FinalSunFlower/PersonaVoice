#!/usr/bin/env bash
# ============================================================
# PersonaVoice v10.4.8 — 一键环境搭建脚本 (Linux/macOS)
# ============================================================
# 用法:
#   bash scripts/setup_env.sh             # 默认 CUDA 11.8
#   bash scripts/setup_env.sh --cpu       # CPU 版本
#   bash scripts/setup_env.sh --skip-models  # 跳过模型下载
# ============================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

USE_CPU=0
SKIP_MODELS=0

for arg in "$@"; do
    case "$arg" in
        --cpu) USE_CPU=1 ;;
        --skip-models) SKIP_MODELS=1 ;;
    esac
done

echo ""
echo "============================================================"
echo "  PersonaVoice v10.4.8 环境搭建 (Linux/macOS)"
echo "============================================================"
echo ""

# ── 1. 检查 Python ──
echo "[1/6] 检查 Python ..."
if ! command -v python3 &> /dev/null; then
    echo "  [ERROR] 未找到 python3, 请先安装 Python 3.10+"
    exit 1
fi
PYTHON_VERSION=$(python3 --version)
echo "  ✓ $PYTHON_VERSION"

# ── 2. 创建虚拟环境 ──
echo ""
echo "[2/6] 创建虚拟环境 .venv ..."
if [ -d ".venv" ]; then
    echo "  .venv 已存在, 跳过创建"
else
    python3 -m venv --system-site-packages .venv
    echo "  ✓ 虚拟环境创建完成"
fi

# ── 3. 升级 pip ──
echo ""
echo "[3/6] 升级 pip ..."
./.venv/bin/python -m pip install --upgrade pip
echo "  ✓ pip 升级完成"

# ── 4. 安装 PyTorch ──
echo ""
echo "[4/6] 安装 PyTorch ..."
if [ "$USE_CPU" = "1" ]; then
    echo "  安装 CPU 版本 (--cpu)"
    ./.venv/bin/pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
else
    echo "  安装 CUDA 11.8 版本 (默认)"
    echo "  如需 CPU 版本, 请加 --cpu 参数"
    ./.venv/bin/pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
fi
echo "  ✓ PyTorch 安装完成"

# ── 5. 安装 PersonaVoice 依赖 ──
echo ""
echo "[5/6] 安装 PersonaVoice 依赖 ..."
./.venv/bin/pip install -r requirements.txt
./.venv/bin/pip install silero-vad f5-tts imageio-ffmpeg
./.venv/bin/pip install -e .
echo "  ✓ 依赖安装完成"

# ── 6. 验证安装 ──
echo ""
echo "[6/6] 验证安装 ..."
./.venv/bin/python -c "import torch; print(f'  ✓ PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
./.venv/bin/python -c "import personavoice; print(f'  ✓ PersonaVoice {personavoice.__version__}')"
./.venv/bin/python -c "from f5_tts.api import F5TTS; print('  ✓ F5-TTS')"
./.venv/bin/python -c "import speechbrain; print(f'  ✓ SpeechBrain {speechbrain.__version__}')"
./.venv/bin/python -c "import silero_vad; print('  ✓ Silero VAD')"

# ── 下载模型 (可选) ──
if [ "$SKIP_MODELS" = "0" ]; then
    echo ""
    echo "下载预训练模型 (首次运行需要, ~3GB)..."
    ./.venv/bin/python scripts/download_models.py
fi

echo ""
echo "============================================================"
echo "  ✓ PersonaVoice 环境搭建完成!"
echo "============================================================"
echo ""
echo "  下一步:"
echo "    1. 下载评估数据集:  ./.venv/bin/python scripts/prepare_data.py"
echo "    2. 运行 Web Demo:   ./.venv/bin/python -m personavoice.demo.api_server"
echo "    3. 运行 1111 测试:  ./.venv/bin/python test_1111_clone.py"
echo ""
