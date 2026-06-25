# ============================================================
# PersonaVoice v10.4.8 — 一键环境搭建脚本 (Windows PowerShell)
# ============================================================
# 用法:
#   powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
#
# 功能:
#   1. 创建 .venv 虚拟环境
#   2. 升级 pip
#   3. 安装 PyTorch (CUDA 11.8 / CPU)
#   4. 安装 PersonaVoice 依赖
#   5. 安装 silero-vad, f5-tts
#   6. 验证安装
# ============================================================

param(
    [switch]$UseCPU,
    [switch]$SkipModels
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  PersonaVoice v10.4.8 环境搭建 (Windows PowerShell)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# ── 1. 检查 Python ──
Write-Host "[1/6] 检查 Python ..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "  ✓ $pythonVersion"
} catch {
    Write-Host "  [ERROR] 未找到 Python, 请先安装 Python 3.10+: https://www.python.org/downloads/" -ForegroundColor Red
    exit 1
}

# ── 2. 创建虚拟环境 ──
Write-Host ""
Write-Host "[2/6] 创建虚拟环境 .venv ..." -ForegroundColor Yellow
if (Test-Path ".venv") {
    Write-Host "  .venv 已存在, 跳过创建"
} else {
    # 继承系统 site-packages (复用已安装的 torch/speechbrain/vocos)
    python -m venv --system-site-packages .venv
    Write-Host "  ✓ 虚拟环境创建完成"
}

# ── 3. 升级 pip ──
Write-Host ""
Write-Host "[3/6] 升级 pip ..." -ForegroundColor Yellow
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
Write-Host "  ✓ pip 升级完成"

# ── 4. 安装 PyTorch ──
Write-Host ""
Write-Host "[4/6] 安装 PyTorch ..." -ForegroundColor Yellow
if ($UseCPU) {
    Write-Host "  安装 CPU 版本 (--UseCPU)"
    & .\.venv\Scripts\pip.exe install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
} else {
    Write-Host "  安装 CUDA 11.8 版本 (默认)"
    Write-Host "  如需 CPU 版本, 请加 -UseCPU 参数"
    & .\.venv\Scripts\pip.exe install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
}
Write-Host "  ✓ PyTorch 安装完成"

# ── 5. 安装 PersonaVoice 依赖 ──
Write-Host ""
Write-Host "[5/6] 安装 PersonaVoice 依赖 ..." -ForegroundColor Yellow
& .\.venv\Scripts\pip.exe install -r requirements.txt
& .\.venv\Scripts\pip.exe install silero-vad f5-tts imageio-ffmpeg
& .\.venv\Scripts\pip.exe install -e .
Write-Host "  ✓ 依赖安装完成"

# ── 6. 验证安装 ──
Write-Host ""
Write-Host "[6/6] 验证安装 ..." -ForegroundColor Yellow
& .\.venv\Scripts\python.exe -c "import torch; print(f'  ✓ PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
& .\.venv\Scripts\python.exe -c "import personavoice; print(f'  ✓ PersonaVoice {personavoice.__version__}')"
& .\.venv\Scripts\python.exe -c "from f5_tts.api import F5TTS; print('  ✓ F5-TTS')"
& .\.venv\Scripts\python.exe -c "import speechbrain; print(f'  ✓ SpeechBrain {speechbrain.__version__}')"
& .\.venv\Scripts\python.exe -c "import silero_vad; print('  ✓ Silero VAD')"

# ── 下载模型 (可选) ──
if (-not $SkipModels) {
    Write-Host ""
    Write-Host "下载预训练模型 (首次运行需要, ~3GB)..." -ForegroundColor Yellow
    & .\.venv\Scripts\python.exe scripts\download_models.py
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  ✓ PersonaVoice 环境搭建完成!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  下一步:"
Write-Host "    1. 下载评估数据集:  .\.venv\Scripts\python.exe scripts\prepare_data.py"
Write-Host "    2. 运行 Web Demo:   .\.venv\Scripts\python.exe -m personavoice.demo.api_server"
Write-Host "    3. 运行 1111 测试:  .\.venv\Scripts\python.exe test_1111_clone.py"
Write-Host ""
