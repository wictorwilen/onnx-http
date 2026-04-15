# setup.ps1 — Automated setup for onnx-http embedding server
# Downloads model files, ONNX Runtime, and builds the project.

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "====================================" -ForegroundColor Cyan
Write-Host "  onnx-http Setup Script" -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan
Write-Host ""

# --- Check prerequisites ---

Write-Host "[1/5] Checking prerequisites..." -ForegroundColor Yellow

# Rust
if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: Rust/Cargo not found. Install from https://rustup.rs/" -ForegroundColor Red
    exit 1
}
$rustVersion = (rustc --version) 2>&1
Write-Host "  Rust: $rustVersion" -ForegroundColor Green

# Python
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: Python not found. Install Python 3.x from https://python.org/" -ForegroundColor Red
    exit 1
}
$pyVersion = (python --version) 2>&1
Write-Host "  Python: $pyVersion" -ForegroundColor Green

# --- Download model files ---

Write-Host ""
Write-Host "[2/5] Downloading model files (all-MiniLM-L6-v2)..." -ForegroundColor Yellow

if (-not (Test-Path "models")) {
    New-Item -ItemType Directory -Path "models" | Out-Null
}

# Install huggingface_hub if not present
python -m pip install --quiet huggingface_hub 2>&1 | Out-Null

if (-not (Test-Path "models\model.onnx")) {
    Write-Host "  Downloading model.onnx (~90 MB)..."
    python -c "from huggingface_hub import hf_hub_download; hf_hub_download('sentence-transformers/all-MiniLM-L6-v2', 'onnx/model.onnx', local_dir='models', local_dir_use_symlinks=False)"
    # The file downloads to models/onnx/model.onnx — move it up
    if (Test-Path "models\onnx\model.onnx") {
        Move-Item "models\onnx\model.onnx" "models\model.onnx" -Force
        Remove-Item "models\onnx" -Recurse -Force -ErrorAction SilentlyContinue
    }
    Write-Host "  model.onnx downloaded" -ForegroundColor Green
} else {
    Write-Host "  model.onnx already exists, skipping" -ForegroundColor DarkGray
}

if (-not (Test-Path "models\tokenizer.json")) {
    Write-Host "  Downloading tokenizer.json (~466 KB)..."
    python -c "from huggingface_hub import hf_hub_download; hf_hub_download('sentence-transformers/all-MiniLM-L6-v2', 'tokenizer.json', local_dir='models', local_dir_use_symlinks=False)"
    Write-Host "  tokenizer.json downloaded" -ForegroundColor Green
} else {
    Write-Host "  tokenizer.json already exists, skipping" -ForegroundColor DarkGray
}

# --- Download ONNX Runtime ---

Write-Host ""
Write-Host "[3/5] Downloading ONNX Runtime 1.24.4..." -ForegroundColor Yellow

if (-not (Test-Path "onnxruntime.dll")) {
    # Detect architecture
    $arch = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture
    if ($arch -eq "Arm64") {
        $ortUrl = "https://github.com/microsoft/onnxruntime/releases/download/v1.24.4/onnxruntime-win-arm64x-1.24.4.zip"
        $ortDir = "onnxruntime-win-arm64x-1.24.4"
    } else {
        $ortUrl = "https://github.com/microsoft/onnxruntime/releases/download/v1.24.4/onnxruntime-win-x64-1.24.4.zip"
        $ortDir = "onnxruntime-win-x64-1.24.4"
    }

    Write-Host "  Architecture: $arch"
    Write-Host "  Downloading from $ortUrl..."
    Invoke-WebRequest -Uri $ortUrl -OutFile "ort-download.zip"

    Write-Host "  Extracting onnxruntime.dll..."
    Expand-Archive "ort-download.zip" -DestinationPath "ort-extract" -Force
    Copy-Item "ort-extract\$ortDir\lib\onnxruntime.dll" "onnxruntime.dll"

    # Cleanup
    Remove-Item "ort-download.zip" -Force
    Remove-Item "ort-extract" -Recurse -Force
    Write-Host "  onnxruntime.dll ready" -ForegroundColor Green
} else {
    Write-Host "  onnxruntime.dll already exists, skipping" -ForegroundColor DarkGray
}

# --- Build ---

Write-Host ""
Write-Host "[4/5] Building project (release mode)..." -ForegroundColor Yellow

# Kill any running instance that might lock the exe
$running = Get-Process -Name "onnx-http" -ErrorAction SilentlyContinue
if ($running) {
    Write-Host "  Stopping running onnx-http process..."
    $running | Stop-Process -Force
    Start-Sleep -Seconds 1
}

$env:ORT_DYLIB_PATH = "$PWD\onnxruntime.dll"
cargo build --release
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Build failed!" -ForegroundColor Red
    exit 1
}
Write-Host "  Build succeeded" -ForegroundColor Green

# --- Done ---

Write-Host ""
Write-Host "====================================" -ForegroundColor Cyan
Write-Host "  Setup Complete!" -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To start the server:" -ForegroundColor White
Write-Host ""
Write-Host '  $env:ORT_DYLIB_PATH = "$PWD\onnxruntime.dll"' -ForegroundColor Yellow
Write-Host '  .\target\release\onnx-http.exe' -ForegroundColor Yellow
Write-Host ""
Write-Host "Then test with:" -ForegroundColor White
Write-Host ""
Write-Host '  curl http://localhost:8901/health' -ForegroundColor Yellow
Write-Host '  curl http://localhost:8901/v1/embeddings -H "Content-Type: application/json" -d "{\"input\": \"hello\"}"' -ForegroundColor Yellow
Write-Host ""
