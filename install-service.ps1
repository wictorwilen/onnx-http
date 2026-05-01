# install-service.ps1 — Install onnx-http as a Windows Service using NSSM
# Run this script as Administrator.

#Requires -RunAsAdministrator

$ErrorActionPreference = "Stop"

$ServiceName = "onnx-http"
$ProjectDir = $PSScriptRoot
$ExePath = Join-Path $ProjectDir "target\release\onnx-http.exe"
$OrtDll = Join-Path $ProjectDir "onnxruntime.dll"
$LogDir = Join-Path $ProjectDir "logs"

Write-Host ""
Write-Host "====================================" -ForegroundColor Cyan
Write-Host "  onnx-http Service Installer" -ForegroundColor Cyan
Write-Host "  Made with ❤️ by Wictor Wilén" -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan
Write-Host ""

# --- Check prerequisites ---

# Check NSSM
if (-not (Get-Command nssm -ErrorAction SilentlyContinue)) {
    Write-Host "NSSM not found. Installing via winget..." -ForegroundColor Yellow
    winget install nssm --accept-package-agreements --accept-source-agreements
    # Refresh PATH
    $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("PATH", "User")
    if (-not (Get-Command nssm -ErrorAction SilentlyContinue)) {
        Write-Host "ERROR: NSSM still not found after install. Add it to PATH and retry." -ForegroundColor Red
        exit 1
    }
}
Write-Host "✅ NSSM found: $(Get-Command nssm | Select-Object -ExpandProperty Source)" -ForegroundColor Green

# Check binary
if (-not (Test-Path $ExePath)) {
    Write-Host "ERROR: Binary not found at $ExePath" -ForegroundColor Red
    Write-Host "  Run 'cargo build --release' first, or run setup.ps1" -ForegroundColor Yellow
    exit 1
}
Write-Host "✅ Binary: $ExePath" -ForegroundColor Green

# Check ORT DLL
if (-not (Test-Path $OrtDll)) {
    Write-Host "ERROR: onnxruntime.dll not found at $OrtDll" -ForegroundColor Red
    Write-Host "  Run setup.ps1 to download it" -ForegroundColor Yellow
    exit 1
}
Write-Host "✅ ORT DLL: $OrtDll" -ForegroundColor Green

# Check model files
$modelsDir = Join-Path $ProjectDir "models"
$hasModels = $false
if (Test-Path $modelsDir) {
    $modelDirs = Get-ChildItem $modelsDir -Directory | Where-Object {
        (Test-Path (Join-Path $_.FullName "model.onnx")) -and (Test-Path (Join-Path $_.FullName "tokenizer.json"))
    }
    if ($modelDirs.Count -gt 0) {
        $hasModels = $true
        Write-Host ("✅ Models installed: " + ($modelDirs.Name -join ", ")) -ForegroundColor Green
    }
}
if (-not $hasModels) {
    Write-Host "ERROR: No models found. Run 'onnx-http.exe download <model>' first." -ForegroundColor Red
    exit 1
}

# Create log directory
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir | Out-Null
}

# --- Remove existing service if present ---

$ErrorActionPreference = "Continue"
$existingCheck = nssm status $ServiceName 2>&1
$ErrorActionPreference = "Stop"
if ($existingCheck -match "SERVICE_") {
    Write-Host ""
    Write-Host "Service '$ServiceName' already exists (status: $existingCheck). Reinstalling..." -ForegroundColor Yellow
    nssm stop $ServiceName 2>&1 | Out-Null
    Start-Sleep -Seconds 2
    nssm remove $ServiceName confirm 2>&1 | Out-Null
    Write-Host "  Removed existing service" -ForegroundColor DarkGray
} else {
    Write-Host "✅ No existing service to remove" -ForegroundColor Green
}

# --- Prompt for execution provider ---

Write-Host ""
$useNpu = Read-Host "Enable NPU acceleration? (y/N)"
if ($useNpu -eq "y" -or $useNpu -eq "Y") {
    $appArgs = "--npu"
    Write-Host "  Using: --npu (QNN NPU with CPU fallback)" -ForegroundColor Cyan
} else {
    $appArgs = "--cpu"
    Write-Host "  Using: --cpu" -ForegroundColor Cyan
}

# Prompt for pool size
$poolInput = Read-Host "Session pool size per model? (default: 2 for NPU, 4 for CPU)"
if ($poolInput -and $poolInput -match '^\d+$' -and [int]$poolInput -ge 1) {
    $appArgs += " --pool-size $poolInput"
    Write-Host "  Pool size: $poolInput" -ForegroundColor Cyan
} else {
    Write-Host "  Pool size: auto (default)" -ForegroundColor Cyan
}

# --- Install service ---

Write-Host ""
Write-Host "Installing service..." -ForegroundColor Yellow

nssm install $ServiceName $ExePath
nssm set $ServiceName AppParameters $appArgs
nssm set $ServiceName AppDirectory $ProjectDir
nssm set $ServiceName AppEnvironmentExtra "ORT_DYLIB_PATH=$OrtDll"

# Display name and description
nssm set $ServiceName DisplayName "ONNX Embedding Server"
nssm set $ServiceName Description "onnx-http: ONNX Runtime embedding server with optional NPU acceleration"

# Auto-start on boot
nssm set $ServiceName Start SERVICE_AUTO_START

# Logging
nssm set $ServiceName AppStdout (Join-Path $LogDir "stdout.log")
nssm set $ServiceName AppStderr (Join-Path $LogDir "stderr.log")
nssm set $ServiceName AppRotateFiles 1
nssm set $ServiceName AppRotateBytes 10485760  # 10 MB

# Restart on failure
nssm set $ServiceName AppExit Default Restart
nssm set $ServiceName AppRestartDelay 5000  # 5 seconds

Write-Host "✅ Service installed" -ForegroundColor Green

# --- Start service ---

Write-Host ""
Write-Host "Starting service..." -ForegroundColor Yellow
nssm start $ServiceName
Start-Sleep -Seconds 5

$status = nssm status $ServiceName 2>&1
if ($status -match "SERVICE_RUNNING") {
    Write-Host "✅ Service is running!" -ForegroundColor Green
} else {
    Write-Host "⚠️  Service status: $status" -ForegroundColor Yellow
    Write-Host "  Check logs at: $LogDir" -ForegroundColor Yellow
}

# --- Summary ---

Write-Host ""
Write-Host "====================================" -ForegroundColor Cyan
Write-Host "  Service Installed!" -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Service name:  $ServiceName" -ForegroundColor White
Write-Host "  Endpoint:      http://localhost:8901/v1/embeddings" -ForegroundColor White
Write-Host "  Health check:  http://localhost:8901/health" -ForegroundColor White
Write-Host "  Logs:          $LogDir" -ForegroundColor White
Write-Host ""
Write-Host "Manage the service:" -ForegroundColor White
Write-Host "  nssm status $ServiceName        # Check status" -ForegroundColor Yellow
Write-Host "  nssm stop $ServiceName          # Stop" -ForegroundColor Yellow
Write-Host "  nssm start $ServiceName         # Start" -ForegroundColor Yellow
Write-Host "  nssm restart $ServiceName       # Restart" -ForegroundColor Yellow
Write-Host "  nssm remove $ServiceName confirm  # Uninstall" -ForegroundColor Yellow
Write-Host "  services.msc                      # Windows Services UI" -ForegroundColor Yellow
Write-Host ""
