# install.ps1 — One-line installer for onnx-http
#
# Usage:
#   irm https://raw.githubusercontent.com/wictorwilen/onnx-http/main/install.ps1 | iex
#
# What it does:
#   1. Clones the repo (or updates if already present)
#   2. Downloads ONNX Runtime 1.24.4 (ARM64+QNN or x64)
#   3. Checks for Rust and builds the project
#   4. Downloads the default embedding model
#   5. Optionally installs as a Windows Service

$ErrorActionPreference = "Stop"

$RepoUrl = "https://github.com/wictorwilen/onnx-http.git"
$DefaultInstallDir = "C:\onnx-http"

Write-Host ""
Write-Host "====================================" -ForegroundColor Cyan
Write-Host "  onnx-http Installer" -ForegroundColor Cyan
Write-Host "  Made with " -NoNewline -ForegroundColor Cyan
Write-Host ([char]0x2764) -NoNewline -ForegroundColor Red
Write-Host " by Wictor Wilen" -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan
Write-Host ""

# --- Determine install directory ---

$InstallDir = $DefaultInstallDir
Write-Host "Install directory: $InstallDir"
Write-Host "  (set `$env:ONNX_HTTP_DIR` to override)" -ForegroundColor DarkGray
if ($env:ONNX_HTTP_DIR) {
    $InstallDir = $env:ONNX_HTTP_DIR
    Write-Host "  Using: $InstallDir" -ForegroundColor Cyan
}
Write-Host ""

# --- Step 1: Clone or update repo ---

Write-Host "[1/5] Getting source code..." -ForegroundColor Yellow

if (Test-Path (Join-Path $InstallDir ".git")) {
    Write-Host "  Repository exists, pulling latest..."
    Push-Location $InstallDir
    git pull --quiet 2>&1 | Out-Null
    Pop-Location
    Write-Host "  Updated to latest" -ForegroundColor Green
} else {
    if (Test-Path $InstallDir) {
        Write-Host "  Directory exists but is not a git repo. Cloning fresh..."
        Remove-Item $InstallDir -Recurse -Force
    }
    Write-Host "  Cloning $RepoUrl..."
    git clone --quiet $RepoUrl $InstallDir
    Write-Host "  Cloned to $InstallDir" -ForegroundColor Green
}

Push-Location $InstallDir

try {

# --- Step 2: Download ONNX Runtime ---

Write-Host ""
Write-Host "[2/5] Setting up ONNX Runtime 1.24.4..." -ForegroundColor Yellow

if (Test-Path "onnxruntime.dll") {
    Write-Host "  onnxruntime.dll already exists, skipping" -ForegroundColor DarkGray
} else {
    $arch = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture
    if ($arch -eq "Arm64") {
        $nugetUrl = "https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.QNN/1.24.4"
        Write-Host "  Architecture: $arch (downloading QNN-enabled build)"
        Invoke-WebRequest -Uri $nugetUrl -OutFile "ort-qnn.nupkg.zip"

        Expand-Archive "ort-qnn.nupkg.zip" -DestinationPath "ort-qnn-extract" -Force
        $nativeDir = "ort-qnn-extract\runtimes\win-arm64\native"
        if (Test-Path $nativeDir) {
            Copy-Item "$nativeDir\onnxruntime.dll" "onnxruntime.dll"
            Get-ChildItem "$nativeDir\*.dll" | Where-Object { $_.Name -ne "onnxruntime.dll" } | ForEach-Object {
                Copy-Item $_.FullName $_.Name
            }
        } else {
            Write-Host "  ERROR: Native DLLs not found in NuGet package" -ForegroundColor Red
            exit 1
        }
        Remove-Item "ort-qnn.nupkg.zip" -Force
        Remove-Item "ort-qnn-extract" -Recurse -Force
        Write-Host "  onnxruntime.dll + QNN DLLs ready" -ForegroundColor Green
    } else {
        $ortUrl = "https://github.com/microsoft/onnxruntime/releases/download/v1.24.4/onnxruntime-win-x64-1.24.4.zip"
        Write-Host "  Architecture: $arch (CPU-only build)"
        Invoke-WebRequest -Uri $ortUrl -OutFile "ort-download.zip"

        Expand-Archive "ort-download.zip" -DestinationPath "ort-extract" -Force
        Copy-Item "ort-extract\onnxruntime-win-x64-1.24.4\lib\onnxruntime.dll" "onnxruntime.dll"
        Remove-Item "ort-download.zip" -Force
        Remove-Item "ort-extract" -Recurse -Force
        Write-Host "  onnxruntime.dll ready (CPU only)" -ForegroundColor Green
    }
}

# --- Step 3: Build ---

Write-Host ""
Write-Host "[3/5] Building onnx-http..." -ForegroundColor Yellow

if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    Write-Host "  ERROR: Rust/Cargo not found." -ForegroundColor Red
    Write-Host "  Install Rust first: https://rustup.rs/" -ForegroundColor Yellow
    Write-Host "  Then re-run this installer." -ForegroundColor Yellow
    exit 1
}

$rustVersion = (rustc --version) 2>&1
Write-Host "  Rust: $rustVersion"

# Kill any running instance to avoid locking the exe
$running = Get-Process -Name "onnx-http" -ErrorAction SilentlyContinue
if ($running) {
    Write-Host "  Stopping running onnx-http process..."
    $running | Stop-Process -Force
    Start-Sleep -Seconds 1
}

$env:ORT_DYLIB_PATH = Join-Path $InstallDir "onnxruntime.dll"
cargo build --release 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "  ERROR: Build failed!" -ForegroundColor Red
    exit 1
}
Write-Host "  Build succeeded" -ForegroundColor Green

# --- Step 4: Download default model ---

Write-Host ""
Write-Host "[4/5] Downloading default model..." -ForegroundColor Yellow

$exe = Join-Path $InstallDir "target\release\onnx-http.exe"

if (Test-Path "models\all-MiniLM-L6-v2\model.onnx") {
    Write-Host "  all-MiniLM-L6-v2 already installed, skipping" -ForegroundColor DarkGray
} else {
    & $exe download all-MiniLM-L6-v2
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ERROR: Model download failed!" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "  Installed models:" -ForegroundColor White
& $exe models
Write-Host ""
Write-Host "  Download more with: onnx-http.exe download --list" -ForegroundColor DarkGray

# --- Step 5: Optional service install ---

Write-Host ""
Write-Host "[5/5] Windows Service setup" -ForegroundColor Yellow
Write-Host ""
$installService = Read-Host "  Install as Windows Service? (y/N)"

if ($installService -eq "y" -or $installService -eq "Y") {
    # Check if running as admin
    $isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator
    )
    if (-not $isAdmin) {
        Write-Host ""
        Write-Host "  Service installation requires Administrator privileges." -ForegroundColor Yellow
        Write-Host "  Run the following in an elevated PowerShell:" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "    cd $InstallDir" -ForegroundColor Cyan
        Write-Host "    .\install-service.ps1" -ForegroundColor Cyan
        Write-Host ""
    } else {
        # Check/install NSSM
        if (-not (Get-Command nssm -ErrorAction SilentlyContinue)) {
            Write-Host "  Installing NSSM via winget..." -ForegroundColor Yellow
            winget install nssm --accept-package-agreements --accept-source-agreements
            $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("PATH", "User")
            if (-not (Get-Command nssm -ErrorAction SilentlyContinue)) {
                Write-Host "  ERROR: NSSM not found after install. Add to PATH and run install-service.ps1 manually." -ForegroundColor Red
                $installService = "n"
            }
        }

        if ($installService -eq "y" -or $installService -eq "Y") {
            $ServiceName = "onnx-http"
            $LogDir = Join-Path $InstallDir "logs"
            $OrtDll = Join-Path $InstallDir "onnxruntime.dll"

            if (-not (Test-Path $LogDir)) {
                New-Item -ItemType Directory -Path $LogDir | Out-Null
            }

            # Remove existing service if present
            $ErrorActionPreference = "Continue"
            $existingCheck = nssm status $ServiceName 2>&1
            $ErrorActionPreference = "Stop"
            if ($existingCheck -match "SERVICE_") {
                Write-Host "  Removing existing service..."
                nssm stop $ServiceName 2>&1 | Out-Null
                Start-Sleep -Seconds 2
                nssm remove $ServiceName confirm 2>&1 | Out-Null
            }

            # Prompt for execution provider
            $arch = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture
            if ($arch -eq "Arm64") {
                $useNpu = Read-Host "  Enable NPU acceleration? (y/N)"
                $appArgs = if ($useNpu -eq "y" -or $useNpu -eq "Y") { "--npu" } else { "--cpu" }
            } else {
                $appArgs = "--cpu"
            }

            Write-Host "  Installing service ($appArgs)..."
            nssm install $ServiceName $exe | Out-Null
            nssm set $ServiceName AppParameters $appArgs | Out-Null
            nssm set $ServiceName AppDirectory $InstallDir | Out-Null
            nssm set $ServiceName AppEnvironmentExtra "ORT_DYLIB_PATH=$OrtDll" | Out-Null
            nssm set $ServiceName DisplayName "ONNX Embedding Server" | Out-Null
            nssm set $ServiceName Description "onnx-http: ONNX Runtime embedding server with optional NPU acceleration" | Out-Null
            nssm set $ServiceName Start SERVICE_AUTO_START | Out-Null
            nssm set $ServiceName AppStdout (Join-Path $LogDir "stdout.log") | Out-Null
            nssm set $ServiceName AppStderr (Join-Path $LogDir "stderr.log") | Out-Null
            nssm set $ServiceName AppRotateFiles 1 | Out-Null
            nssm set $ServiceName AppRotateBytes 10485760 | Out-Null
            nssm set $ServiceName AppExit Default Restart | Out-Null
            nssm set $ServiceName AppRestartDelay 5000 | Out-Null

            Write-Host "  Starting service..."
            nssm start $ServiceName | Out-Null
            Start-Sleep -Seconds 5

            $status = nssm status $ServiceName 2>&1
            if ($status -match "SERVICE_RUNNING") {
                Write-Host "  Service is running!" -ForegroundColor Green
            } else {
                Write-Host "  Service status: $status" -ForegroundColor Yellow
                Write-Host "  Check logs at: $LogDir" -ForegroundColor Yellow
            }
        }
    }
} else {
    Write-Host "  Skipped. You can install later with: .\install-service.ps1" -ForegroundColor DarkGray
}

# --- Done ---

Write-Host ""
Write-Host "====================================" -ForegroundColor Cyan
Write-Host "  Installation Complete!" -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Install dir:  $InstallDir" -ForegroundColor White
Write-Host "  Binary:       $exe" -ForegroundColor White
Write-Host ""

if (-not ($installService -eq "y" -or $installService -eq "Y") -or -not $isAdmin) {
    Write-Host "To start the server manually:" -ForegroundColor White
    Write-Host ""
    Write-Host "  cd $InstallDir" -ForegroundColor Yellow
    Write-Host "  `$env:ORT_DYLIB_PATH = `"$InstallDir\onnxruntime.dll`"" -ForegroundColor Yellow
    Write-Host "  .\target\release\onnx-http.exe" -ForegroundColor Yellow
    Write-Host ""
}

Write-Host "Download more models:" -ForegroundColor White
Write-Host ""
Write-Host "  cd $InstallDir" -ForegroundColor Yellow
Write-Host "  .\target\release\onnx-http.exe download --list" -ForegroundColor Yellow
Write-Host "  .\target\release\onnx-http.exe download bge-base-en-v1.5" -ForegroundColor Yellow
Write-Host ""
Write-Host "Test with:" -ForegroundColor White
Write-Host ""
Write-Host '  curl http://localhost:8901/v1/models' -ForegroundColor Yellow
Write-Host '  curl http://localhost:8901/v1/embeddings -H "Content-Type: application/json" -d "{\"model\": \"all-MiniLM-L6-v2\", \"input\": \"hello\"}"' -ForegroundColor Yellow
Write-Host ""

} finally {
    Pop-Location
}
