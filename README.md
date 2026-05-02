# onnx-http 🚀

A Rust-based HTTP server that loads any ONNX embedding model, runs inference via ONNX Runtime with optional NPU acceleration (QNNExecutionProvider), and exposes an OpenAI-compatible `/v1/embeddings` endpoint. Designed to run natively on **Windows ARM64** (Snapdragon) and serve embedding requests from WSL, scripts, or any HTTP client.

> 📌 This project ships with setup instructions for [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) as the default model, but you can load **multiple ONNX models** simultaneously — just create a subdirectory per model under `models/` with a `model.onnx` and matching `tokenizer.json`.

## ✨ Features

- 🧠 **Sentence embeddings** — 384 to 1024-dim vectors, with models supporting up to 8K tokens
- 🔀 **Multi-model support** — load multiple models and select per request
- ⚡ **NPU acceleration** via QNNExecutionProvider (opt-in, for Snapdragon devices)
- 🔄 **Session pooling** — concurrent inference via `--pool-size` for higher throughput
- 🖥️ **CPU fallback** — works on any Windows machine
- 📦 **Single and batch** embedding requests
- 🔌 **OpenAI-compatible** `/v1/embeddings` and `/v1/models` endpoints
- 📝 **Structured logging** via `tracing`
- 💚 **Health check** endpoint at `/health`
- 🔄 **Bring your own model** — swap in any ONNX embedding model

## 📋 Prerequisites

| Requirement | Details |
|-------------|---------|
| 🦀 **Rust** (stable 1.75+) | [Install via rustup](https://rustup.rs/) |
| 🐍 **Python 3.x** | For downloading model files (one-time setup) |
| 📦 **ONNX Runtime 1.24.x** | DLL downloaded automatically by setup script |
| 🔧 **C++ Build Tools** | Required by Rust linker — install via [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) |

## 🏁 Quick Start

### One-line install

```powershell
irm https://raw.githubusercontent.com/wictorwilen/onnx-http/main/install.ps1 | iex
```

This will clone the repo to `C:\onnx-http`, download ONNX Runtime, build the project, download the default model, and optionally install as a Windows Service.

> Set `$env:ONNX_HTTP_DIR` before running to install to a different directory.

### Local setup (if you already cloned the repo)

```powershell
.\setup.ps1
```

This will:
1. Check that Rust is installed
2. Download **ONNX Runtime 1.24.4** for your architecture
3. Build the project in release mode
4. Download the **all-MiniLM-L6-v2** embedding model

Then start the server:

```powershell
$env:ORT_DYLIB_PATH = "$PWD\onnxruntime.dll"
.\target\release\onnx-http.exe                              # Start server (CPU)
.\target\release\onnx-http.exe --npu                        # Start server (NPU)
```

## 🔧 Manual Setup

If you prefer to set things up step by step:

### 1. Clone and enter the project

```powershell
git clone <repo-url>
cd onnx-http
```

### 2. Download embedding models

The built-in downloader fetches models directly from HuggingFace:

```powershell
# List available models
.\target\release\onnx-http.exe download --list

# Download a model (e.g., all-MiniLM-L6-v2)
.\target\release\onnx-http.exe download all-MiniLM-L6-v2

# Download additional models
.\target\release\onnx-http.exe download bge-base-en-v1.5
```

Available models:

| Name | Dims | Max Tokens | Description |
|------|------|------------|-------------|
| `all-MiniLM-L6-v2` | 384 | 512 | Fast, lightweight general-purpose embeddings |
| `all-MiniLM-L12-v2` | 384 | 512 | Better quality than L6, still fast |
| `all-mpnet-base-v2` | 768 | 512 | Best all-around sentence-transformers model |
| `bge-base-en-v1.5` | 768 | 512 | Top retrieval/search quality, great for documents |
| `bge-large-en-v1.5` | 1024 | 512 | Highest quality BGE model, 1024 dimensions |
| `bge-m3` | 1024 | 8,192 | 8K context, multilingual, dense+sparse retrieval (2.3GB) |
| `jina-embeddings-v2-small-en` | 512 | 8,192 | 8K context, lightweight English embeddings (130MB) |
| `embeddinggemma-300m` | 768 | 2,048 | Google Gemma, 2K context, multilingual (1.2GB) |
| `multi-qa-mpnet-base-cos-v1` | 768 | 512 | Trained for semantic search and QA |

After downloading, your `models/` directory will look like:

```
models/
├── all-MiniLM-L6-v2/
│   ├── model.onnx
│   └── tokenizer.json
└── bge-base-en-v1.5/
    ├── model.onnx
    └── tokenizer.json
```

Then use any model in requests:

```bash
curl http://localhost:8901/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "bge-base-en-v1.5", "input": "Hello, world!"}'
```

<details>
<summary>Manual download (without the CLI)</summary>

You can also download models manually using Python:

```powershell
pip install huggingface_hub

New-Item -ItemType Directory -Path "models\all-MiniLM-L6-v2" -Force
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('sentence-transformers/all-MiniLM-L6-v2', 'onnx/model.onnx', local_dir='models/all-MiniLM-L6-v2', local_dir_use_symlinks=False)"
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('sentence-transformers/all-MiniLM-L6-v2', 'tokenizer.json', local_dir='models/all-MiniLM-L6-v2', local_dir_use_symlinks=False)"
Move-Item models\all-MiniLM-L6-v2\onnx\model.onnx models\all-MiniLM-L6-v2\model.onnx
Remove-Item models\all-MiniLM-L6-v2\onnx -Recurse
```

</details>

### 3. Download ONNX Runtime 1.24.x

The `ort` crate requires ONNX Runtime **1.24.x**. For NPU support on ARM64, use the QNN-enabled build from NuGet:

**ARM64 with NPU support (recommended for Snapdragon):**

```powershell
# Download QNN-enabled ORT from NuGet (includes all QNN DLLs)
Invoke-WebRequest -Uri "https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.QNN/1.24.4" -OutFile ort-qnn.zip
Expand-Archive ort-qnn.zip -DestinationPath ort-extract
Copy-Item ort-extract\runtimes\win-arm64\native\*.dll .
Remove-Item ort-qnn.zip, ort-extract -Recurse
```

This gives you `onnxruntime.dll` plus QNN DLLs (`QnnHtp.dll`, `QnnGpu.dll`, `QnnCpu.dll`, `QnnSystem.dll`, etc.).

**x64 (CPU only):**

```powershell
Invoke-WebRequest -Uri "https://github.com/microsoft/onnxruntime/releases/download/v1.24.4/onnxruntime-win-x64-1.24.4.zip" -OutFile ort.zip
Expand-Archive ort.zip -DestinationPath ort-extract
Copy-Item ort-extract\onnxruntime-win-x64-1.24.4\lib\onnxruntime.dll .
Remove-Item ort.zip, ort-extract -Recurse
```

> **⚠️ Important:** Windows ships a bundled `onnxruntime.dll` in `C:\Windows\System32` that is **incompatible** (version 5.x, not 1.24.x). You **must** set `ORT_DYLIB_PATH` to point to the correct DLL, or the server will fail to start. The setup script handles this automatically.

### 4. Build

```powershell
cargo build --release
```

The compiled binary will be at `target\release\onnx-http.exe`.

### 5. Run the server

```powershell
# Set the path to the correct ONNX Runtime DLL
$env:ORT_DYLIB_PATH = "$PWD\onnxruntime.dll"

# CPU mode (default)
.\target\release\onnx-http.exe

# NPU mode (Snapdragon devices with QNN)
.\target\release\onnx-http.exe --npu

# Custom port
.\target\release\onnx-http.exe --port 9000

# All options
.\target\release\onnx-http.exe --help
```

You should see output like:

```
2026-04-15T20:00:00Z  INFO Initializing ONNX Runtime from C:\code\onnx-http\onnxruntime.dll
2026-04-15T20:00:00Z  INFO ONNX Runtime loaded successfully
2026-04-15T20:00:00Z  INFO Loading tokenizer from models\tokenizer.json
2026-04-15T20:00:00Z  INFO Loading ONNX model from models\model.onnx
2026-04-15T20:00:01Z  INFO Model loaded. Input names: ["input_ids", "attention_mask", "token_type_ids"]
2026-04-15T20:00:01Z  INFO Model and tokenizer loaded successfully
2026-04-15T20:00:01Z  INFO Starting ONNX embedding server on http://0.0.0.0:8901
```

## 📡 API Reference

### `POST /v1/embeddings`

Generate embeddings for one or more text inputs. Compatible with the OpenAI embeddings API format.

**Single input:**

```bash
curl http://localhost:8901/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "all-MiniLM-L6-v2", "input": "Hello, world!"}'
```

**Batch input:**

```bash
curl http://localhost:8901/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "all-MiniLM-L6-v2", "input": ["Hello, world!", "How are you?", "Machine learning is great"]}'
```

**Response:**

```json
{
  "object": "list",
  "model": "all-MiniLM-L6-v2",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [-0.197, 0.177, 0.038, ...]
    }
  ]
}
```

The embedding dimensions depend on the model (384 for MiniLM, 512 for Jina, 768 for BGE/mpnet, 1024 for BGE-large/M3).

**Error response (400):**

```json
{
  "error": "Input text must not be empty"
}
```

### `GET /v1/models`

List all loaded models. Compatible with the OpenAI models API format.

```bash
curl http://localhost:8901/v1/models
```

```json
{
  "object": "list",
  "data": [
    { "id": "all-MiniLM-L6-v2", "object": "model", "owned_by": "local" }
  ]
}
```

### `GET /health`

```bash
curl http://localhost:8901/health
```

```json
{"status": "ok", "models": ["all-MiniLM-L6-v2"]}
```

## 🔗 WSL ↔ Windows Interop

The server binds to `0.0.0.0:8901`, making it accessible from WSL via `localhost`:

```bash
# From WSL — single embedding
curl http://localhost:8901/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "all-MiniLM-L6-v2", "input": "hello from WSL"}'

# From WSL — batch of embeddings
curl http://localhost:8901/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "all-MiniLM-L6-v2", "input": ["sentence one", "sentence two", "sentence three"]}'
```

No Windows-specific APIs are used — pure Rust networking. The standard WSL ↔ Windows localhost bridge works out of the box.

### Python example (from WSL or Windows)

```python
import requests

response = requests.post(
    "http://localhost:8901/v1/embeddings",
    json={"model": "all-MiniLM-L6-v2", "input": ["hello world", "how are you"]},
)
data = response.json()
for item in data["data"]:
    print(f"[{item['index']}] {len(item['embedding'])} dims, first 3: {item['embedding'][:3]}")
```

## ⚙️ Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `ORT_DYLIB_PATH` | **Required.** Full path to `onnxruntime.dll` (v1.24.x) | Auto-detects next to exe |
| `PORT` | Server listen port (overridden by `--port`) | `8901` |
| `DEFAULT_MODEL` | Default model name (used when only one model loaded) | First alphabetically |
| `RUST_LOG` | Log level (`trace`, `debug`, `info`, `warn`, `error`) | `info` |

### Command-line options

| Flag | Description |
|------|-------------|
| `--npu` | Use QNN NPU execution provider (with CPU fallback) |
| `--cpu` | Use CPU execution provider only (default) |
| `--port N` | Server listen port (overrides `PORT` env var) |
| `--pool-size N` | Number of inference sessions per model (default: 2 for NPU, 4 for CPU) |
| `--help` | Show help message |

### 🚀 NPU acceleration (Snapdragon devices)

If you have a Qualcomm Snapdragon device with NPU, the setup script automatically downloads the QNN-enabled ONNX Runtime build (from the `Microsoft.ML.OnnxRuntime.QNN` NuGet package), which includes all required QNN DLLs.

```powershell
$env:ORT_DYLIB_PATH = "$PWD\onnxruntime.dll"
.\target\release\onnx-http.exe --npu
```

The server will attempt QNNExecutionProvider first, falling back to CPU if unavailable.

> **Note:** The first launch with `--npu` takes significantly longer (1-2 minutes) as QNN compiles and optimizes the model graph for the NPU. Subsequent launches are faster.

### 🧪 Preparing models for full NPU utilization

Standard HuggingFace ONNX models have **dynamic shapes** and use the **Erf** operator (exact GELU activation). Both are incompatible with QNN:

| Issue | Symptom | Fix |
|-------|---------|-----|
| Dynamic shapes | "Cannot get shape" warnings, 0% NPU usage | Export with fixed `batch_size=1` and `seq_len` |
| Erf operator | "QNN graph execute error 6002" | Replace GELU with tanh approximation before export |

The included export script handles both automatically:

```powershell
# Install dependencies (one-time)
pip install torch transformers onnx onnxsim

# Export with static shapes + tanh GELU (QNN-compatible)
python scripts/export_static_onnx.py \
  --model BAAI/bge-base-en-v1.5 \
  --seq-len 256 \
  --output models/bge-base-en-v1.5-static

python scripts/export_static_onnx.py \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --seq-len 256 \
  --output models/all-MiniLM-L6-v2-static
```

The script will:
1. Load the HuggingFace model
2. Patch all GELU activations to use `tanh` approximation (eliminates Erf ops)
3. Export ONNX with fixed `[1, seq_len]` input shapes (no dynamic axes)
4. Simplify the graph with `onnxsim` to fold constants and remove Shape ops
5. Verify zero Shape/Erf ops remain in the final model
6. Save `tokenizer.json` and `model_config.json` (with `static_shapes: true`)

**Choosing `--seq-len`:** This is the fixed input length for all inference. Texts shorter than this are zero-padded; texts longer are truncated. Use 256 for general-purpose (covers ~95% of inputs), or 128 for latency-sensitive workloads.

**How to verify NPU is active:** After starting with `--npu` and a static model, check:
- Task Manager → Performance → NPU should show utilization during requests
- No "Cannot get shape" warnings in logs
- No "QNN graph execute error" in logs

**Compatible model architectures:** Any BERT-family encoder model works (BERT, RoBERTa, DistilBERT, BGE, MiniLM, E5, GTE, etc.). The key requirements are:
- Pure encoder architecture (no decoder/cross-attention)
- Attention + FFN layers with GELU activation
- The export script automatically handles GELU → tanh patching

## 📁 Project Structure

```
onnx-http/
├── Cargo.toml           # Dependencies and project config
├── setup.ps1            # Automated setup script
├── onnxruntime.dll      # ONNX Runtime 1.24.x (downloaded, not committed)
├── models/
│   └── all-MiniLM-L6-v2/
│       ├── model.onnx       # ONNX model (downloaded, not committed)
│       └── tokenizer.json   # HuggingFace tokenizer (downloaded, not committed)
└── src/
    ├── main.rs          # Server bootstrap: ORT init → model registry → Axum server
    ├── model.rs         # OnnxModel + SessionPool + ModelRegistry: loads and manages multiple models
    ├── catalog.rs       # Known models catalog for the download command
    ├── download.rs      # Model downloader: fetches from HuggingFace
    ├── embedding.rs     # Tokenization → tensor creation → ONNX inference → mean pooling
    └── routes.rs        # HTTP handlers: POST /v1/embeddings, GET /v1/models, GET /health
```

## 🔬 How It Works

1. **Startup:** Loads the ONNX Runtime DLL via `ort::init_from()` *before* starting the async Tokio runtime (avoids a known deadlock in the `ort` crate's dynamic loading)
2. **Model loading:** Scans `models/` for subdirectories, each containing `model.onnx` and `tokenizer.json`. Loads all valid models into a `ModelRegistry`. Each model creates a pool of ONNX sessions (`--pool-size`) to allow concurrent inference.
3. **Request handling:** For each `/v1/embeddings` request:
   - Tokenizes input text(s) using the HuggingFace tokenizer
   - Builds padded tensors for `input_ids`, `attention_mask`, and `token_type_ids`
   - Runs ONNX inference on a session from the pool via `spawn_blocking` (avoids blocking the async runtime)
   - Applies mean pooling with attention mask over the hidden states
   - Returns embedding vectors (dimensions depend on the model)

## 📦 Dependencies

| Crate | Purpose |
|-------|---------|
| [ort](https://github.com/pykeio/ort) | ONNX Runtime Rust bindings with dynamic DLL loading |
| [axum](https://crates.io/crates/axum) 0.7 | Async HTTP framework |
| [tokio](https://crates.io/crates/tokio) 1 | Async runtime |
| [tokenizers](https://crates.io/crates/tokenizers) 0.20 | HuggingFace tokenizer (fast Rust implementation) |
| [ndarray](https://crates.io/crates/ndarray) 0.16 | N-dimensional arrays for mean pooling math |
| [serde](https://crates.io/crates/serde) / [serde_json](https://crates.io/crates/serde_json) | JSON serialization |
| [tracing](https://crates.io/crates/tracing) / [tracing-subscriber](https://crates.io/crates/tracing-subscriber) | Structured logging |
| [anyhow](https://crates.io/crates/anyhow) | Error handling |

> **Note:** The `ort` crate is pinned to the git `main` branch (not the crates.io release) because v2.0.0-rc.12 has a [deadlock bug (#560)](https://github.com/pykeio/ort/issues/560) in the `load-dynamic` feature that causes hangs on startup.

## 🔍 Troubleshooting

### Server hangs on startup

**Cause:** The Windows-bundled `onnxruntime.dll` in `C:\Windows\System32` (version 5.x) is incompatible with the `ort` crate, which expects version 1.24.x. When the wrong DLL is loaded, it triggers a deadlock.

**Fix:** Always set `ORT_DYLIB_PATH` to the full path of the correct 1.24.x DLL:

```powershell
$env:ORT_DYLIB_PATH = "C:\code\onnx-http\onnxruntime.dll"
```

### "Model not found"

The server scans `models/` for subdirectories containing `model.onnx` and `tokenizer.json`. Make sure each model is in its own subdirectory. Always run the server from the project root:

```powershell
cd C:\code\onnx-http
.\target\release\onnx-http.exe
```

Use `GET /v1/models` to see which models were loaded successfully.

### "Access is denied" when building

A previous server process may still be running and locking the executable. Find and stop it:

```powershell
Get-Process -Name "onnx-http" -ErrorAction SilentlyContinue | Stop-Process -Force
```

### QNN execution provider not available

QNN is opt-in (`--npu`). Without it, the server uses CPU. If you enable QNN but the QNN-enabled ORT build isn't installed, the server falls back to CPU gracefully.

### WSL cannot reach the server

- Confirm the server is running: `curl http://localhost:8901/health` from Windows
- Check Windows Firewall isn't blocking port 8901
- On older WSL 1 versions, use the Windows host IP instead of `localhost`

### 🔄 Using additional models

You can load any ONNX model that outputs hidden states in shape `[batch, sequence_length, hidden_dim]`. Create a new subdirectory under `models/` with `model.onnx` and `tokenizer.json`. The directory name becomes the model identifier used in API requests. The server applies mean pooling over the sequence dimension to produce the final embedding vectors.

## 🪟 Running as a Windows Service

You can install onnx-http as a Windows Service that starts automatically on boot using [NSSM](https://nssm.cc/):

```powershell
# Run as Administrator
.\install-service.ps1
```

The script will:
1. Install NSSM via `winget` if not present
2. Ask whether to use NPU or CPU mode
3. Register the service with auto-start, log rotation, and auto-restart on failure
4. Start the service immediately

**Manage the service:**

```powershell
nssm status onnx-http          # Check status
nssm stop onnx-http            # Stop
nssm start onnx-http           # Start
nssm restart onnx-http         # Restart
nssm remove onnx-http confirm  # Uninstall
```

Service logs are written to the `logs/` directory in the project root.

## 📄 License

MIT — see [LICENSE.md](LICENSE.md)

## ⚠️ Disclaimer

This is a personal open‑source project. It is not affiliated with, endorsed by, or an official product of Microsoft. Any internal experimentation does not imply product adoption.

---

Made with ❤️ by Wictor Wilén
