# onnx-http

A Rust-based HTTP server that loads an ONNX embedding model ([all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)), runs inference via ONNX Runtime with optional NPU acceleration (QNNExecutionProvider), and exposes an OpenAI-compatible `/v1/embeddings` endpoint. Designed to run natively on **Windows ARM64** (Snapdragon) and serve embedding requests from WSL, scripts, or any HTTP client.

## Features

- **384-dimensional sentence embeddings** using all-MiniLM-L6-v2
- **ONNX Runtime** inference with dynamic library loading
- **NPU acceleration** via QNNExecutionProvider (opt-in, for Snapdragon devices)
- **CPU fallback** — works on any Windows machine
- **Single and batch** embedding requests
- **OpenAI-compatible** `/v1/embeddings` response format
- **Structured logging** via `tracing`
- **Health check** endpoint at `/health`

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **Rust** (stable 1.75+) | [Install via rustup](https://rustup.rs/) |
| **Python 3.x** | For downloading model files (one-time setup) |
| **ONNX Runtime 1.24.x** | DLL downloaded automatically by setup script |
| **C++ Build Tools** | Required by Rust linker — install via [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) |

## Quick Start (Automated)

The easiest way to get everything set up is with the included setup script:

```powershell
.\setup.ps1
```

This will:
1. Check that Rust and Python are installed
2. Download the **all-MiniLM-L6-v2** ONNX model and tokenizer from HuggingFace
3. Download **ONNX Runtime 1.24.4** for Windows ARM64
4. Build the project in release mode
5. Show you the exact command to start the server

Then start the server:

```powershell
$env:ORT_DYLIB_PATH = "$PWD\onnxruntime.dll"
.\target\release\onnx-http.exe           # CPU mode (default)
.\target\release\onnx-http.exe --npu     # NPU mode (Snapdragon)
```

## Manual Setup

If you prefer to set things up step by step:

### 1. Clone and enter the project

```powershell
git clone <repo-url>
cd onnx-http
```

### 2. Download the ONNX model

The server uses [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), a purpose-built sentence embedding model that produces 384-dimensional vectors.

```powershell
# Install the HuggingFace Hub Python package
pip install huggingface_hub

# Download model.onnx (~90 MB)
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('sentence-transformers/all-MiniLM-L6-v2', 'onnx/model.onnx', local_dir='models', local_dir_use_symlinks=False)"

# Download tokenizer.json (~466 KB)
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('sentence-transformers/all-MiniLM-L6-v2', 'tokenizer.json', local_dir='models', local_dir_use_symlinks=False)"

# The model downloads into models/onnx/model.onnx — move it up
Move-Item models\onnx\model.onnx models\model.onnx
Remove-Item models\onnx -Recurse
```

After this step your `models/` directory should contain:

```
models/
├── model.onnx        # ~90 MB ONNX model
└── tokenizer.json    # ~466 KB HuggingFace tokenizer
```

### 3. Download ONNX Runtime 1.24.x

The `ort` crate requires ONNX Runtime **1.24.x**. Download the correct build for your platform:

| Platform | Download |
|----------|----------|
| **Windows ARM64** (Snapdragon) | [onnxruntime-win-arm64x-1.24.4.zip](https://github.com/microsoft/onnxruntime/releases/download/v1.24.4/onnxruntime-win-arm64x-1.24.4.zip) |
| **Windows x64** | [onnxruntime-win-x64-1.24.4.zip](https://github.com/microsoft/onnxruntime/releases/download/v1.24.4/onnxruntime-win-x64-1.24.4.zip) |

Extract `onnxruntime.dll` from the `lib/` folder inside the zip and place it in the project root:

```powershell
# Example for ARM64
Invoke-WebRequest -Uri "https://github.com/microsoft/onnxruntime/releases/download/v1.24.4/onnxruntime-win-arm64x-1.24.4.zip" -OutFile ort.zip
Expand-Archive ort.zip -DestinationPath ort-extract
Copy-Item ort-extract\onnxruntime-win-arm64x-1.24.4\lib\onnxruntime.dll .
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

## API Reference

### `POST /v1/embeddings`

Generate embeddings for one or more text inputs. Compatible with the OpenAI embeddings API format.

**Single input:**

```bash
curl http://localhost:8901/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello, world!"}'
```

**Batch input:**

```bash
curl http://localhost:8901/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": ["Hello, world!", "How are you?", "Machine learning is great"]}'
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

Each embedding is a 384-dimensional float vector.

**Error response (400):**

```json
{
  "error": "Input text must not be empty"
}
```

### `GET /health`

```bash
curl http://localhost:8901/health
```

```json
{"status": "ok"}
```

## WSL ↔ Windows Interop

The server binds to `0.0.0.0:8901`, making it accessible from WSL via `localhost`:

```bash
# From WSL — single embedding
curl http://localhost:8901/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "hello from WSL"}'

# From WSL — batch of embeddings
curl http://localhost:8901/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": ["sentence one", "sentence two", "sentence three"]}'
```

No Windows-specific APIs are used — pure Rust networking. The standard WSL ↔ Windows localhost bridge works out of the box.

### Python example (from WSL or Windows)

```python
import requests

response = requests.post(
    "http://localhost:8901/v1/embeddings",
    json={"input": ["hello world", "how are you"]},
)
data = response.json()
for item in data["data"]:
    print(f"[{item['index']}] {len(item['embedding'])} dims, first 3: {item['embedding'][:3]}")
```

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `ORT_DYLIB_PATH` | **Required.** Full path to `onnxruntime.dll` (v1.24.x) | Auto-detects next to exe |
| `PORT` | Server listen port (overridden by `--port`) | `8901` |
| `RUST_LOG` | Log level (`trace`, `debug`, `info`, `warn`, `error`) | `info` |

### Command-line options

| Flag | Description |
|------|-------------|
| `--npu` | Use QNN NPU execution provider (with CPU fallback) |
| `--cpu` | Use CPU execution provider only (default) |
| `--port N` | Server listen port (overrides `PORT` env var) |
| `--help` | Show help message |

### NPU acceleration (Snapdragon devices)

If you have a Qualcomm Snapdragon device with NPU, the setup script automatically downloads the QNN-enabled ONNX Runtime build (from the `Microsoft.ML.OnnxRuntime.QNN` NuGet package), which includes all required QNN DLLs.

```powershell
$env:ORT_DYLIB_PATH = "$PWD\onnxruntime.dll"
.\target\release\onnx-http.exe --npu
```

The server will attempt QNNExecutionProvider first, falling back to CPU if unavailable.

## Project Structure

```
onnx-http/
├── Cargo.toml           # Dependencies and project config
├── setup.ps1            # Automated setup script
├── onnxruntime.dll      # ONNX Runtime 1.24.x (downloaded, not committed)
├── models/
│   ├── model.onnx       # all-MiniLM-L6-v2 ONNX model (downloaded, not committed)
│   └── tokenizer.json   # HuggingFace tokenizer (downloaded, not committed)
└── src/
    ├── main.rs          # Server bootstrap: ORT init → model load → Axum server
    ├── model.rs         # OnnxModel: wraps Mutex<Session> + Tokenizer
    ├── embedding.rs     # Tokenization → tensor creation → ONNX inference → mean pooling
    └── routes.rs        # HTTP handlers: POST /v1/embeddings, GET /health
```

## How It Works

1. **Startup:** Loads the ONNX Runtime DLL via `ort::init_from()` *before* starting the async Tokio runtime (avoids a known deadlock in the `ort` crate's dynamic loading)
2. **Model loading:** Creates an ONNX Runtime session from `models/model.onnx` and a HuggingFace tokenizer from `models/tokenizer.json`
3. **Request handling:** For each `/v1/embeddings` request:
   - Tokenizes input text(s) using the HuggingFace tokenizer
   - Builds padded tensors for `input_ids`, `attention_mask`, and `token_type_ids`
   - Runs ONNX inference (thread-safe via `Mutex<Session>`)
   - Applies mean pooling with attention mask over the hidden states
   - Returns 384-dimensional embedding vectors

## Dependencies

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

## Troubleshooting

### Server hangs on startup

**Cause:** The Windows-bundled `onnxruntime.dll` in `C:\Windows\System32` (version 5.x) is incompatible with the `ort` crate, which expects version 1.24.x. When the wrong DLL is loaded, it triggers a deadlock.

**Fix:** Always set `ORT_DYLIB_PATH` to the full path of the correct 1.24.x DLL:

```powershell
$env:ORT_DYLIB_PATH = "C:\code\onnx-http\onnxruntime.dll"
```

### "Model file not found"

The server looks for `models/model.onnx` and `models/tokenizer.json` relative to the **working directory**. Always run the server from the project root:

```powershell
cd C:\code\onnx-http
.\target\release\onnx-http.exe
```

### "Access is denied" when building

A previous server process may still be running and locking the executable. Find and stop it:

```powershell
Get-Process -Name "onnx-http" -ErrorAction SilentlyContinue | Stop-Process -Force
```

### QNN execution provider not available

QNN is opt-in (`USE_QNN=1`). Without it, the server uses CPU. If you enable QNN but don't have the QNN SDK installed, the server falls back to CPU gracefully.

### WSL cannot reach the server

- Confirm the server is running: `curl http://localhost:8901/health` from Windows
- Check Windows Firewall isn't blocking port 8901
- On older WSL 1 versions, use the Windows host IP instead of `localhost`

### Wrong embedding dimensions

If you're using a different ONNX model, ensure it outputs hidden states in shape `[batch, sequence_length, hidden_dim]`. The server applies mean pooling over the sequence dimension.

## License

MIT
