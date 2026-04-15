# onnx-http

A Rust-based HTTP server that loads an ONNX model, runs inference via ONNX Runtime with optional NPU acceleration (QNNExecutionProvider), and exposes an Ollama-compatible `/v1/embeddings` endpoint. Designed to run natively on Windows ARM64 and be called from WSL or any HTTP client.

## Features

- **ONNX Runtime inference** with automatic execution provider fallback (QNN → CPU)
- **Dynamic loading** — drop ONNX Runtime and QNN DLLs next to the executable
- **HuggingFace tokenizer** support (any `tokenizer.json`)
- **Mean pooling** over hidden states with attention mask
- **Single and batch** embedding requests
- **Structured logging** via `tracing`
- **Health check** endpoint

## Prerequisites

- **Rust** (stable, 1.75+) — [install](https://rustup.rs/)
- **ONNX Runtime 1.24.x** shared library (`onnxruntime.dll`) — [download](https://github.com/microsoft/onnxruntime/releases/tag/v1.24.4)
  - For Windows ARM64: download `onnxruntime-win-arm64x-1.24.4.zip`
- An ONNX model that outputs hidden states in shape `[batch, sequence, hidden_dim]`
- A HuggingFace `tokenizer.json` matching your model

### Recommended Model

[sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) — a purpose-built 384-dim embedding model.

Download model files:

```bash
pip install huggingface_hub
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('sentence-transformers/all-MiniLM-L6-v2', 'onnx/model.onnx', local_dir='models', local_dir_use_symlinks=False)"
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('sentence-transformers/all-MiniLM-L6-v2', 'tokenizer.json', local_dir='models', local_dir_use_symlinks=False)"
```

Then move/rename as needed so the files are at `models/model.onnx` and `models/tokenizer.json`.

### Optional (for NPU acceleration)

- **Qualcomm QNN SDK** — required for `QNNExecutionProvider` on Snapdragon/ARM64 devices
- Place QNN DLLs (`QnnHtp.dll`, `QnnSystem.dll`, etc.) in the same directory as the executable or on `PATH`
- Set `USE_QNN=1` environment variable to enable

## Quick Start

### 1. Clone and build

```bash
git clone <repo-url>
cd onnx-http
cargo build --release
```

### 2. Add your model files

```
models/
├── model.onnx        # Your ONNX embedding model
└── tokenizer.json    # HuggingFace tokenizer config
```

### 3. Add ONNX Runtime library

Download ONNX Runtime **1.24.x** for your platform and place `onnxruntime.dll` either:

- Next to the compiled binary (`target/release/`), **or**
- Set `ORT_DYLIB_PATH` to the full path:

```powershell
$env:ORT_DYLIB_PATH = "C:\path\to\onnxruntime.dll"
```

> **⚠️ Important:** Windows ships a bundled `onnxruntime.dll` in System32 that is incompatible. Always use `ORT_DYLIB_PATH` to point to the correct 1.24.x DLL, or place it next to your executable.

### 4. Run the server

```powershell
$env:ORT_DYLIB_PATH = "C:\path\to\onnxruntime.dll"
cargo run --release
```

The server starts on `http://0.0.0.0:8901`.

## API Reference

### `POST /v1/embeddings`

Generate embeddings for one or more text inputs.

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
  -d '{"input": ["Hello, world!", "How are you?", "ONNX is great"]}'
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
      "embedding": [0.0123, -0.0456, 0.0789, ...]
    }
  ]
}
```

### `GET /health`

```bash
curl http://localhost:8901/health
# {"status":"ok"}
```

## WSL ↔ Windows Interop

The server binds to `0.0.0.0:8901`, making it accessible from WSL via `localhost`:

```bash
# From WSL
curl http://localhost:8901/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "hello from WSL"}'
```

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Server listen port | `8901` |
| `ORT_DYLIB_PATH` | Path to `onnxruntime.dll` | Auto-detected next to exe |
| `USE_QNN` | Set to `1` to enable QNN NPU acceleration | Disabled |
| `RUST_LOG` | Log level (`trace`, `debug`, `info`, `warn`, `error`) | `info` |

## Project Structure

```
onnx-http/
├── Cargo.toml
├── models/
│   ├── model.onnx         # ONNX model (user-supplied)
│   └── tokenizer.json     # HuggingFace tokenizer (user-supplied)
└── src/
    ├── main.rs            # Server bootstrap, ORT init, Axum server
    ├── model.rs           # OnnxModel: ONNX session + tokenizer loading
    ├── embedding.rs       # Tokenization, inference, mean pooling
    └── routes.rs          # HTTP handlers and JSON types
```

## Dependencies

| Crate | Purpose |
|-------|---------|
| [ort](https://github.com/pykeio/ort) | ONNX Runtime bindings (dynamic loading) |
| [axum](https://crates.io/crates/axum) | HTTP framework |
| [tokio](https://crates.io/crates/tokio) | Async runtime |
| [tokenizers](https://crates.io/crates/tokenizers) | HuggingFace tokenizer |
| [ndarray](https://crates.io/crates/ndarray) | N-dimensional arrays for pooling |
| [serde](https://crates.io/crates/serde) / [serde_json](https://crates.io/crates/serde_json) | JSON serialization |
| [tracing](https://crates.io/crates/tracing) | Structured logging |
| [anyhow](https://crates.io/crates/anyhow) | Error handling |

## Troubleshooting

### Server hangs on startup

The Windows-bundled `onnxruntime.dll` (in System32) is incompatible and causes a deadlock. Always set `ORT_DYLIB_PATH` to point to the proper ONNX Runtime 1.24.x DLL.

### "Model file not found"

Run the server from the project root so `models/` is in the working directory.

### QNN execution provider not available

QNN is opt-in (`USE_QNN=1`). The server uses CPU by default and falls back gracefully.

### WSL cannot reach the server

- Confirm the server is running: `curl http://localhost:8901/health` from Windows
- On older WSL versions, use the Windows host IP instead of `localhost`

## License

MIT
