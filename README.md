# onnx-http

A Rust-based HTTP server that loads an ONNX model (e.g., Phi‑3.5 Mini), runs inference via ONNX Runtime with optional NPU acceleration (QNNExecutionProvider), and exposes an Ollama-compatible `/v1/embeddings` endpoint. Designed to run natively on Windows ARM64 and be called from WSL or any HTTP client.

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
- **ONNX Runtime** shared library (`onnxruntime.dll`) — [download](https://github.com/microsoft/onnxruntime/releases)
- An ONNX model that outputs hidden states in shape `[batch, sequence, hidden_dim]`
- A HuggingFace `tokenizer.json` matching your model

### Optional (for NPU acceleration)

- **Qualcomm QNN SDK** — required for `QNNExecutionProvider` on Snapdragon/ARM64 devices
- Place QNN DLLs (`QnnHtp.dll`, `QnnSystem.dll`, etc.) in the same directory as the executable or on `PATH`

## Quick Start

### 1. Clone and build

```bash
git clone <repo-url>
cd onnx-http
cargo build --release
```

The compiled binary will be at `target/release/onnx-http.exe`.

### 2. Add your model files

Place your ONNX model and tokenizer in the `models/` directory:

```
models/
├── model.onnx        # Your ONNX embedding model
└── tokenizer.json    # HuggingFace tokenizer config
```

> **Tip:** For Phi‑3.5 Mini, download the ONNX variant from [Hugging Face](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) and export/convert to ONNX format with `optimum-cli`.

### 3. Add ONNX Runtime library

The server uses dynamic loading (`load-dynamic` feature). Place `onnxruntime.dll` either:

- In the same directory as `onnx-http.exe`, **or**
- On your system `PATH`

You can set the library path explicitly via the `ORT_DYLIB_PATH` environment variable:

```powershell
$env:ORT_DYLIB_PATH = "C:\path\to\onnxruntime.dll"
```

### 4. Run the server

```powershell
# From the project root (models/ must be in the working directory)
cargo run --release

# Or run the binary directly
.\target\release\onnx-http.exe
```

The server starts on `http://0.0.0.0:11434`.

## API Reference

### `POST /v1/embeddings`

Generate embeddings for one or more text inputs.

**Single input:**

```bash
curl http://localhost:11434/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello, world!"}'
```

**Batch input:**

```bash
curl http://localhost:11434/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": ["Hello, world!", "How are you?", "ONNX is great"]}'
```

**Response:**

```json
{
  "object": "list",
  "model": "phi-3.5-mini-onnx",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.0123, -0.0456, 0.0789, ...]
    }
  ]
}
```

**Error response:**

```json
{
  "error": "Input text must not be empty"
}
```

### `GET /health`

Health check endpoint.

```bash
curl http://localhost:11434/health
```

```json
{
  "status": "ok"
}
```

## WSL ↔ Windows Interop

The server binds to `0.0.0.0:11434`, making it accessible from WSL via `localhost`:

```bash
# From WSL
curl http://localhost:11434/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "hello from WSL"}'
```

No Windows-specific APIs are used — it's pure Rust networking, so the standard WSL ↔ Windows localhost bridge works out of the box.

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `RUST_LOG` | Log level filter (`trace`, `debug`, `info`, `warn`, `error`) | `info` |
| `ORT_DYLIB_PATH` | Path to `onnxruntime.dll` | Auto-detected |

### Examples

```powershell
# Enable debug logging
$env:RUST_LOG = "debug"
cargo run --release

# Specify ONNX Runtime location
$env:ORT_DYLIB_PATH = "C:\onnxruntime\lib\onnxruntime.dll"
cargo run --release
```

## Project Structure

```
onnx-http/
├── Cargo.toml             # Dependencies and project metadata
├── models/
│   ├── model.onnx         # ONNX model (user-supplied)
│   └── tokenizer.json     # HuggingFace tokenizer (user-supplied)
└── src/
    ├── main.rs            # Server bootstrap, startup validation
    ├── model.rs           # OnnxModel: ONNX session + tokenizer loading
    ├── embedding.rs       # Tokenization, inference, mean pooling
    └── routes.rs          # Axum HTTP handlers and JSON types
```

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  HTTP Client (WSL, curl, app)                                │
│  POST /v1/embeddings { "input": "text" }                     │
└─────────────────────┬────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────────┐
│  Axum HTTP Server (0.0.0.0:11434)                            │
│  routes.rs — parse request, validate, return JSON            │
└─────────────────────┬────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────────┐
│  embedding.rs                                                │
│  1. Tokenize (HuggingFace tokenizers)                        │
│  2. Build input tensors (input_ids, attention_mask)           │
│  3. Run ONNX inference (via ort crate)                       │
│  4. Mean pooling with attention mask                          │
└─────────────────────┬────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────────┐
│  ONNX Runtime                                                │
│  Execution Providers: QNN (NPU) → CPU (fallback)             │
│  Model: models/model.onnx                                    │
└──────────────────────────────────────────────────────────────┘
```

## Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| [axum](https://crates.io/crates/axum) | 0.7 | HTTP framework |
| [tokio](https://crates.io/crates/tokio) | 1 | Async runtime |
| [ort](https://crates.io/crates/ort) | 2.0.0-rc.12 | ONNX Runtime bindings (dynamic loading) |
| [tokenizers](https://crates.io/crates/tokenizers) | 0.20 | HuggingFace tokenizer |
| [ndarray](https://crates.io/crates/ndarray) | 0.16 | N-dimensional arrays for pooling |
| [serde](https://crates.io/crates/serde) / [serde_json](https://crates.io/crates/serde_json) | 1 | JSON serialization |
| [tracing](https://crates.io/crates/tracing) / [tracing-subscriber](https://crates.io/crates/tracing-subscriber) | 0.1 / 0.3 | Structured logging |
| [anyhow](https://crates.io/crates/anyhow) | 1 | Error handling |

## Troubleshooting

### "Model file not found"

Ensure `models/model.onnx` exists relative to your working directory. Run the server from the project root.

### "Failed to load model" / ONNX Runtime errors

- Verify `onnxruntime.dll` is discoverable (same dir as exe, on PATH, or via `ORT_DYLIB_PATH`).
- Ensure the DLL architecture matches your platform (ARM64 for Snapdragon devices).

### QNN execution provider not available

- QNN is optional. The server falls back to CPU automatically.
- To enable QNN: install the Qualcomm QNN SDK and place its DLLs alongside the executable.

### WSL cannot reach the server

- Confirm the server is running: `curl http://localhost:11434/health` from Windows.
- Check Windows Firewall isn't blocking port 11434.
- On older WSL versions, you may need to use the Windows host IP instead of `localhost`.

## License

MIT
