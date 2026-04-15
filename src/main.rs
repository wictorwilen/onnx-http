mod embedding;
mod model;
mod routes;

use std::path::PathBuf;

use axum::routing::{get, post};
use axum::Router;
use tokio::net::TcpListener;
use tracing::info;

#[derive(Debug, Clone, Copy, PartialEq)]
enum ExecutionProvider {
    Cpu,
    Npu,
}

fn print_usage() {
    eprintln!("Usage: onnx-http [OPTIONS]");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --npu       Use QNN NPU execution provider (with CPU fallback)");
    eprintln!("  --cpu       Use CPU execution provider only (default)");
    eprintln!("  --port N    Server listen port (default: 8901, or PORT env var)");
    eprintln!("  --help      Show this help message");
    eprintln!();
    eprintln!("Environment variables:");
    eprintln!("  ORT_DYLIB_PATH  Path to onnxruntime.dll (v1.24.x)");
    eprintln!("  PORT            Server listen port (overridden by --port)");
    eprintln!("  RUST_LOG        Log level (trace, debug, info, warn, error)");
}

struct CliArgs {
    provider: ExecutionProvider,
    port: String,
}

fn parse_args() -> anyhow::Result<CliArgs> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut provider = ExecutionProvider::Cpu;
    let mut port: Option<String> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--npu" => provider = ExecutionProvider::Npu,
            "--cpu" => provider = ExecutionProvider::Cpu,
            "--port" => {
                i += 1;
                if i >= args.len() {
                    anyhow::bail!("--port requires a value");
                }
                port = Some(args[i].clone());
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown option: {other}");
                print_usage();
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let port = port.unwrap_or_else(|| {
        std::env::var("PORT").unwrap_or_else(|_| "8901".to_string())
    });

    Ok(CliArgs { provider, port })
}

fn init_ort_and_model(provider: ExecutionProvider) -> anyhow::Result<std::sync::Arc<model::OnnxModel>> {
    let model_path = PathBuf::from("models/model.onnx");
    let tokenizer_path = PathBuf::from("models/tokenizer.json");

    if !model_path.exists() {
        anyhow::bail!(
            "Model file not found at {}. Place your ONNX model there.",
            model_path.display()
        );
    }
    if !tokenizer_path.exists() {
        anyhow::bail!(
            "Tokenizer file not found at {}. Place your HuggingFace tokenizer.json there.",
            tokenizer_path.display()
        );
    }

    // Force load the correct ORT DLL before any ort API calls
    let dylib_path = std::env::var("ORT_DYLIB_PATH")
        .unwrap_or_else(|_| {
            let exe_dir = std::env::current_exe()
                .ok()
                .and_then(|p| p.parent().map(|d| d.to_path_buf()))
                .unwrap_or_else(|| PathBuf::from("."));
            exe_dir.join("onnxruntime.dll").to_string_lossy().to_string()
        });
    info!("Initializing ONNX Runtime from {dylib_path}");
    ort::init_from(&dylib_path)
        .map_err(|e| anyhow::anyhow!("Failed to init ORT from {dylib_path}: {e}"))?
        .commit();
    info!("ONNX Runtime loaded successfully");

    let use_qnn = provider == ExecutionProvider::Npu;
    model::OnnxModel::load(&model_path, &tokenizer_path, use_qnn)
}

fn main() -> anyhow::Result<()> {
    let cli = parse_args()?;

    tracing_subscriber::fmt()
        .with_target(false)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    info!("Execution provider: {:?}", cli.provider);

    // Load model BEFORE starting tokio runtime
    let model = init_ort_and_model(cli.provider)?;
    info!("Model and tokenizer loaded successfully");

    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(async {
            let app = Router::new()
                .route("/v1/embeddings", post(routes::embeddings))
                .route("/health", get(routes::health))
                .with_state(model);

            let addr = format!("0.0.0.0:{}", cli.port);
            info!("Starting ONNX embedding server on http://{addr}");
            let listener = TcpListener::bind(&addr).await?;
            axum::serve(listener, app).await?;
            Ok::<(), anyhow::Error>(())
        })?;

    Ok(())
}
