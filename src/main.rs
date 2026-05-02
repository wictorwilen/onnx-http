mod catalog;
mod download;
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
    eprintln!("onnx-http — ONNX Embedding Server");
    eprintln!("Made with ❤️ by Wictor Wilén");
    eprintln!();
    eprintln!("Usage:");
    eprintln!("  onnx-http [OPTIONS]              Start the embedding server");
    eprintln!("  onnx-http models                 List installed models");
    eprintln!("  onnx-http download <model>       Download a model from HuggingFace");
    eprintln!("  onnx-http download --list        List available models to download");
    eprintln!();
    eprintln!("Server options:");
    eprintln!("  --npu          Use QNN NPU execution provider (with CPU fallback)");
    eprintln!("  --cpu          Use CPU execution provider only (default)");
    eprintln!("  --port N       Server listen port (default: 8901, or PORT env var)");
    eprintln!("  --pool-size N  Number of inference sessions per model (default: 2 NPU, 4 CPU)");
    eprintln!("  --help         Show this help message");
    eprintln!();
    eprintln!("Model layout:");
    eprintln!("  Place each model in a subdirectory under models/:");
    eprintln!("    models/all-MiniLM-L6-v2/model.onnx");
    eprintln!("    models/all-MiniLM-L6-v2/tokenizer.json");
    eprintln!();
    eprintln!("Environment variables:");
    eprintln!("  ORT_DYLIB_PATH   Path to onnxruntime.dll (v1.24.x)");
    eprintln!("  PORT             Server listen port (overridden by --port)");
    eprintln!("  DEFAULT_MODEL    Default model name when multiple are loaded");
    eprintln!("  RUST_LOG         Log level (trace, debug, info, warn, error)");
}

enum Command {
    Serve(ServeArgs),
    Download(String),
    ListCatalog,
    ListInstalled,
}

struct ServeArgs {
    provider: ExecutionProvider,
    port: String,
    pool_size: usize,
}

fn parse_args() -> anyhow::Result<Command> {
    let args: Vec<String> = std::env::args().skip(1).collect();

    if args.is_empty() {
        return Ok(Command::Serve(ServeArgs {
            provider: ExecutionProvider::Cpu,
            port: std::env::var("PORT").unwrap_or_else(|_| "8901".to_string()),
            pool_size: 0, // 0 = auto-detect
        }));
    }

    // Check for subcommands
    if args[0] == "download" {
        if args.len() < 2 || args[1] == "--list" {
            return Ok(Command::ListCatalog);
        }
        return Ok(Command::Download(args[1].clone()));
    }

    if args[0] == "models" {
        return Ok(Command::ListInstalled);
    }

    // Parse serve options
    let mut provider = ExecutionProvider::Cpu;
    let mut port: Option<String> = None;
    let mut pool_size: usize = 0; // 0 = auto-detect

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
            "--pool-size" => {
                i += 1;
                if i >= args.len() {
                    anyhow::bail!("--pool-size requires a value");
                }
                pool_size = args[i].parse::<usize>()
                    .map_err(|_| anyhow::anyhow!("--pool-size must be a positive integer"))?;
                if pool_size == 0 {
                    anyhow::bail!("--pool-size must be >= 1");
                }
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

    Ok(Command::Serve(ServeArgs { provider, port, pool_size }))
}

fn init_ort(provider: ExecutionProvider) -> anyhow::Result<()> {
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
    let _ = provider; // used by caller for QNN config
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let command = parse_args()?;

    match command {
        Command::ListCatalog => {
            download::print_catalog();
            Ok(())
        }
        Command::ListInstalled => {
            download::print_installed();
            Ok(())
        }
        Command::Download(name) => {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()?
                .block_on(download::download_model(&name))
        }
        Command::Serve(cli) => {
            tracing_subscriber::fmt()
                .with_target(false)
                .with_env_filter(
                    tracing_subscriber::EnvFilter::try_from_default_env()
                        .unwrap_or_else(|_| "info".into()),
                )
                .init();

            info!("onnx-http — Made with \u{2764}\u{FE0F} by Wictor Wilén");
            info!("Execution provider: {:?}", cli.provider);

            // Auto-detect pool size: 2 for NPU, 4 for CPU
            let pool_size = if cli.pool_size > 0 {
                cli.pool_size
            } else if cli.provider == ExecutionProvider::Npu {
                2
            } else {
                4
            };
            info!("Session pool size: {}", pool_size);

            // Load ORT DLL (fast) before starting runtime
            init_ort(cli.provider)?;

            // Create empty registry — server starts immediately
            let registry = model::ModelRegistry::new_empty();

            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?
                .block_on(async {
                    // Start loading models in background
                    let registry_bg = registry.clone();
                    let use_qnn = cli.provider == ExecutionProvider::Npu;
                    let models_dir = PathBuf::from("models");
                    tokio::task::spawn_blocking(move || {
                        if let Err(e) = registry_bg.load_all_into(&models_dir, use_qnn, pool_size) {
                            tracing::error!("Failed to load models: {e}");
                            std::process::exit(1);
                        }
                    });

                    let app = Router::new()
                        .route("/v1/embeddings", post(routes::embeddings))
                        .route("/v1/models", get(routes::list_models))
                        .route("/health", get(routes::health))
                        .with_state(registry);

                    let addr = format!("0.0.0.0:{}", cli.port);
                    info!("Starting ONNX embedding server on http://{addr}");
                    info!("Health endpoint available immediately — models loading in background");
                    let listener = TcpListener::bind(&addr).await?;
                    axum::serve(listener, app).await?;
                    Ok::<(), anyhow::Error>(())
                })?;

            Ok(())
        }
    }
}
