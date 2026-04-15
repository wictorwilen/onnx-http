mod embedding;
mod model;
mod routes;

use std::path::PathBuf;

use axum::routing::{get, post};
use axum::Router;
use tokio::net::TcpListener;
use tracing::info;

fn init_ort_and_model() -> anyhow::Result<std::sync::Arc<model::OnnxModel>> {
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

    model::OnnxModel::load(&model_path, &tokenizer_path)
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_target(false)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    // Load model BEFORE starting tokio runtime
    let model = init_ort_and_model()?;
    info!("Model and tokenizer loaded successfully");

    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(async {
            let app = Router::new()
                .route("/v1/embeddings", post(routes::embeddings))
                .route("/health", get(routes::health))
                .with_state(model);

            let port = std::env::var("PORT").unwrap_or_else(|_| "8901".to_string());
            let addr = format!("0.0.0.0:{port}");
            info!("Starting ONNX embedding server on http://{addr}");
            let listener = TcpListener::bind(&addr).await?;
            axum::serve(listener, app).await?;
            Ok::<(), anyhow::Error>(())
        })?;

    Ok(())
}
