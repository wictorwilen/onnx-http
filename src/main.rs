mod embedding;
mod model;
mod routes;

use std::path::PathBuf;

use axum::routing::{get, post};
use axum::Router;
use tokio::net::TcpListener;
use tracing::info;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_target(false)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

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

    let model = model::OnnxModel::load(&model_path, &tokenizer_path)?;
    info!("Model and tokenizer loaded successfully");

    let app = Router::new()
        .route("/v1/embeddings", post(routes::embeddings))
        .route("/health", get(routes::health))
        .with_state(model);

    let addr = "0.0.0.0:11434";
    info!("Starting ONNX embedding server on http://{addr}");
    let listener = TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
