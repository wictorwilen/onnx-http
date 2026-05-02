use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::embedding;
use crate::model::ModelRegistry;

// --- Request / Response types ---

#[derive(Deserialize)]
pub struct EmbeddingRequest {
    pub model: String,
    pub input: EmbeddingInput,
}

#[derive(Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    Single(String),
    Batch(Vec<String>),
}

#[derive(Serialize)]
pub struct EmbeddingResponse {
    pub object: &'static str,
    pub model: String,
    pub data: Vec<EmbeddingData>,
}

#[derive(Serialize)]
pub struct EmbeddingData {
    pub object: &'static str,
    pub index: usize,
    pub embedding: Vec<f32>,
}

#[derive(Serialize)]
pub struct ErrorResponse {
    pub error: String,
}

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
    pub models: Vec<String>,
}

// --- /v1/models types (OpenAI-compatible) ---

#[derive(Serialize)]
pub struct ModelsResponse {
    pub object: &'static str,
    pub data: Vec<ModelInfo>,
}

#[derive(Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: &'static str,
    pub owned_by: &'static str,
}

// --- Handlers ---

pub async fn health(State(registry): State<Arc<ModelRegistry>>) -> impl IntoResponse {
    if registry.is_ready() {
        let body = HealthResponse {
            status: "ok",
            models: registry.model_names(),
        };
        (StatusCode::OK, Json(body)).into_response()
    } else {
        let body = HealthResponse {
            status: "loading",
            models: vec![],
        };
        (
            StatusCode::OK,
            [("Retry-After", "5")],
            Json(body),
        ).into_response()
    }
}

pub async fn list_models(
    State(registry): State<Arc<ModelRegistry>>,
) -> Result<Json<ModelsResponse>, (StatusCode, [(&'static str, &'static str); 1], Json<ErrorResponse>)> {
    if !registry.is_ready() {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            [("Retry-After", "5")],
            Json(ErrorResponse {
                error: "Models are still loading, please retry shortly".into(),
            }),
        ));
    }

    let data = registry
        .model_names()
        .into_iter()
        .map(|name| ModelInfo {
            id: name,
            object: "model",
            owned_by: "local",
        })
        .collect();

    Ok(Json(ModelsResponse {
        object: "list",
        data,
    }))
}

pub async fn embeddings(
    State(registry): State<Arc<ModelRegistry>>,
    Json(payload): Json<EmbeddingRequest>,
) -> Result<Json<EmbeddingResponse>, (StatusCode, [(&'static str, &'static str); 1], Json<ErrorResponse>)> {
    if !registry.is_ready() {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            [("Retry-After", "5")],
            Json(ErrorResponse {
                error: "Models are still loading, please retry shortly".into(),
            }),
        ));
    }

    let model_name = payload.model;

    let model = registry.get(&model_name).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            [("Retry-After", "0")],
            Json(ErrorResponse {
                error: format!(
                    "Model '{}' not found. Available models: {:?}",
                    model_name,
                    registry.model_names()
                ),
            }),
        )
    })?;

    let texts: Vec<String> = match payload.input {
        EmbeddingInput::Single(s) => {
            if s.is_empty() {
                return Err((
                    StatusCode::BAD_REQUEST,
                    [("Retry-After", "0")],
                    Json(ErrorResponse {
                        error: "Input text must not be empty".into(),
                    }),
                ));
            }
            vec![s]
        }
        EmbeddingInput::Batch(v) => {
            if v.is_empty() {
                return Err((
                    StatusCode::BAD_REQUEST,
                    [("Retry-After", "0")],
                    Json(ErrorResponse {
                        error: "Input batch must not be empty".into(),
                    }),
                ));
            }
            v
        }
    };

    info!("Embedding request: model='{}', {} text(s)", model_name, texts.len());

    let model = model.clone();
    let results = tokio::task::spawn_blocking(move || {
        embedding::embed_batch(&model, &texts)
    })
    .await
    .map_err(|e| {
        tracing::error!("Spawn blocking failed: {e}");
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            [("Retry-After", "0")],
            Json(ErrorResponse {
                error: format!("Internal error: {e}"),
            }),
        )
    })?
    .map_err(|e| {
        tracing::error!("Embedding failed: {e}");
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            [("Retry-After", "0")],
            Json(ErrorResponse {
                error: format!("Embedding failed: {e}"),
            }),
        )
    })?;

    let data: Vec<EmbeddingData> = results
        .into_iter()
        .enumerate()
        .map(|(i, emb)| EmbeddingData {
            object: "embedding",
            index: i,
            embedding: emb,
        })
        .collect();

    Ok(Json(EmbeddingResponse {
        object: "list",
        model: model_name,
        data,
    }))
}

impl IntoResponse for ErrorResponse {
    fn into_response(self) -> axum::response::Response {
        (StatusCode::INTERNAL_SERVER_ERROR, Json(self)).into_response()
    }
}
