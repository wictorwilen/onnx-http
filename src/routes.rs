use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::embedding;
use crate::model::OnnxModel;

// --- Request / Response types ---

#[derive(Deserialize)]
pub struct EmbeddingRequest {
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
    pub model: &'static str,
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
}

// --- Handlers ---

pub async fn health() -> Json<HealthResponse> {
    Json(HealthResponse { status: "ok" })
}

pub async fn embeddings(
    State(model): State<Arc<OnnxModel>>,
    Json(payload): Json<EmbeddingRequest>,
) -> Result<Json<EmbeddingResponse>, (StatusCode, Json<ErrorResponse>)> {
    let texts: Vec<String> = match payload.input {
        EmbeddingInput::Single(s) => {
            if s.is_empty() {
                return Err((
                    StatusCode::BAD_REQUEST,
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
                    Json(ErrorResponse {
                        error: "Input batch must not be empty".into(),
                    }),
                ));
            }
            v
        }
    };

    info!("Embedding request: {} text(s)", texts.len());

    let results = embedding::embed_batch(&model, &texts).map_err(|e| {
        tracing::error!("Embedding failed: {e}");
        (
            StatusCode::INTERNAL_SERVER_ERROR,
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
        model: "all-MiniLM-L6-v2",
        data,
    }))
}

impl IntoResponse for ErrorResponse {
    fn into_response(self) -> axum::response::Response {
        (StatusCode::INTERNAL_SERVER_ERROR, Json(self)).into_response()
    }
}
