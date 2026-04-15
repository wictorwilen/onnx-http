use std::path::Path;
use std::sync::{Arc, Mutex};

use ort::session::Session;
use tokenizers::Tokenizer;
use tracing::info;

pub struct OnnxModel {
    pub session: Mutex<Session>,
    pub tokenizer: Tokenizer,
}

impl OnnxModel {
    pub fn load(model_path: &Path, tokenizer_path: &Path) -> anyhow::Result<Arc<Self>> {
        info!("Loading tokenizer from {}", tokenizer_path.display());
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;

        info!("Loading ONNX model from {}", model_path.display());

        let use_qnn = std::env::var("USE_QNN").unwrap_or_default() == "1";
        let mut builder = Session::builder()
            .map_err(|e| anyhow::anyhow!("Failed to create session builder: {e}"))?;

        if use_qnn {
            info!("Execution providers: QNNExecutionProvider -> CPUExecutionProvider");
            builder = builder
                .with_execution_providers([
                    ort::execution_providers::QNNExecutionProvider::default().build(),
                    ort::execution_providers::CPUExecutionProvider::default().build(),
                ])
                .map_err(|e| anyhow::anyhow!("Failed to set execution providers: {e}"))?;
        } else {
            info!("Execution provider: CPUExecutionProvider (set USE_QNN=1 to enable NPU)");
        }

        let session = builder
            .commit_from_file(model_path)
            .map_err(|e| anyhow::anyhow!("Failed to load model: {e}"))?;

        let input_names: Vec<_> = session
            .inputs()
            .iter()
            .map(|i| i.name().to_string())
            .collect();
        info!("Model loaded. Input names: {:?}", input_names);

        Ok(Arc::new(Self {
            session: Mutex::new(session),
            tokenizer,
        }))
    }
}
