use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use ort::session::Session;
use tokenizers::Tokenizer;
use tracing::{info, warn};

pub struct OnnxModel {
    pub session: Mutex<Session>,
    pub tokenizer: Tokenizer,
}

impl OnnxModel {
    pub fn load(model_path: &Path, tokenizer_path: &Path, use_qnn: bool) -> anyhow::Result<Arc<Self>> {
        info!("Loading tokenizer from {}", tokenizer_path.display());
        let mut tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;

        // Enable truncation to the model's max sequence length (512 for BERT-style models).
        // Without this, inputs longer than 512 tokens cause ONNX shape mismatch errors.
        tokenizer.with_truncation(Some(tokenizers::TruncationParams {
            max_length: 512,
            ..Default::default()
        })).map_err(|e| anyhow::anyhow!("Failed to set truncation: {e}"))?;

        info!("Loading ONNX model from {}", model_path.display());

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
            info!("Execution provider: CPUExecutionProvider (use --npu to enable NPU)");
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

pub struct ModelRegistry {
    models: HashMap<String, Arc<OnnxModel>>,
    default_model: String,
}

impl ModelRegistry {
    /// Scan the models directory and load all valid model subdirectories.
    /// Each subdirectory must contain `model.onnx` and `tokenizer.json`.
    /// Falls back to flat layout (`models/model.onnx`) as model named "default".
    pub fn load_all(models_dir: &Path, use_qnn: bool) -> anyhow::Result<Arc<Self>> {
        let mut models = HashMap::new();

        // Try subdirectory layout first
        if models_dir.is_dir() {
            let mut entries: Vec<_> = std::fs::read_dir(models_dir)?
                .filter_map(|e| e.ok())
                .filter(|e| e.path().is_dir())
                .collect();
            entries.sort_by_key(|e| e.file_name());

            for entry in entries {
                let dir = entry.path();
                let model_path = dir.join("model.onnx");
                let tokenizer_path = dir.join("tokenizer.json");

                if !model_path.exists() || !tokenizer_path.exists() {
                    continue;
                }

                let name = dir
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string();

                info!("Loading model '{name}' from {}", dir.display());
                match OnnxModel::load(&model_path, &tokenizer_path, use_qnn) {
                    Ok(model) => {
                        models.insert(name, model);
                    }
                    Err(e) => {
                        warn!("Failed to load model '{}': {e}", dir.display());
                    }
                }
            }
        }

        // Backward compatibility: flat layout (models/model.onnx)
        if models.is_empty() {
            let model_path = models_dir.join("model.onnx");
            let tokenizer_path = models_dir.join("tokenizer.json");

            if model_path.exists() && tokenizer_path.exists() {
                info!("Loading model 'default' from flat layout");
                let model = OnnxModel::load(&model_path, &tokenizer_path, use_qnn)?;
                models.insert("default".to_string(), model);
            }
        }

        if models.is_empty() {
            anyhow::bail!(
                "No models found. Place model subdirectories in {} \
                 (each containing model.onnx and tokenizer.json).",
                models_dir.display()
            );
        }

        // Default: use --default-model if set, otherwise first alphabetically
        let default_model = std::env::var("DEFAULT_MODEL").unwrap_or_else(|_| {
            let mut names: Vec<_> = models.keys().cloned().collect();
            names.sort();
            names[0].clone()
        });

        if !models.contains_key(&default_model) {
            anyhow::bail!(
                "Default model '{default_model}' not found. Available: {:?}",
                models.keys().collect::<Vec<_>>()
            );
        }

        info!(
            "Loaded {} model(s): {:?} (default: '{}')",
            models.len(),
            models.keys().collect::<Vec<_>>(),
            default_model
        );

        Ok(Arc::new(Self {
            models,
            default_model,
        }))
    }

    pub fn get(&self, name: &str) -> Option<&Arc<OnnxModel>> {
        self.models.get(name)
    }

    pub fn default_model_name(&self) -> &str {
        &self.default_model
    }

    pub fn model_names(&self) -> Vec<&String> {
        let mut names: Vec<_> = self.models.keys().collect();
        names.sort();
        names
    }
}
