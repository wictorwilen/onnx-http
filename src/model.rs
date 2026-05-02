use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use ort::session::Session;
use tokenizers::Tokenizer;
use tracing::{info, warn};

/// A pool of ONNX sessions to allow concurrent inference.
pub struct SessionPool {
    sessions: Vec<Mutex<Session>>,
}

impl SessionPool {
    fn new(sessions: Vec<Session>) -> Self {
        Self {
            sessions: sessions.into_iter().map(Mutex::new).collect(),
        }
    }

    /// Acquire a session from the pool; tries non-blocking first, then blocks.
    pub fn acquire(&self) -> std::sync::MutexGuard<'_, Session> {
        for session in &self.sessions {
            if let Ok(guard) = session.try_lock() {
                return guard;
            }
        }
        // All sessions busy — block on the first one
        self.sessions[0].lock().expect("session mutex poisoned")
    }

    pub fn size(&self) -> usize {
        self.sessions.len()
    }
}

pub struct OnnxModel {
    pub pool: SessionPool,
    pub tokenizer: Tokenizer,
    /// Whether this model accepts token_type_ids as input
    pub has_token_type_ids: bool,
    /// Max sequence length for this model (used as fixed padding length for static-shape models)
    pub max_tokens: usize,
    /// Whether this model uses static shapes (batch=1, seq=max_tokens)
    pub static_shapes: bool,
}

impl OnnxModel {
    pub fn load(model_path: &Path, tokenizer_path: &Path, use_qnn: bool, pool_size: usize) -> anyhow::Result<Arc<Self>> {
        let (max_tokens, static_shapes) = Self::read_model_config(model_path.parent().unwrap_or(model_path));

        info!("Loading tokenizer from {}", tokenizer_path.display());
        let mut tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;

        // Enable truncation to the model's max sequence length.
        // Without this, inputs longer than max_tokens cause ONNX shape mismatch errors.
        tokenizer.with_truncation(Some(tokenizers::TruncationParams {
            max_length: max_tokens,
            ..Default::default()
        })).map_err(|e| anyhow::anyhow!("Failed to set truncation: {e}"))?;

        info!("Loading ONNX model from {} (max_tokens={}, pool_size={})", model_path.display(), max_tokens, pool_size);

        let htp_backend_path = if use_qnn {
            Some(Self::resolve_qnn_backend_path("QnnHtp.dll"))
        } else {
            None
        };

        let mut sessions = Vec::with_capacity(pool_size);
        for i in 0..pool_size {
            let session = Self::build_session(model_path, use_qnn, htp_backend_path.as_deref())?;
            if i == 0 {
                let input_names: Vec<_> = session
                    .inputs()
                    .iter()
                    .map(|n| n.name().to_string())
                    .collect();
                let has_tt = input_names.iter().any(|n| n == "token_type_ids");
                info!("Model inputs: {:?}, has_token_type_ids: {}", input_names, has_tt);
            }
            sessions.push(session);
            if pool_size > 1 {
                info!("Session {}/{} created", i + 1, pool_size);
            }
        }

        let has_token_type_ids = sessions[0]
            .inputs()
            .iter()
            .any(|n| n.name() == "token_type_ids");

        Ok(Arc::new(Self {
            pool: SessionPool::new(sessions),
            tokenizer,
            has_token_type_ids,
            max_tokens,
            static_shapes,
        }))
    }

    /// Build a single ONNX session with the given execution provider config.
    fn build_session(model_path: &Path, use_qnn: bool, htp_backend_path: Option<&Path>) -> anyhow::Result<Session> {
        let mut builder = Session::builder()
            .map_err(|e| anyhow::anyhow!("Failed to create session builder: {e}"))?;

        if use_qnn {
            let htp = htp_backend_path.expect("htp_backend_path required for QNN");
            builder = builder
                .with_intra_threads(1)
                .map_err(|e| anyhow::anyhow!("Failed to set intra threads: {e}"))?
                .with_inter_threads(1)
                .map_err(|e| anyhow::anyhow!("Failed to set inter threads: {e}"))?;

            info!("Execution providers: QNNExecutionProvider (backend={}) -> CPUExecutionProvider", htp.display());

            builder = builder
                .with_execution_providers([
                    ort::execution_providers::QNNExecutionProvider::default()
                        .with_backend_path(htp.to_string_lossy())
                        .with_performance_mode(ort::execution_providers::qnn::PerformanceMode::SustainedHighPerformance)
                        .with_htp_fp16_precision(true)
                        .with_htp_graph_finalization_optimization_mode(3)
                        .build(),
                    ort::execution_providers::CPUExecutionProvider::default().build(),
                ])
                .map_err(|e| anyhow::anyhow!("Failed to set execution providers: {e}"))?;
        } else {
            info!("Execution provider: CPUExecutionProvider (use --npu to enable NPU)");
        }

        builder
            .commit_from_file(model_path)
            .map_err(|e| anyhow::anyhow!("Failed to load model: {e}"))
    }

    /// Resolve the path to a QNN backend DLL next to the executable.
    fn resolve_qnn_backend_path(dll_name: &str) -> PathBuf {
        let exe_dir = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|d| d.to_path_buf()))
            .unwrap_or_else(|| PathBuf::from("."));
        let candidate = exe_dir.join(dll_name);
        if candidate.exists() {
            return candidate;
        }
        // Fall back to current directory
        let cwd_candidate = PathBuf::from(dll_name);
        if cwd_candidate.exists() {
            return std::env::current_dir()
                .unwrap_or_else(|_| PathBuf::from("."))
                .join(dll_name);
        }
        // Return the name as-is, let QNN runtime resolve it
        PathBuf::from(dll_name)
    }

    /// Read max_tokens and static_shapes from model_config.json in the model directory.
    fn read_model_config(model_dir: &Path) -> (usize, bool) {
        let config_path = model_dir.join("model_config.json");
        let mut max_tokens = 512usize;
        let mut static_shapes = false;

        if let Ok(contents) = std::fs::read_to_string(&config_path) {
            // Parse max_tokens
            if let Some(pos) = contents.find("\"max_tokens\"") {
                let rest = &contents[pos..];
                if let Some(colon) = rest.find(':') {
                    let after_colon = rest[colon + 1..].trim();
                    if let Some(end) = after_colon.find(|c: char| !c.is_ascii_digit()) {
                        if let Ok(v) = after_colon[..end].trim().parse::<usize>() {
                            if v > 0 {
                                max_tokens = v;
                            }
                        }
                    } else if let Ok(v) = after_colon.trim_end().parse::<usize>() {
                        if v > 0 {
                            max_tokens = v;
                        }
                    }
                }
            }
            // Parse static_shapes
            if contents.contains("\"static_shapes\"") && contents.contains("true") {
                static_shapes = true;
            }
            info!("Model config: max_tokens={}, static_shapes={} from {}", max_tokens, static_shapes, config_path.display());
        }
        (max_tokens, static_shapes)
    }
}

pub struct ModelRegistry {
    models: RwLock<HashMap<String, Arc<OnnxModel>>>,
    default_model: RwLock<String>,
    ready: AtomicBool,
}

impl ModelRegistry {
    /// Create an empty registry (server can start immediately).
    pub fn new_empty() -> Arc<Self> {
        Arc::new(Self {
            models: RwLock::new(HashMap::new()),
            default_model: RwLock::new(String::new()),
            ready: AtomicBool::new(false),
        })
    }

    /// Whether models have finished loading.
    pub fn is_ready(&self) -> bool {
        self.ready.load(Ordering::Relaxed)
    }

    /// Load all models and mark registry as ready. Call from a background thread.
    pub fn load_all_into(self: &Arc<Self>, models_dir: &Path, use_qnn: bool, pool_size: usize) -> anyhow::Result<()> {
        let mut loaded = HashMap::new();

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
                match OnnxModel::load(&model_path, &tokenizer_path, use_qnn, pool_size) {
                    Ok(model) => {
                        loaded.insert(name, model);
                    }
                    Err(e) => {
                        warn!("Failed to load model '{}': {e}", dir.display());
                    }
                }
            }
        }

        // Backward compatibility: flat layout (models/model.onnx)
        if loaded.is_empty() {
            let model_path = models_dir.join("model.onnx");
            let tokenizer_path = models_dir.join("tokenizer.json");

            if model_path.exists() && tokenizer_path.exists() {
                info!("Loading model 'default' from flat layout");
                let model = OnnxModel::load(&model_path, &tokenizer_path, use_qnn, pool_size)?;
                loaded.insert("default".to_string(), model);
            }
        }

        if loaded.is_empty() {
            anyhow::bail!(
                "No models found. Place model subdirectories in {} \
                 (each containing model.onnx and tokenizer.json).",
                models_dir.display()
            );
        }

        // Default: use DEFAULT_MODEL env var if set, otherwise first alphabetically
        let default_model = std::env::var("DEFAULT_MODEL").unwrap_or_else(|_| {
            let mut names: Vec<_> = loaded.keys().cloned().collect();
            names.sort();
            names[0].clone()
        });

        if !loaded.contains_key(&default_model) {
            anyhow::bail!(
                "Default model '{default_model}' not found. Available: {:?}",
                loaded.keys().collect::<Vec<_>>()
            );
        }

        info!(
            "Loaded {} model(s): {:?} (default: '{}')",
            loaded.len(),
            loaded.keys().collect::<Vec<_>>(),
            default_model
        );

        // Publish models atomically
        {
            let mut models = self.models.write().unwrap();
            *models = loaded;
        }
        {
            let mut dm = self.default_model.write().unwrap();
            *dm = default_model;
        }
        self.ready.store(true, Ordering::Release);
        info!("All models ready — accepting inference requests");

        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<Arc<OnnxModel>> {
        let models = self.models.read().unwrap();
        models.get(name).cloned()
    }

    pub fn default_model_name(&self) -> String {
        self.default_model.read().unwrap().clone()
    }

    pub fn model_names(&self) -> Vec<String> {
        let models = self.models.read().unwrap();
        let mut names: Vec<_> = models.keys().cloned().collect();
        names.sort();
        names
    }
}
